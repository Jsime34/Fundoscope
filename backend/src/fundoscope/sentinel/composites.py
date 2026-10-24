"""Monthly cloud-free composites from openEO (feeds % active area, `ml`, and `valley`).

For a box in UTM, openEO on CDSE takes every Sentinel-2 L2A image in the period, masks
clouds, shadows, and cirrus with the Scene Classification Layer (the same classes as the
daily statistics), and keeps the median of the clear values per pixel and calendar month.
The result is a stack (month, band, y, x) of surface reflectance 0..1 on a 10 m grid, with
NaN where a pixel had no clear view that month.

openEO returns reflectance x 10,000 with the processing-baseline offset already removed.
Small areas (parcels) use a synchronous request; whole valleys will need batch jobs. Each
response reports its cost in openEO credits (free tier: 10,000 per month). openEO accepts
the same OAuth client as Sentinel Hub (client-credentials login, no browser needed).

Rate limits: each worker thread reuses one authenticated connection (re-logging in every
request triggered "429 Too Many Requests"), and a 429 answer is retried after a pause.
"""

import functools
import io
import operator
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date, timedelta

import numpy as np
import openeo
import pandas as pd
import requests
import shapely
import xarray as xr
from openeo import DataCube
from openeo.rest import OpenEoRestError
from shapely.geometry.base import BaseGeometry

from fundoscope.config import get_settings
from fundoscope.sentinel.evalscripts import NOT_CLEAR_SCL_CLASSES
from fundoscope.sentinel.statistical import SentinelError

OPENEO_URL = "openeo.dataspace.copernicus.eu"
COLLECTION = "SENTINEL2_L2A"
COMPOSITE_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
RESOLUTION_M = 10
REFLECTANCE_SCALE = 10_000
COST_HEADER = "OpenEO-Costs-experimental"
SYNC_TIMEOUT_S = 900
CONNECTION_MAX_AGE_S = 300  # re-authenticate after 5 minutes
RATE_LIMIT_WAITS_S = (5, 15, 45)  # pauses before retrying a "429" (openEO asks for ~3 s)

# (process graph) -> (NetCDF bytes, credits spent). Replaced by a fake in tests.
Sender = Callable[[dict], tuple[bytes, float]]


@dataclass(frozen=True)
class MonthlyComposites:
    """Monthly surface reflectance for a box.

    Attributes:
        data: DataArray with dims (month, band, y, x); coordinates are the first day of each
            month, band names, and UTM pixel-center coordinates in meters.
        epsg: UTM EPSG code of the x/y coordinates.
        credits: openEO credits the request cost.
    """

    data: xr.DataArray
    epsg: int
    credits: float

    @property
    def months(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.data["month"].to_numpy(), name="month")

    def band(self, name: str) -> np.ndarray:
        """One band as a (month, y, x) array of reflectance."""
        return self.data.sel(band=name).to_numpy()

    def inside(self, geometry_utm: BaseGeometry) -> np.ndarray:
        """(y, x) boolean mask: pixels whose center lies inside the geometry (UTM meters)."""
        xx, yy = np.meshgrid(self.data["x"].to_numpy(), self.data["y"].to_numpy())
        return shapely.contains_xy(geometry_utm, xx, yy)

    def masked_to(self, geometry_utm: BaseGeometry) -> "MonthlyComposites":
        """The same composites with every pixel outside the geometry set to NaN."""
        keep = xr.DataArray(self.inside(geometry_utm), dims=("y", "x"))
        return replace(self, data=self.data.where(keep))


def fetch_monthly_composites(
    geometry_utm: BaseGeometry,
    epsg: int,
    start: date,
    end: date,
    *,
    bands: tuple[str, ...] = COMPOSITE_BANDS,
    send: Sender | None = None,
) -> MonthlyComposites:
    """Monthly composites over the bounding box of `geometry_utm`, from `start` to `end`.

    Months are calendar months clipped to the period: pass campaign bounds (Aug 1 - Jul 31)
    to get the 12 months of a campaign. Pixels outside the geometry are NOT masked here; use
    `masked_to` (neighbors and valleys want the whole box).
    """
    if end < start:
        raise ValueError(f"The end date {end} is before the start date {start}")
    graph = build_graph(geometry_utm.bounds, epsg, start, end, bands)
    content, credits = (send or _send_sync)(graph)
    return MonthlyComposites(parse_netcdf(content, bands), epsg, credits)


def build_graph(
    bounds_utm: tuple[float, float, float, float],
    epsg: int,
    start: date,
    end: date,
    bands: tuple[str, ...] = COMPOSITE_BANDS,
) -> dict:
    """The openEO process graph (built offline, without a connection)."""
    west, south, east, north = bounds_utm
    pad = RESOLUTION_M  # one extra pixel on each side so edge pixels are complete
    cube = DataCube.load_collection(
        COLLECTION,
        connection=None,
        spatial_extent={
            "west": west - pad,
            "south": south - pad,
            "east": east + pad,
            "north": north + pad,
            "crs": epsg,
        },
        # openEO time ranges exclude their end, so add a day to include `end`.
        temporal_extent=[start.isoformat(), (end + timedelta(days=1)).isoformat()],
        bands=[*bands, "SCL"],
        fetch_metadata=False,
    )
    scl = cube.band("SCL")
    not_clear = functools.reduce(operator.or_, (scl == c for c in NOT_CLEAR_SCL_CLASSES))
    monthly = (
        cube.filter_bands(list(bands))
        .mask(not_clear)
        .aggregate_temporal_period(period="month", reducer="median")
        .resample_spatial(resolution=RESOLUTION_M, projection=epsg)
    )
    return monthly.save_result(format="netCDF").flat_graph()


def parse_netcdf(content: bytes, bands: tuple[str, ...] = COMPOSITE_BANDS) -> xr.DataArray:
    """Turn openEO's NetCDF (one variable per band) into a (month, band, y, x) reflectance stack."""
    try:
        with xr.open_dataset(io.BytesIO(content), engine="h5netcdf") as dataset:
            dataset = dataset.load()
    except (OSError, ValueError) as exc:
        raise SentinelError(f"openEO returned a file that is not readable NetCDF: {exc}") from exc
    missing = [b for b in bands if b not in dataset]
    if missing:
        raise SentinelError(f"openEO's result is missing the bands: {', '.join(missing)}")
    stack = dataset[list(bands)].to_dataarray(dim="band").transpose("t", "band", "y", "x")
    return (stack.astype(np.float32) / REFLECTANCE_SCALE).rename(t="month")


def openeo_connection() -> openeo.Connection:
    """An authenticated openEO connection using our CDSE OAuth client (no browser needed)."""
    settings = get_settings()
    client_secret = settings.cdse_client_secret.get_secret_value()
    if not settings.cdse_client_id or not client_secret:
        raise SentinelError("Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET in .env to use openEO.")
    connection = openeo.connect(OPENEO_URL)
    connection.authenticate_oidc_client_credentials(
        client_id=settings.cdse_client_id, client_secret=client_secret
    )
    return connection


_local = threading.local()


def _thread_connection() -> openeo.Connection:
    """One authenticated connection per worker thread, renewed every few minutes."""
    cached = getattr(_local, "connection", None)
    if cached is None or time.monotonic() - cached[1] > CONNECTION_MAX_AGE_S:
        _local.connection = (openeo_connection(), time.monotonic())
    return _local.connection[0]


def _send_sync(
    graph: dict,
    connection: openeo.Connection | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[bytes, float]:
    """POST the graph for synchronous processing; retry on 429. Errors become SentinelError."""
    connection = connection or _thread_connection()
    for wait in (*RATE_LIMIT_WAITS_S, None):
        try:
            response = connection.post(
                "/result",
                json={"process": {"process_graph": graph}},
                expected_status=200,
                timeout=SYNC_TIMEOUT_S,
            )
        except OpenEoRestError as exc:
            if getattr(exc, "http_status_code", None) == 429 and wait is not None:
                sleep(wait)
                continue
            raise SentinelError(f"openEO request failed: {exc}") from exc
        except requests.RequestException as exc:
            raise SentinelError(f"Could not reach openEO: {exc}") from exc
        return response.content, float(response.headers.get(COST_HEADER, 0.0))
