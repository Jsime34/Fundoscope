"""ESA WorldCover 2021 land cover from openEO (feeds the valley's farmland mask, FR-14).

WorldCover maps the world at 10 m in 11 classes, for example 40 = cropland, 50 = built-up,
80 = permanent water, and 0 = no data. A valley reads it once, from openEO on CDSE (collection
ESA_WORLDCOVER_10M_2021_V2), resampled by nearest neighbor to the valley's 10 m UTM grid.
"""

import io
from dataclasses import dataclass

import numpy as np
import xarray as xr
from openeo import DataCube
from shapely.geometry.base import BaseGeometry

from fundoscope.sentinel.composites import RESOLUTION_M, Sender, send_sync
from fundoscope.sentinel.statistical import SentinelError

LAND_COVER_COLLECTION = "ESA_WORLDCOVER_10M_2021_V2"
LAND_COVER_BAND = "MAP"
LAND_COVER_YEAR = 2021
NO_DATA_CLASS = 0


@dataclass(frozen=True)
class LandCover:
    """WorldCover classes over a box.

    Attributes:
        data: (y, x) uint8 classes, with UTM pixel-center coordinates in meters.
        epsg: UTM EPSG code of the coordinates.
        credits: openEO credits the request cost.
    """

    data: xr.DataArray
    epsg: int
    credits: float


def fetch_land_cover(
    geometry_utm: BaseGeometry, epsg: int, *, send: Sender | None = None
) -> LandCover:
    """WorldCover 2021 over the bounding box of `geometry_utm` (one synchronous request)."""
    graph = build_land_cover_graph(geometry_utm.bounds, epsg)
    content, credits = (send or send_sync)(graph)
    return LandCover(parse_land_cover(content), epsg, credits)


def build_land_cover_graph(bounds_utm: tuple[float, float, float, float], epsg: int) -> dict:
    """The openEO process graph (built offline, without a connection)."""
    west, south, east, north = bounds_utm
    pad = RESOLUTION_M  # one extra pixel on each side, as for the composites
    cube = DataCube.load_collection(
        LAND_COVER_COLLECTION,
        connection=None,
        spatial_extent={
            "west": west - pad,
            "south": south - pad,
            "east": east + pad,
            "north": north + pad,
            "crs": epsg,
        },
        temporal_extent=[f"{LAND_COVER_YEAR}-01-01", f"{LAND_COVER_YEAR + 1}-01-01"],
        bands=[LAND_COVER_BAND],
        fetch_metadata=False,
    )
    cover = cube.max_time().resample_spatial(
        resolution=RESOLUTION_M, projection=epsg, method="near"
    )
    return cover.save_result(format="netCDF").flat_graph()


def parse_land_cover(content: bytes) -> xr.DataArray:
    """Turn openEO's NetCDF into a (y, x) uint8 array of classes (no data = 0)."""
    try:
        with xr.open_dataset(io.BytesIO(content), engine="h5netcdf") as dataset:
            dataset = dataset.load()
    except (OSError, ValueError) as exc:
        raise SentinelError(f"openEO returned a file that is not readable NetCDF: {exc}") from exc
    if LAND_COVER_BAND not in dataset:
        raise SentinelError(f"openEO's land cover has no {LAND_COVER_BAND} band.")
    cover = dataset[LAND_COVER_BAND]
    if "t" in cover.dims:
        cover = cover.isel(t=0, drop=True)
    cover = cover.transpose("y", "x").fillna(NO_DATA_CLASS)
    return cover.astype(np.uint8).rename(LAND_COVER_BAND)
