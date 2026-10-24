import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Point, box, mapping

from fundoscope.ingest import from_utm, to_utm
from fundoscope.sentinel import DailyStats, MonthlyComposites, SentinelError


@pytest.fixture
def parcel_geojson() -> bytes:
    """A 300 m x 300 m (9 ha) parcel near Ica, as an uploaded GeoJSON file."""
    center = to_utm(Point(-75.73, -14.07), 32718)
    square = box(center.x - 150, center.y - 150, center.x + 150, center.y + 150)
    return json.dumps(
        {
            "type": "Feature",
            "properties": {"name": "Lote demo"},
            "geometry": mapping(from_utm(square, 32718)),
        }
    ).encode()


class FakeFetch:
    """Replaces `fetch_daily_stats`: a vineyard-like yearly NDVI cycle, no network."""

    def __init__(self):
        self.calls: list[tuple] = []

    def __call__(self, geometry_utm, epsg, start, end):
        self.calls.append((geometry_utm, epsg, start, end))
        index = pd.date_range(start, end, freq="5D", name="date")
        day_of_year = index.dayofyear.to_numpy()
        # Low (0.2) around June, full canopy (0.75) around October.
        ndvi = 0.475 + 0.275 * np.cos(2 * np.pi * (day_of_year - 280) / 365)
        observations = pd.DataFrame(
            {"ndvi": ndvi, "ndwi": 0.15, "ndvi_std": 0.05, "valid_fraction": 1.0, "pixels": 900},
            index=index,
        )
        return DailyStats(observations, processing_units=2.5, requests=2, failed_intervals=0)


@pytest.fixture
def fake_fetch() -> FakeFetch:
    return FakeFetch()


class FakeComposites:
    """Replaces `fetch_monthly_composites`: the western 70 m of the parcel is bare, the rest green.

    `failures` maps a campaign's first day ("2022-08-01") to how many times its request
    should fail before succeeding, to test retries; `cloudy` lists campaigns (by first day)
    with no clear pixel at all.
    """

    def __init__(self, failures: dict[str, int] | None = None, cloudy: set[str] | None = None):
        self.calls: list[tuple] = []
        self.failures = dict(failures or {})
        self.cloudy = set(cloudy or ())  # campaign first days with no clear pixel at all

    def __call__(self, geometry_utm, epsg, start, end, bands):
        self.calls.append((start, end, bands))
        if self.failures.get(start.isoformat(), 0) > 0:
            self.failures[start.isoformat()] -= 1
            raise SentinelError("simulated openEO failure")
        minx, miny, maxx, maxy = geometry_utm.bounds
        x = np.arange(minx + 5, maxx, 10.0)  # 10 m pixel centers
        y = np.arange(maxy - 5, miny, -10.0)
        first_month = pd.Timestamp(start).to_period("M").to_timestamp()
        months = pd.date_range(first_month, end, freq="MS")
        red = np.full((len(months), len(y), len(x)), 0.05)
        nir = np.full_like(red, 0.45)  # NDVI 0.8: a growing crop
        nir[:, :, x < minx + 70] = 0.06  # NDVI ~0.09: bare soil in the western 7 columns
        if start.isoformat() in self.cloudy:
            red[:], nir[:] = np.nan, np.nan
        data = xr.DataArray(
            np.stack([red, nir], axis=1),
            dims=("month", "band", "y", "x"),
            coords={"month": months, "band": list(bands), "y": y, "x": x},
        )
        return MonthlyComposites(data, epsg, credits=7.0)


@pytest.fixture
def fake_composites() -> FakeComposites:
    return FakeComposites()


@pytest.fixture
def make_composites() -> type[FakeComposites]:
    """`make_composites(failures={...})` builds a fake that fails some requests first."""
    return FakeComposites
