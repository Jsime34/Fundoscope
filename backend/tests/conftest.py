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


class FakeCampaignComposites:
    """Replaces `fetch_monthly_composites` for training data: every band, a green crop everywhere.

    `fail` maps a campaign's first day ("2024-08-01") to how many requests fail before one
    succeeds; `cloudy` campaigns have no clear pixel; `bands` overrides the bands returned;
    `drop_month` leaves that month (0 = August) out of the result; parcels for which
    `bare(geometry_utm)` is true stay bare soil all year (NDVI about 0.09).
    """

    def __init__(self, fail=None, cloudy=(), bands=None, drop_month=None, bare=None):
        self.calls: list[tuple] = []
        self.fail = dict(fail or {})
        self.cloudy = set(cloudy)
        self.bands = bands
        self.drop_month = drop_month
        self.bare = bare

    def __call__(self, geometry_utm, epsg, start, end, bands):
        self.calls.append((geometry_utm.bounds, start))
        if self.fail.get(start.isoformat(), 0) > 0:
            self.fail[start.isoformat()] -= 1
            raise SentinelError("simulated openEO failure")
        bands = list(self.bands or bands)
        minx, miny, maxx, maxy = geometry_utm.bounds
        x = np.arange(minx + 5, maxx, 10.0)
        y = np.arange(maxy - 5, miny, -10.0)
        months = pd.date_range(start, end, freq="MS")
        if self.drop_month is not None:
            months = months.delete(self.drop_month)
        data = np.full((len(months), len(bands), len(y), len(x)), 0.1)
        data[:, bands.index("B08")] = 0.45  # NDVI about 0.64: a growing crop
        if self.bare and self.bare(geometry_utm):
            data[:, bands.index("B08")] = 0.12
        if start.isoformat() in self.cloudy:
            data[:] = np.nan
        stack = xr.DataArray(
            data,
            dims=("month", "band", "y", "x"),
            coords={"month": months, "band": bands, "y": y, "x": x},
        )
        return MonthlyComposites(stack, epsg, credits=7.0)


@pytest.fixture
def make_campaign_composites() -> type[FakeCampaignComposites]:
    return FakeCampaignComposites


TRAINING_LINK = "https://maps.app.goo.gl/abc"


def ready_labels(narrow: bool = False):
    """150 photo labels (135 grape, 15 fallow) from 3 regions that pass `labels check`, plus
    one field-verified label. With `narrow`, the first grape parcel is too narrow to sample.
    """
    import geopandas as gpd

    center = to_utm(Point(-75.73, -14.07), 32718)
    regions = ["ARE", "ICA", "LLI"]

    def square(i, width=100.0):
        x = center.x + i * 500
        return from_utm(box(x, center.y, x + width, center.y + 250), 32718)

    rows = [
        {
            "parcel_id": f"{regions[i % 3]}-{i + 1:04d}",
            "region": regions[i % 3],
            "crop": "grape" if i < 135 else "fallow",
            "from_campaign": "2024/25",
            "to_campaign": "2024/25",
            "source": "photo",
            "confidence": "high",
            "evidence": f"SV 2024-08 {'mature' if i < 135 else 'bare'} {TRAINING_LINK}",
            "notes": None,
            "geometry": square(i, 25.0 if narrow and i == 0 else 100.0),
        }
        for i in range(150)
    ]
    rows.append(rows[1] | {"parcel_id": "ICA-0999", "source": "field", "geometry": square(999)})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)


@pytest.fixture(scope="session")
def training_inputs(tmp_path_factory):
    """Labels ready for training and their saved pixels: grape green, fallow bare all year.

    Built once per test session (150 fake downloads). Returns (labels, features folder);
    the first grape parcel is too narrow, so it has no pixels.
    """
    from fundoscope.pipeline import extract_training_data, extraction_tasks
    from fundoscope.sentinel import CreditBudget

    labels = ready_labels(narrow=True)
    first_fallow_x = to_utm(Point(-75.73, -14.07), 32718).x + 135 * 500 - 250
    fake = FakeCampaignComposites(bare=lambda geometry: geometry.bounds[0] > first_fallow_x)
    features = tmp_path_factory.mktemp("features")
    extract_training_data(extraction_tasks(labels), features, CreditBudget(10_000), fetch=fake)
    return labels, features
