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
    with no clear pixel at all. Every requested band is returned (B04 and B08 make the NDVI,
    the others are 0.1), unless `only_bands` limits them.
    """

    def __init__(
        self,
        failures: dict[str, int] | None = None,
        cloudy: set[str] | None = None,
        only_bands: tuple[str, ...] | None = None,
    ):
        self.calls: list[tuple] = []
        self.failures = dict(failures or {})
        self.cloudy = set(cloudy or ())  # campaign first days with no clear pixel at all
        self.only_bands = only_bands

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
        returned = list(self.only_bands or bands)
        layers = {"B04": red, "B08": nir}
        data = xr.DataArray(
            np.stack([layers.get(name, np.full_like(red, 0.1)) for name in returned], axis=1),
            dims=("month", "band", "y", "x"),
            coords={"month": months, "band": returned, "y": y, "x": x},
        )
        return MonthlyComposites(data, epsg, credits=7.0)


class FakeCropModel:
    """Replaces the trained model: a pixel whose NDVI peak passes 0.4 is 90 % grape."""

    classes_ = np.array(["fallow/bare", "grape"])

    def predict_proba(self, features):
        from fundoscope.ml import FEATURE_COLUMNS

        green = features[:, FEATURE_COLUMNS.index("ndvi_peak")] > 0.4
        grape = np.where(green, 0.9, 0.1)
        return np.column_stack([1 - grape, grape])


@pytest.fixture
def fake_crop_model() -> FakeCropModel:
    return FakeCropModel()


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


# --- the valley crop count (FR-14) ------------------------------------------------------------

VALLEY_CLASSES = ["alfalfa", "fallow/bare", "grape", "other annual crops"]
_VALLEY_CORNER = to_utm(Point(-75.73, -14.07), 32718)  # public farmland near Ica
VALLEY_WEST = float(np.floor(_VALLEY_CORNER.x / 10) * 10)
VALLEY_SOUTH = float(np.floor(_VALLEY_CORNER.y / 10) * 10)


def _ndvi_nir(ndvi):
    return 0.05 * (1 + ndvi) / (1 - ndvi)  # with red at 0.05


class FakeValleyComposites:
    """Replaces `fetch_monthly_composites` for a valley 8 columns wide, like openEO (one
    pixel of padding on each side):

    - columns 0-1 are desert (NDVI 0.09), never farmland;
    - columns 2-3 are a vineyard (NDVI 0.8 all year);
    - columns 4-7 are sown every November: bare (0.09) from August to October, then an
      unsure crop (0.5).

    The pixel in row 1 of column 3 is cloudy in August and September 2026. `offset` moves
    the grid off the 10 m lines; `fail` makes every request fail.
    """

    def __init__(self, offset: float = 0.0, fail: bool = False):
        self.calls: list[tuple] = []
        self.offset = offset
        self.fail = fail

    def __call__(self, geometry_utm, epsg, start, end, bands):
        self.calls.append((geometry_utm.bounds, start, end))
        if self.fail:
            raise SentinelError("simulated openEO failure")
        minx, miny, maxx, maxy = geometry_utm.bounds
        x = np.arange(minx - 5, maxx + 10, 10.0) + self.offset
        y = np.arange(maxy + 5, miny - 10, -10.0)
        months = pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS")
        column = np.round((x - VALLEY_WEST - 5) / 10)
        ndvi = np.empty((len(months), len(y), len(x)))
        for i, month in enumerate(months):
            sown = np.where(month.month in (8, 9, 10), 0.09, 0.5)
            ndvi[i] = np.select([column < 2, column < 4], [0.09, 0.8], sown)[None, :]
        data = np.full((len(months), len(bands), len(y), len(x)), 0.1)
        data[:, list(bands).index("B04")] = 0.05
        data[:, list(bands).index("B08")] = _ndvi_nir(ndvi)
        row = np.round((VALLEY_SOUTH + 40 - 5 - y) / 10)
        for i, month in enumerate(months):
            if month in (pd.Timestamp("2026-08-01"), pd.Timestamp("2026-09-01")):
                data[i, :, (row == 1).argmax(), (column == 3).argmax()] = np.nan
        stack = xr.DataArray(
            data,
            dims=("month", "band", "y", "x"),
            coords={"month": months, "band": list(bands), "y": y, "x": x},
        )
        return MonthlyComposites(stack, epsg, credits=7.0)


class FakeLandCover:
    """Replaces `fetch_land_cover`: cropland, with one built-up pixel (row 0, column 2).

    Like the real WorldCover, its 10 m grid is shifted by a few meters from the valley's.
    """

    def __init__(self, empty: bool = False):
        self.calls = 0
        self.empty = empty

    def __call__(self, geometry_utm, epsg):
        from fundoscope.sentinel.landcover import LandCover

        self.calls += 1
        minx, miny, maxx, maxy = geometry_utm.bounds
        x = np.arange(minx - 5, maxx + 10, 10.0) - 2.75
        y = np.arange(maxy + 5, miny - 10, -10.0) + 3.79
        cover = np.full((len(y), len(x)), 40, np.uint8)
        cover[1, 3] = 50  # row 0, column 2, after the padding
        if self.empty:
            x, y, cover = x[:0], y[:0], cover[:0, :0]
        data = xr.DataArray(cover, dims=("y", "x"), coords={"y": y, "x": x})
        return LandCover(data, epsg, credits=7.0)


@pytest.fixture
def small_valley():
    """The valley "demo": 80 m x 40 m near Ica, in two tiles."""
    from fundoscope.valley import valley

    area = box(VALLEY_WEST, VALLEY_SOUTH, VALLEY_WEST + 80, VALLEY_SOUTH + 40)
    return valley("demo", area, 32718, max_tile_m=40)


class FakeCatalog:
    """Replaces `newest_image_date`: the newest image is the day before `until`."""

    def __init__(self, fail: bool = False):
        self.calls: list[tuple] = []
        self.fail = fail

    def __call__(self, bounds_wgs84, until):
        import datetime

        self.calls.append((bounds_wgs84, until))
        if self.fail:
            raise SentinelError("simulated catalog failure")
        return until - datetime.timedelta(days=1)


@pytest.fixture
def valley_fakes():
    """Fresh fakes for one test: `fetch` (composites), `fetch_cover` (land cover), and
    `newest_image` (the catalog)."""
    return {
        "fetch": FakeValleyComposites(),
        "fetch_cover": FakeLandCover(),
        "newest_image": FakeCatalog(),
    }


def _peak_tree(months: int):
    """A real (pickleable) model that looks only at the NDVI peak: above 0.7 is grape,
    0.3 to 0.7 is unsure (other annual crops 40 %), below 0.3 is fallow/bare."""
    from sklearn.tree import DecisionTreeClassifier

    from fundoscope.ml import feature_columns

    peaks = [0.9] * 5 + [0.1] * 5 + [0.5] * 5
    classes = ["grape"] * 5 + ["fallow/bare"] * 5
    classes += ["other annual crops", "other annual crops", "grape", "fallow/bare", "alfalfa"]
    features = np.zeros((len(peaks), len(feature_columns(months))))
    features[:, -6] = peaks  # ndvi_peak
    return DecisionTreeClassifier(random_state=0).fit(features, classes)


@pytest.fixture
def valley_models(tmp_path_factory):
    """A models folder with the full model, the model to November, and their card."""
    import joblib

    from fundoscope.pipeline import CARD_FILE, MODEL_FILE, SEASON_DIR, season_model_file

    folder = tmp_path_factory.mktemp("models")
    (folder / SEASON_DIR).mkdir()
    joblib.dump(_peak_tree(12), folder / MODEL_FILE)
    joblib.dump(_peak_tree(4), folder / SEASON_DIR / season_model_file(4))
    matrix = [[9, 0, 0, 1], [0, 10, 0, 0], [0, 0, 8, 2], [0, 1, 1, 18]]
    confusion = {"labeled_rows_predicted_columns": VALLEY_CLASSES, "matrix": matrix}
    card = {
        "version": "rf_v1",
        "trained_at": "2026-11-23T10:00:00",
        "cross_validation": {"confusion": confusion},
        "season_to_date": [{"months": 4, "confusion": confusion}],
    }
    (folder / CARD_FILE).write_text(json.dumps(card), encoding="utf-8")
    return folder


# --- the neighbor comparison (FR-08) ----------------------------------------------------------

_NEIGHBOR_CORNER = to_utm(Point(-75.70, -14.05), 32718)  # public farmland near Ica
NEIGHBOR_WEST = float(np.floor(_NEIGHBOR_CORNER.x / 10) * 10)
NEIGHBOR_SOUTH = float(np.floor(_NEIGHBOR_CORNER.y / 10) * 10)
# The parcel's NDVI in each campaign: below every neighbor, above every neighbor, in between.
PARCEL_NDVI = {"2024/25": 0.15, "2025/26": 0.9, "2026/27": 0.5}


def parcel_square_geojson(x: float, y: float, size: float = 80.0) -> bytes:
    """A size x size m square parcel with its south-west corner at UTM (x, y), as GeoJSON."""
    square = from_utm(box(x, y, x + size, y + size), 32718)
    return json.dumps(
        {"type": "Feature", "properties": {"name": "Vecino"}, "geometry": mapping(square)}
    ).encode()


@pytest.fixture
def neighbor_valley(tmp_path):
    """A counted valley 400 m x 400 m near Ica, written by hand: (valleys folder, parcel file).

    Every neighbor pixel's NDVI grows from west to east (0.2 + 0.015 per column); the 80 m
    parcel in the middle has `PARCEL_NDVI` of its campaign. Columns 0-1 aren't farmland.
    """
    from fundoscope.pipeline.valley_store import save_grid, save_month
    from fundoscope.sentinel import COMPOSITE_BANDS
    from fundoscope.valley import valley
    from fundoscope.valley.season import Count

    folder = tmp_path / "valleys" / "demo"
    area = box(NEIGHBOR_WEST, NEIGHBOR_SOUTH, NEIGHBOR_WEST + 400, NEIGHBOR_SOUTH + 400)
    demo = valley("demo", area, 32718)
    folder.mkdir(parents=True)
    feature = {
        "type": "Feature",
        "properties": {},
        "geometry": mapping(from_utm(area.buffer(-1, join_style="mitre"), 32718)),
    }
    (folder / "aoi.geojson").write_text(json.dumps(feature), encoding="utf-8")
    save_grid(demo, folder)
    tile = demo.tiles[0]
    columns = np.tile(np.arange(len(tile.x)), (len(tile.y), 1))
    parcel = (
        (columns >= 16)
        & (columns < 24)
        & (np.arange(len(tile.y))[:, None] >= 16)
        & (np.arange(len(tile.y))[:, None] < 24)
    )
    counts = [
        {"campaign": "2026/27", "months": 4, "through": "2026-11-22", "map": "maps/now.nc"},
        {"campaign": "2025/26", "months": 4, "through": "2025-11-22", "map": "maps/then.nc"},
        {"campaign": "2025/26", "months": 12, "through": "2026-07-31", "map": "maps/a.nc"},
        {"campaign": "2024/25", "months": 12, "through": "2025-07-31", "map": "maps/b.nc"},
    ]
    from datetime import date

    for item in counts:
        count = Count(item["campaign"], item["months"], date.fromisoformat(item["through"]))
        for key in count.month_keys():
            values = np.where(parcel, PARCEL_NDVI[count.campaign], 0.2 + 0.015 * columns)
            data = np.full((len(COMPOSITE_BANDS), len(tile.y), len(tile.x)), 0.1)
            data[COMPOSITE_BANDS.index("B04")] = 0.05
            data[COMPOSITE_BANDS.index("B08")] = _ndvi_nir(values)
            layer = xr.DataArray(
                data,
                dims=("band", "y", "x"),
                coords={"band": list(COMPOSITE_BANDS), "y": tile.y, "x": tile.x},
            )
            (folder / "composites" / tile.id).mkdir(parents=True, exist_ok=True)
            save_month(folder / "composites" / tile.id / key.file_name, layer)
    (folder / "maps").mkdir()
    codes = np.where(columns >= 2, 1, 0).astype(np.uint8)
    xr.DataArray(
        codes, dims=("y", "x"), coords={"y": tile.y, "x": tile.x}, name="crop_class"
    ).to_netcdf(folder / "maps" / "now.nc", engine="h5netcdf")
    (folder / "latest.json").write_text(json.dumps({"counts": counts}), encoding="utf-8")
    return folder.parent, parcel_square_geojson(NEIGHBOR_WEST + 160, NEIGHBOR_SOUTH + 160)
