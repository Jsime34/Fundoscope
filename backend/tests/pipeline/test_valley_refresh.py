"""Tests for the valley refresh: a small synthetic valley near Ica (public area), no network.

The fakes (tests/conftest.py) make columns 0-1 desert, columns 2-3 a vineyard, and columns 4-7
fields sown every November; one vineyard pixel is built-up and one is cloudy in August and
September 2026.
"""

import json
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box, mapping

from fundoscope.db import MemoryValleyStore
from fundoscope.ingest import ParcelError, from_utm
from fundoscope.pipeline.valley_refresh import (
    LAND_COVER,
    LOG_FILE,
    Request,
    on_grid,
    plan_refresh,
    refresh_valley,
)
from fundoscope.pipeline.valley_store import (
    ValleyError,
    load_month,
    load_valley,
    peak_ndvi,
    save_month,
    valley_dir,
)
from fundoscope.sentinel import COMPOSITE_BANDS, CreditBudget, SentinelError
from fundoscope.valley import NO_CLEAR_IMAGES, PLANTED, Tile
from fundoscope.valley.season import MonthKey

TODAY = date(2026, 11, 24)  # images up to November 22
RUN_AT = datetime(2026, 11, 24, 9, 30)


def refresh(small_valley, folder, models, fakes, today=TODAY, cap=1_000.0, store=None, **options):
    plan = plan_refresh(small_valley, folder, today)
    return refresh_valley(
        plan,
        folder,
        models,
        CreditBudget(cap),
        store=store or MemoryValleyStore(),
        now=lambda: RUN_AT,
        **fakes,
        **options,
    )


@pytest.fixture
def store():
    return MemoryValleyStore()


def hectares(result) -> dict[str, float]:
    return {area.name: round(area.hectares, 2) for area in result.areas if area.pixels}


# --- the plan -------------------------------------------------------------------------------


def test_the_first_plan_downloads_three_campaigns_and_the_land_cover(small_valley, tmp_path):
    plan = plan_refresh(small_valley, tmp_path, TODAY)

    assert plan.needs_land_cover
    assert plan.downloads == 13  # 6 requests per tile, and the land cover
    assert [request.name for request in plan.requests[:6]] == [
        "r0c0 2024-08 to 2025-01",
        "r0c0 2025-02 to 2025-07",
        "r0c0 2025-08-01 to 2025-11-22",
        "r0c0 2025-11 to 2026-04",
        "r0c0 2026-05 to 2026-10",
        "r0c0 2026-11-01 to 2026-11-22",
    ]
    first = plan.requests[0]
    assert (first.start, first.end) == (date(2024, 8, 1), date(2025, 1, 31))


# --- a full refresh -------------------------------------------------------------------------


@pytest.fixture
def first_run(small_valley, tmp_path, valley_models, valley_fakes, store):
    result = refresh(small_valley, tmp_path, valley_models, valley_fakes, store=store)
    return result, tmp_path, valley_fakes


def test_a_refresh_downloads_everything_then_counts(first_run):
    result, _, fakes = first_run

    assert len(result.downloads) == 13 and not result.failed
    assert result.missing == 0
    assert result.credits == 91.0
    assert fakes["fetch_cover"].calls == 1
    assert result.farmland_ha == pytest.approx(0.23)  # 7 vineyard + 16 sown pixels


def test_the_season_in_progress_counts_sown_fields_as_planted(first_run):
    result, _, _ = first_run
    current, same_date, previous, before = result.counts

    assert current.count.key == "2026-27_to_2026-11-22"
    assert hectares(current) == {"grape": 0.06, PLANTED: 0.16, NO_CLEAR_IMAGES: 0.01}
    assert current.model == "season/rf_v1_to_nov.joblib"
    assert same_date.count.key == "2025-26_to_2025-11-22"
    assert hectares(same_date) == {"grape": 0.07, PLANTED: 0.16}
    assert (previous.count.key, before.count.key) == ("2025-26", "2024-25")
    assert hectares(previous) == hectares(before) == {"grape": 0.07, "other annual crops": 0.16}
    assert previous.model == "rf_v1.joblib"


def test_every_count_adds_up_to_the_farmland(first_run):
    result, _, _ = first_run

    for count in result.counts:
        assert sum(area.hectares for area in count.areas) == pytest.approx(result.farmland_ha)


def test_model_classes_get_ranges_and_rule_classes_do_not(first_run):
    result, _, _ = first_run
    areas = {area.name: area for area in result.counts[0].areas}

    grape = areas["grape"]
    assert grape.low <= grape.adjusted <= grape.high
    assert areas[PLANTED].adjusted is None


def test_the_run_is_saved_with_its_area_maps_and_costs(first_run, store, small_valley):
    result, folder, _ = first_run
    run = result.run

    assert store.latest_run("demo") == run
    inside = from_utm(small_valley.area_utm.centroid, small_valley.epsg)  # WGS84
    assert store.latest_run_at(inside) == run
    assert (run.valley, run.run_at, run.as_of, run.credits, run.downloads) == (
        "demo",
        RUN_AT,
        date(2026, 11, 22),
        91.0,
        13,
    )
    assert run.farmland_ha == 0.23
    assert run.newest_image == date(2026, 11, 21)  # the day before the end of the first count
    assert run.model_version == "rf_v1"
    assert run.model_trained_at == datetime(2026, 11, 23, 10, 0).astimezone()
    assert run.colors[PLANTED] == "#008300" and len(run.corners) == 4
    current = run.counts[0]
    assert (current["campaign"], current["months"], current["in_progress"]) == ("2026/27", 4, True)
    assert current["map"] == "maps/2026-27_to_2026-11-22.nc"
    planted = next(item for item in current["classes"] if item["name"] == PLANTED)
    assert planted == {
        "name": PLANTED,
        "pixels": 16,
        "hectares": 0.16,
        "adjusted": None,
        "low": None,
        "high": None,
    }
    log = pd.read_csv(folder / LOG_FILE)
    assert len(log) == 13 and log["credits"].sum() == 91.0
    assert LAND_COVER in set(log["request"])


def test_the_class_map_covers_the_valley(first_run):
    result, folder, _ = first_run

    with xr.open_dataset(folder / result.counts[0].map_file, engine="h5netcdf") as saved:
        codes = saved["crop_class"].to_numpy()
        classes = json.loads(saved["crop_class"].attrs["classes"])
        assert "spatial_ref" in saved.variables
    assert codes.shape == (4, 8)
    assert classes[codes[0, 3] - 1] == "grape"
    assert classes[codes[0, 5] - 1] == PLANTED
    assert classes[codes[1, 3] - 1] == NO_CLEAR_IMAGES
    assert codes[0, 2] == 0  # built-up
    assert (codes[:, :2] == 0).all()  # desert


def test_a_second_refresh_downloads_nothing(first_run, small_valley, valley_models):
    first, folder, fakes = first_run
    calls = len(fakes["fetch"].calls)

    again = refresh(small_valley, folder, valley_models, fakes)

    assert again.plan.downloads == 0 and again.downloads == []
    assert len(fakes["fetch"].calls) == calls
    assert [hectares(count) for count in again.counts] == [hectares(c) for c in first.counts]


def test_next_week_downloads_only_the_new_days(first_run, small_valley, valley_models):
    _, folder, fakes = first_run

    plan = plan_refresh(small_valley, folder, date(2026, 12, 1))  # images up to November 29
    assert [request.name for request in plan.requests] == [
        "r0c0 2025-11-01 to 2025-11-29",
        "r0c0 2026-11-01 to 2026-11-29",
        "r0c1 2025-11-01 to 2025-11-29",
        "r0c1 2026-11-01 to 2026-11-29",
    ]
    later = refresh(small_valley, folder, valley_models, fakes, today=date(2026, 12, 1))

    assert later.counts[0].count.key == "2026-27_to_2026-11-29"
    tile = folder / "composites" / "r0c0"
    assert not (tile / "2026-11-22.npz").exists()  # replaced by the newer cut
    assert (tile / "2026-11-29.npz").exists() and (tile / "2025-11.npz").exists()


# --- stopping and resuming ------------------------------------------------------------------


def test_the_cap_stops_the_downloads_and_nothing_is_counted(
    small_valley, tmp_path, valley_models, valley_fakes, store
):
    stopped = refresh(
        small_valley, tmp_path, valley_models, valley_fakes, cap=20, store=store, workers=1
    )

    assert stopped.stopped_at_cap
    assert (len(stopped.downloads), stopped.missing) == (2, 11)
    assert (stopped.counts, stopped.run) == ([], None)
    assert store.latest_run("demo") is None

    finished = refresh(small_valley, tmp_path, valley_models, valley_fakes, store=store)

    assert len(finished.downloads) == 11 and store.latest_run("demo") == finished.run


def test_three_failures_in_a_row_stop_the_run(small_valley, tmp_path, valley_models, valley_fakes):
    valley_fakes["fetch"].fail = True

    stopped = refresh(small_valley, tmp_path, valley_models, valley_fakes, workers=1)

    assert stopped.stopped_by_failures
    assert [item.name for item in stopped.failed][0] == "r0c0 2024-08 to 2025-01"
    assert stopped.failed[0].error == "simulated openEO failure"
    assert len(stopped.downloads) == 4  # the land cover worked, then 3 failures
    assert stopped.counts == []


def test_pixels_off_the_grid_fail_the_request(small_valley, tmp_path, valley_models, valley_fakes):
    valley_fakes["fetch"].offset = 5.0

    stopped = refresh(small_valley, tmp_path, valley_models, valley_fakes, workers=1)

    assert "off the 10 m grid of tile r0c0" in stopped.failed[0].error
    assert stopped.failed[0].credits == 7.0  # openEO charged for it all the same


def test_a_land_cover_without_pixels_fails(small_valley, tmp_path, valley_models, valley_fakes):
    valley_fakes["fetch_cover"].empty = True

    stopped = refresh(small_valley, tmp_path, valley_models, valley_fakes, workers=1)

    assert stopped.failed[0].name == "land cover"
    assert stopped.failed[0].error == "openEO returned no pixels for tile valley."
    assert stopped.failed[0].credits == 7.0


def test_a_catalog_failure_leaves_the_newest_image_unknown(
    small_valley, tmp_path, valley_models, valley_fakes
):
    valley_fakes["newest_image"].fail = True

    result = refresh(small_valley, tmp_path, valley_models, valley_fakes)

    assert result.newest_image is None and result.run.newest_image is None
    bounds, until = valley_fakes["newest_image"].calls[0]
    assert until == date(2026, 11, 22)
    assert -75.74 < bounds[0] < bounds[2] < -75.72  # the valley in WGS84


def test_the_models_must_be_trained_first(small_valley, tmp_path, valley_fakes):
    with pytest.raises(ValleyError, match="fundoscope model train"):
        refresh(small_valley, tmp_path, tmp_path / "no-models", valley_fakes)

    assert valley_fakes["fetch"].calls == []


# --- the area of interest -------------------------------------------------------------------


def test_the_valley_comes_from_its_geojson(tmp_path):
    from fundoscope.ingest import to_utm

    square = box(-75.731, -14.071, -75.729, -14.069)
    feature = {"type": "Feature", "properties": {"name": "demo"}, "geometry": mapping(square)}
    (tmp_path / "aoi.geojson").write_text(json.dumps(feature), encoding="utf-8")

    demo = load_valley(tmp_path, "demo")

    assert (demo.slug, demo.epsg, len(demo.tiles)) == ("demo", 32718, 1)
    assert demo.area_km2 == pytest.approx(to_utm(square, 32718).area / 1e6)


def test_a_valley_without_an_area_of_interest(tmp_path):
    with pytest.raises(ValleyError, match="draw it and save it as"):
        load_valley(tmp_path, "demo")


def test_an_unreadable_area_of_interest(tmp_path):
    (tmp_path / "aoi.geojson").write_text("not json", encoding="utf-8")

    with pytest.raises(ParcelError):
        load_valley(tmp_path, "demo")


def test_valleys_live_in_the_data_folder():
    assert valley_dir("majes").parts[-2:] == ("valley", "majes")


# --- month files ----------------------------------------------------------------------------


def month_layer(values) -> xr.DataArray:
    data = np.broadcast_to(np.asarray(values, np.float32), (len(COMPOSITE_BANDS), 1, 3)).copy()
    return xr.DataArray(
        data,
        dims=("band", "y", "x"),
        coords={"band": list(COMPOSITE_BANDS), "y": [8_400_005.0], "x": [5.0, 15.0, 25.0]},
    )


def test_a_month_round_trips_through_16_bit_integers(tmp_path):
    save_month(tmp_path / "2025-11.npz", month_layer([0.12345, np.nan, 9.0]))

    back = load_month(tmp_path / "2025-11.npz")

    assert back.dims == ("band", "y", "x") and back.dtype == np.float32
    values = back.sel(band="B04").to_numpy()[0]
    assert values[0] == pytest.approx(0.1235, abs=1e-4)
    assert np.isnan(values[1])
    assert values[2] == pytest.approx(6.5534)  # clipped to the largest value the file holds


def test_the_peak_ndvi_needs_months(tmp_path):
    with pytest.raises(ValueError, match="No months"):
        peak_ndvi(tmp_path, [])


def test_on_grid_drops_the_padding():
    tile = Tile("r0c0", (0, 0, 20, 10))
    data = xr.DataArray(
        np.arange(12.0).reshape(3, 4),
        dims=("y", "x"),
        coords={"y": [15.0, 5.0, -5.0], "x": [-5.0, 5.0, 15.0, 25.0]},
    )

    assert on_grid(data, tile).to_numpy().tolist() == [[5.0, 6.0]]


def test_on_grid_takes_the_nearest_pixel_of_a_shifted_grid():
    tile = Tile("r0c0", (0, 0, 20, 10))
    shifted = xr.DataArray(
        [[1, 2, 3, 4]], dims=("y", "x"), coords={"y": [7.5], "x": [2.25, 12.25, 22.25, 32.25]}
    )

    with pytest.raises(SentinelError, match="off the 10 m grid"):
        on_grid(shifted, tile)
    assert on_grid(shifted, tile, 0, aligned=False).to_numpy().tolist() == [[1, 2]]


def test_request_names():
    tile = Tile("r1c2", (0, 0, 10, 10))
    whole = MonthKey(date(2025, 10, 1), date(2025, 10, 31))
    cut = MonthKey(date(2025, 11, 1), date(2025, 11, 22))

    assert Request(tile, (whole,)).name == "r1c2 2025-10 to 2025-10"
    assert Request(tile, (whole, cut)).name == "r1c2 2025-10-01 to 2025-11-22"


# --- the saved tile grid --------------------------------------------------------------------


def write_aoi(folder, area_utm):
    from fundoscope.ingest import from_utm

    feature = {"type": "Feature", "properties": {}, "geometry": mapping(from_utm(area_utm, 32718))}
    (folder / "aoi.geojson").write_text(json.dumps(feature), encoding="utf-8")


def test_the_first_refresh_saves_its_grid(first_run, small_valley):
    _, folder, _ = first_run

    grid = json.loads((folder / "grid.json").read_text(encoding="utf-8"))

    assert grid == {"epsg": 32718, "bounds": list(small_valley.bounds), "max_tile_m": 40}


def test_a_smaller_area_keeps_the_saved_grid(first_run, small_valley):
    _, folder, _ = first_run
    west, south, east, north = small_valley.bounds
    write_aoi(folder, box(west + 1, south + 13, east - 1, north - 1))  # would snap 10 m higher

    smaller = load_valley(folder, "demo")

    assert smaller.bounds == small_valley.bounds
    assert smaller.tiles == small_valley.tiles
    assert plan_refresh(smaller, folder, TODAY).requests == ()


def test_an_area_beyond_the_saved_grid_is_refused(first_run, small_valley):
    _, folder, _ = first_run
    west, south, east, north = small_valley.bounds
    write_aoi(folder, box(west, south, east + 50, north))

    with pytest.raises(ValleyError, match="goes beyond the tiles already downloaded"):
        load_valley(folder, "demo")


def test_months_from_another_grid_are_refused(first_run, small_valley, valley_models):
    from fundoscope.valley import valley

    _, folder, fakes = first_run
    west, south, east, north = small_valley.bounds
    # The same two tiles by name, 10 m further south.
    moved = valley(
        "demo", box(west, south, east, north - 10), 32718, 40, (west, south - 10, east, north - 10)
    )

    with pytest.raises(ValleyError, match="on another tile grid than r0c0"):
        refresh_valley(
            plan_refresh(moved, folder, TODAY),
            folder,
            valley_models,
            CreditBudget(0),
            store=MemoryValleyStore(),
            **fakes,
        )
