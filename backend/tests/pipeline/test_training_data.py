import geopandas as gpd
import pandas as pd
import pytest
import shapely
from shapely.geometry import Point, box

from fundoscope.ingest import from_utm, to_utm
from fundoscope.ml import FEATURE_COLUMNS, MAX_PIXELS_PER_SAMPLE
from fundoscope.pipeline import (
    DONE,
    FAILED,
    LOG_FILE,
    NO_CLEAR_PIXELS,
    TOO_NARROW,
    extract_training_data,
    extraction_tasks,
    pending,
)
from fundoscope.sentinel import CreditBudget

CENTER = to_utm(Point(-75.73, -14.07), 32718)  # public farmland near Ica


def field(index: int, width: float = 200.0, height: float = 200.0):
    x = CENTER.x + index * 500
    return from_utm(box(x, CENTER.y, x + width, CENTER.y + height), 32718)


def labels(*rows: dict) -> gpd.GeoDataFrame:
    defaults = {"region": "ICA", "crop": "grape", "source": "photo", "evidence": "SV ..."}
    return gpd.GeoDataFrame([defaults | row for row in rows], geometry="geometry", crs=4326)


def two_labels() -> gpd.GeoDataFrame:
    """ICA-0001 over two campaigns, ICA-0002 over one, and a field-verified parcel."""
    return labels(
        {"parcel_id": "ICA-0001", "from_campaign": "2023/24", "to_campaign": "2024/25",
         "geometry": field(0)},
        {"parcel_id": "ICA-0002", "from_campaign": "2024/25", "to_campaign": "2024/25",
         "geometry": field(1)},
        {"parcel_id": "ICA-0003", "from_campaign": "2024/25", "to_campaign": "2024/25",
         "source": "field", "geometry": field(2)},
    )  # fmt: skip


def test_one_task_per_photo_label_and_campaign():
    tasks = extraction_tasks(two_labels())

    assert [(t.parcel_id, t.campaign) for t in tasks] == [
        ("ICA-0001", "2023/24"),
        ("ICA-0001", "2024/25"),
        ("ICA-0002", "2024/25"),
    ]
    assert tasks[0].epsg == 32718
    assert tasks[0].geometry_utm.area == pytest.approx(40_000, rel=1e-3)  # meters
    assert tasks[0].file_name == "ICA-0001_2023-24.csv"


def test_saves_up_to_50_interior_pixels_per_parcel_and_campaign(tmp_path, make_campaign_composites):
    fake = make_campaign_composites()
    tasks = extraction_tasks(two_labels())

    extraction = extract_training_data(tasks, tmp_path, CreditBudget(1_000), fetch=fake)

    assert [r.status for r in extraction.results] == [DONE] * 3
    assert extraction.credits == 21.0
    assert extraction.remaining == 0
    saved = pd.read_csv(tmp_path / "pixels" / "ICA-0001_2023-24.csv")
    assert list(saved.columns) == ["parcel_id", "campaign", "x", "y", *FEATURE_COLUMNS]
    assert len(saved) == MAX_PIXELS_PER_SAMPLE
    assert set(saved["campaign"]) == {"2023/24"}
    interior = tasks[0].geometry_utm.buffer(-15)
    assert shapely.contains_xy(interior, saved["x"], saved["y"]).all()


def test_every_request_is_logged_with_its_credits(tmp_path, make_campaign_composites):
    tasks = extraction_tasks(two_labels())
    extract_training_data(tasks, tmp_path, CreditBudget(1_000), fetch=make_campaign_composites())

    log = pd.read_csv(tmp_path / LOG_FILE)

    assert len(log) == 3
    assert set(log["status"]) == {DONE}
    assert log["credits"].sum() == 21.0


def test_a_second_run_downloads_nothing(tmp_path, make_campaign_composites):
    fake = make_campaign_composites()
    tasks = extraction_tasks(two_labels())
    extract_training_data(tasks, tmp_path, CreditBudget(1_000), fetch=fake)

    again = extract_training_data(tasks, tmp_path, CreditBudget(1_000), fetch=fake)

    assert len(fake.calls) == 3
    assert (again.already_done, again.results, again.remaining) == (3, [], 0)
    assert pending(tasks, tmp_path) == []


def test_the_cap_stops_the_run_and_the_next_run_finishes(tmp_path, make_campaign_composites):
    fake = make_campaign_composites()
    tasks = extraction_tasks(two_labels())

    first = extract_training_data(tasks, tmp_path, CreditBudget(15), fetch=fake, workers=1)

    assert first.stopped_at_cap
    assert (first.count(DONE), first.remaining, first.credits) == (1, 2, 7.0)

    second = extract_training_data(tasks, tmp_path, CreditBudget(100), fetch=fake)

    assert not second.stopped_at_cap
    assert (second.already_done, second.count(DONE), second.remaining) == (1, 2, 0)
    assert len(fake.calls) == 3  # nothing downloaded twice


def test_failures_are_logged_and_retried_by_the_next_run(tmp_path, make_campaign_composites):
    fake = make_campaign_composites(fail={"2023-08-01": 1})
    tasks = extraction_tasks(two_labels())

    first = extract_training_data(tasks, tmp_path, CreditBudget(1_000), fetch=fake)

    failed = [r for r in first.results if r.status == FAILED]
    assert [r.task.campaign for r in failed] == ["2023/24"]
    assert "simulated openEO failure" in failed[0].error
    assert first.remaining == 1
    assert not (tmp_path / "pixels" / "ICA-0001_2023-24.csv").exists()
    assert FAILED in set(pd.read_csv(tmp_path / LOG_FILE)["status"])

    second = extract_training_data(tasks, tmp_path, CreditBudget(1_000), fetch=fake)
    assert (second.count(DONE), second.remaining) == (1, 0)


def test_three_failures_in_a_row_stop_the_run(tmp_path, make_campaign_composites):
    fake = make_campaign_composites(fail={f"{year}-08-01": 1 for year in range(2019, 2025)})
    rows = labels(
        {"parcel_id": "ICA-0001", "from_campaign": "2019/20", "to_campaign": "2024/25",
         "geometry": field(0)},
    )  # fmt: skip
    tasks = extraction_tasks(rows)

    extraction = extract_training_data(tasks, tmp_path, CreditBudget(1_000), fetch=fake, workers=1)

    assert extraction.stopped_by_failures
    assert (extraction.count(FAILED), extraction.remaining) == (3, 6)


def test_a_narrow_parcel_is_not_downloaded(tmp_path, make_campaign_composites):
    fake = make_campaign_composites()
    rows = labels(
        {"parcel_id": "ICA-0001", "from_campaign": "2024/25", "to_campaign": "2024/25",
         "geometry": field(0, width=25.0, height=300.0)},
    )  # fmt: skip

    extraction = extract_training_data(
        extraction_tasks(rows), tmp_path, CreditBudget(1_000), fetch=fake
    )

    assert [r.status for r in extraction.results] == [TOO_NARROW]
    assert fake.calls == []
    assert extraction.remaining == 0


def test_a_cloudy_campaign_is_saved_empty_and_not_downloaded_again(
    tmp_path, make_campaign_composites
):
    fake = make_campaign_composites(cloudy={"2024-08-01"})
    tasks = extraction_tasks(two_labels())

    extraction = extract_training_data(tasks, tmp_path, CreditBudget(1_000), fetch=fake)

    assert extraction.count(NO_CLEAR_PIXELS) == 2
    assert pd.read_csv(tmp_path / "pixels" / "ICA-0002_2024-25.csv").empty
    assert pending(tasks, tmp_path) == []


def test_a_month_missing_from_the_result_is_filled(tmp_path, make_campaign_composites):
    fake = make_campaign_composites(drop_month=3)
    tasks = extraction_tasks(two_labels())[:1]

    extraction = extract_training_data(tasks, tmp_path, CreditBudget(1_000), fetch=fake)

    assert extraction.results[0].pixels == MAX_PIXELS_PER_SAMPLE


def test_unusable_composites_fail_but_count_their_credits(tmp_path, make_campaign_composites):
    fake = make_campaign_composites(bands=["B02", "B03", "B04", "B08"])
    tasks = extraction_tasks(two_labels())[:1]

    result = extract_training_data(tasks, tmp_path, CreditBudget(1_000), fetch=fake).results[0]

    assert (result.status, result.credits) == (FAILED, 7.0)
    assert "missing the bands" in result.error
