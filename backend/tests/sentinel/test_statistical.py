"""Tests for the Statistical API adapter. No network: a recorded response and a fake sender."""

import copy
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sentinelhub import CRS, BBox, Geometry, SHConfig

from fundoscope.sentinel import (
    DailyStats,
    SentinelError,
    date_chunks,
    fetch_daily_stats,
    parse_response,
)
from fundoscope.sentinel.statistical import build_request

FIXTURE = Path(__file__).parent / "fixtures" / "statistical_ica_2024-01.json"
ICA_BOX = BBox((-75.74, -14.08, -75.73, -14.07), crs=CRS.WGS84).transform(CRS(32718))


@pytest.fixture
def recorded() -> dict:
    """A real response: public 1 km box near Ica, January 2024 (Jan 20 was fully cloudy)."""
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


# --- parse_response ------------------------------------------------------------------------


def test_parse_a_real_response(recorded):
    frame, failed = parse_response(recorded)
    assert failed == 0
    assert list(frame.columns) == ["ndvi", "ndwi", "ndvi_std", "valid_fraction", "pixels"]
    assert isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is None
    assert frame.index[0] == pd.Timestamp("2024-01-05")
    assert len(frame) == 6
    clear = frame.drop(pd.Timestamp("2024-01-20"))
    assert (clear["valid_fraction"] == 1.0).all()
    assert clear["ndvi"].between(-1, 1).all()


def test_fully_cloudy_date_is_kept_with_no_values(recorded):
    frame, _ = parse_response(recorded)
    cloudy = frame.loc["2024-01-20"]
    assert np.isnan(cloudy["ndvi"]) and np.isnan(cloudy["ndwi"])
    assert cloudy["valid_fraction"] == 0.0


def test_clear_share_is_measured_inside_the_parcel_not_its_bounding_box():
    # Real numbers from a diagonal 3 ha parcel: the API counts 638 pixels over the bounding
    # box, of which 283 lie outside the parcel and are reported as "no data". Only 355 pixels
    # are inside. On a clear day all 355 are clear, so the parcel is 100 % clear, not 56 %.
    stats = {"mean": 0.6, "stDev": 0.1, "sampleCount": 638, "noDataCount": 283}
    outputs = {name: {"bands": {"B0": {"stats": stats}}} for name in ("ndvi", "ndwi")}
    interval = {"from": "2024-01-09T00:00:00Z", "to": "2024-01-10T00:00:00Z"}
    payload = {
        "status": "OK",
        "geometryPixelCount": 355,
        "data": [{"interval": interval, "outputs": outputs}],
    }
    frame, _ = parse_response(payload)
    assert frame["valid_fraction"].iloc[0] == pytest.approx(1.0)
    assert frame["pixels"].iloc[0] == 355


def test_failed_dates_are_skipped_and_counted(recorded):
    payload = copy.deepcopy(recorded)
    payload["data"][1] = {"interval": payload["data"][1]["interval"], "error": {"type": "X"}}
    frame, failed = parse_response(payload)
    assert failed == 1
    assert len(frame) == 5


def test_a_response_where_every_date_failed_is_an_error(recorded):
    payload = copy.deepcopy(recorded)
    payload["data"] = [{"interval": d["interval"], "error": {"type": "X"}} for d in payload["data"]]
    with pytest.raises(SentinelError, match="could not process any date"):
        parse_response(payload)


def test_a_failed_status_is_an_error():
    with pytest.raises(SentinelError, match="status"):
        parse_response({"status": "FAILED", "data": []})


def test_a_period_without_images_gives_an_empty_table():
    frame, failed = parse_response({"status": "OK", "data": []})
    assert frame.empty and failed == 0
    assert "valid_fraction" in frame.columns


# --- date_chunks and build_request ---------------------------------------------------------


def test_chunks_cover_the_period_without_gaps_or_overlaps():
    chunks = list(date_chunks(date(2018, 8, 1), date(2026, 9, 30)))
    assert chunks[0][0] == date(2018, 8, 1) and chunks[-1][1] == date(2026, 9, 30)
    for (_, end), (next_start, _) in zip(chunks, chunks[1:], strict=False):
        assert (next_start - end).days == 1
    assert all((end - start).days < 365 for start, end in chunks)


def test_a_single_day_is_one_chunk():
    assert list(date_chunks(date(2024, 1, 5), date(2024, 1, 5))) == [
        (date(2024, 1, 5), date(2024, 1, 5))
    ]


def test_request_asks_for_whole_days_at_10_m():
    request = build_request(
        Geometry(ICA_BOX.geometry, CRS(32718)), date(2024, 1, 1), date(2024, 1, 31), SHConfig()
    )
    aggregation = request.payload["aggregation"]
    assert aggregation["timeRange"] == {
        "from": "2024-01-01T00:00:00Z",
        "to": "2024-02-01T00:00:00Z",  # exclusive end: January 31 is included in full
    }
    assert aggregation["aggregationInterval"] == {"of": "P1D"}
    assert (aggregation["resx"], aggregation["resy"]) == (10, 10)


# --- fetch_daily_stats ---------------------------------------------------------------------


class FakeSender:
    """Stands in for the network: answers each request with one clear date at its start."""

    def __init__(self, units: float = 0.25):
        self.units = units
        self.time_ranges: list[dict] = []

    def __call__(self, request):
        time_range = request.payload["aggregation"]["timeRange"]
        self.time_ranges.append(time_range)
        stats = {"mean": 0.5, "stDev": 0.1, "sampleCount": 300, "noDataCount": 30}
        interval = {"from": time_range["from"], "to": time_range["from"]}
        outputs = {name: {"bands": {"B0": {"stats": stats}}} for name in ("ndvi", "ndwi")}
        return {"status": "OK", "data": [{"interval": interval, "outputs": outputs}]}, self.units


def test_fetch_makes_one_request_per_year_and_sums_the_cost():
    sender = FakeSender(units=0.25)
    result = fetch_daily_stats(
        ICA_BOX.geometry, 32718, date(2018, 8, 1), date(2021, 7, 30), config=SHConfig(), send=sender
    )
    assert isinstance(result, DailyStats)
    assert result.requests == 3 and len(sender.time_ranges) == 3
    assert result.processing_units == pytest.approx(0.75)
    assert sender.time_ranges[0]["to"] == sender.time_ranges[1]["from"]  # contiguous
    assert list(result.observations.index) == list(
        pd.to_datetime(["2018-08-01", "2019-08-01", "2020-07-31"])
    )
    assert result.observations["valid_fraction"].tolist() == pytest.approx([0.9, 0.9, 0.9])


def test_fetch_rejects_a_reversed_period():
    with pytest.raises(ValueError, match="before the start"):
        fetch_daily_stats(ICA_BOX.geometry, 32718, date(2024, 1, 1), date(2023, 1, 1))
