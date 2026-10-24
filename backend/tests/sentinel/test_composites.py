"""Tests for openEO monthly composites. No network: offline graphs and a recorded result."""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests
from openeo.rest import OpenEoApiPlainError
from shapely.geometry import box

from fundoscope.config import get_settings
from fundoscope.sentinel import (
    COMPOSITE_BANDS,
    MonthlyComposites,
    SentinelError,
    build_graph,
    fetch_monthly_composites,
    openeo_connection,
    parse_netcdf,
)
from fundoscope.sentinel.composites import _send_sync
from fundoscope.sentinel.evalscripts import NOT_CLEAR_SCL_CLASSES

FIXTURE = Path(__file__).parent / "fixtures" / "composite_ica_2024-01_02.nc"
# The recorded box: 200 m x 200 m near Ica (public area), Jan-Feb 2024, all 10 bands.


@pytest.fixture
def recorded() -> bytes:
    return FIXTURE.read_bytes()


def _nodes(graph: dict, process_id: str) -> list[dict]:
    """All nodes of a process graph (including nested callbacks) with this process id."""
    found = []

    def walk(value):
        if isinstance(value, dict):
            if value.get("process_id") == process_id:
                found.append(value)
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(graph)
    return found


# --- parse_netcdf --------------------------------------------------------------------------


def test_parse_a_real_result(recorded):
    stack = parse_netcdf(recorded)
    assert stack.dims == ("month", "band", "y", "x")
    assert stack.shape == (2, 10, 21, 21)
    assert list(stack["band"].to_numpy()) == list(COMPOSITE_BANDS)
    assert list(pd.DatetimeIndex(stack["month"].to_numpy())) == list(
        pd.to_datetime(["2024-01-01", "2024-02-01"])
    )
    assert stack.dtype == np.float32
    values = stack.to_numpy()
    assert np.nanmin(values) >= 0 and np.nanmax(values) <= 1  # reflectance, not x 10,000
    assert float(stack["x"][1] - stack["x"][0]) == 10.0  # 10 m pixels


def test_unreadable_content_is_an_error():
    with pytest.raises(SentinelError, match="not readable NetCDF"):
        parse_netcdf(b"<html>Service unavailable</html>")


def test_missing_band_is_an_error(recorded):
    with pytest.raises(SentinelError, match="missing the bands: B01"):
        parse_netcdf(recorded, bands=("B01", "B04"))


# --- build_graph ---------------------------------------------------------------------------


def test_graph_steps_in_order():
    graph = build_graph((1000, 2000, 1300, 2200), 32718, date(2023, 8, 1), date(2024, 7, 31))
    top_level = [node["process_id"] for node in graph.values()]
    assert top_level == [
        "load_collection",
        "filter_bands",
        "reduce_dimension",  # picks the SCL band out of the cube
        "mask",
        "aggregate_temporal_period",
        "resample_spatial",
        "save_result",
    ]


def test_graph_loads_a_padded_box_and_whole_days():
    graph = build_graph((1000, 2000, 1300, 2200), 32718, date(2023, 8, 1), date(2024, 7, 31))
    [load] = _nodes(graph, "load_collection")
    args = load["arguments"]
    assert args["spatial_extent"] == {
        "west": 990,
        "south": 1990,
        "east": 1310,
        "north": 2210,
        "crs": 32718,
    }
    assert args["temporal_extent"] == ["2023-08-01", "2024-08-01"]  # end excluded by openEO
    assert args["bands"] == [*COMPOSITE_BANDS, "SCL"]


def test_graph_masks_the_same_cloud_classes_as_the_daily_statistics():
    graph = build_graph((0, 0, 100, 100), 32717, date(2024, 1, 1), date(2024, 1, 31))
    compared = {node["arguments"]["y"] for node in _nodes(graph, "eq")}
    assert compared == set(NOT_CLEAR_SCL_CLASSES)


def test_graph_takes_monthly_medians_on_a_10_m_utm_grid():
    graph = build_graph((0, 0, 100, 100), 32717, date(2024, 1, 1), date(2024, 1, 31))
    [aggregate] = _nodes(graph, "aggregate_temporal_period")
    assert aggregate["arguments"]["period"] == "month"
    assert _nodes(aggregate, "median")
    [resample] = _nodes(graph, "resample_spatial")
    assert resample["arguments"]["resolution"] == 10
    assert resample["arguments"]["projection"] == 32717
    [save] = _nodes(graph, "save_result")
    assert save["arguments"]["format"] == "netCDF"
    json.dumps(graph)  # the graph must be plain JSON to send it


# --- fetch_monthly_composites and MonthlyComposites ----------------------------------------


@pytest.fixture
def composites(recorded) -> MonthlyComposites:
    sent = []

    def fake_send(graph):
        sent.append(graph)
        return recorded, 7.0

    result = fetch_monthly_composites(
        box(420_600, 8_443_800, 420_800, 8_444_000),
        32718,
        date(2024, 1, 1),
        date(2024, 2, 29),
        send=fake_send,
    )
    assert len(sent) == 1
    return result


def test_fetch_returns_the_stack_and_its_cost(composites):
    assert composites.credits == 7.0
    assert composites.epsg == 32718
    assert list(composites.months.strftime("%Y-%m")) == ["2024-01", "2024-02"]
    assert composites.band("B04").shape == (2, 21, 21)


def test_inside_selects_pixel_centers_within_a_parcel(composites):
    x, y = composites.data["x"].to_numpy(), composites.data["y"].to_numpy()
    # A square around the 5 x 5 central pixels (pixel centers are 10 m apart).
    parcel = box(x[8] - 5, y[12] - 5, x[12] + 5, y[8] + 5)
    assert composites.inside(parcel).sum() == 25


def test_masked_to_blanks_everything_outside_the_parcel(composites):
    x, y = composites.data["x"].to_numpy(), composites.data["y"].to_numpy()
    parcel = box(x[8] - 5, y[12] - 5, x[12] + 5, y[8] + 5)
    masked = composites.masked_to(parcel)
    b04 = masked.band("B04")
    assert np.isnan(b04[:, 0, 0]).all()  # a corner pixel: outside
    assert not np.isnan(b04[:, 10, 10]).any()  # the center: inside
    assert np.count_nonzero(~np.isnan(b04[0])) == 25


def test_fetch_rejects_a_reversed_period(recorded):
    with pytest.raises(ValueError, match="before the start"):
        fetch_monthly_composites(box(0, 0, 10, 10), 32718, date(2024, 2, 1), date(2024, 1, 1))


def test_connection_requires_credentials(monkeypatch):
    monkeypatch.setenv("CDSE_CLIENT_ID", "")
    monkeypatch.setenv("CDSE_CLIENT_SECRET", "")
    get_settings.cache_clear()
    try:
        with pytest.raises(SentinelError, match="CDSE_CLIENT_ID"):
            openeo_connection()
    finally:
        get_settings.cache_clear()


# --- _send_sync: rate limits and errors (fake connection, no network) ----------------------


class FakeResponse:
    content = b"netcdf-bytes"
    headers = {"OpenEO-Costs-experimental": "7"}


class FakeConnection:
    def __init__(self, errors):
        self.errors = list(errors)
        self.posts = 0

    def post(self, *_args, **_kwargs):
        self.posts += 1
        if self.errors:
            raise self.errors.pop(0)
        return FakeResponse()


def too_many_requests():
    return OpenEoApiPlainError(message="Too Many Requests", http_status_code=429)


def test_rate_limited_requests_wait_and_retry():
    waits = []
    connection = FakeConnection([too_many_requests(), too_many_requests()])
    content, credits = _send_sync({}, connection=connection, sleep=waits.append)
    assert (content, credits) == (b"netcdf-bytes", 7.0)
    assert waits == [5, 15] and connection.posts == 3


def test_persistent_rate_limiting_gives_up_with_a_clear_error():
    waits = []
    connection = FakeConnection([too_many_requests() for _ in range(4)])
    with pytest.raises(SentinelError, match="429"):
        _send_sync({}, connection=connection, sleep=waits.append)
    assert waits == [5, 15, 45]


def test_a_server_error_is_reported_without_waiting():
    waits = []
    connection = FakeConnection([OpenEoApiPlainError(message="Internal", http_status_code=500)])
    with pytest.raises(SentinelError, match="openEO request failed"):
        _send_sync({}, connection=connection, sleep=waits.append)
    assert waits == []


def test_a_network_failure_is_reported():
    connection = FakeConnection([requests.ConnectionError("no route to host")])
    with pytest.raises(SentinelError, match="Could not reach openEO"):
        _send_sync({}, connection=connection, sleep=lambda _s: None)
