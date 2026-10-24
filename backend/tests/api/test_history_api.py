"""POST /api/history with fake downloaders, a temporary cache folder, and a fixed date."""

from datetime import date

import pytest
from fastapi.testclient import TestClient

from fundoscope.api import history as history_api
from fundoscope.api.main import app
from fundoscope.sentinel import SentinelError

TODAY = date(2024, 7, 31)  # data from Aug 2018 to here: campaigns 2018/19 ... 2023/24


@pytest.fixture
def client(tmp_path, fake_fetch, fake_composites):
    app.dependency_overrides = {
        history_api.daily_fetcher: lambda: fake_fetch,
        history_api.composites_fetcher: lambda: fake_composites,
        history_api.cache_folder: lambda: tmp_path / "cache",
        history_api.today: lambda: TODAY,
    }
    yield TestClient(app)
    app.dependency_overrides = {}


def upload(client, content: bytes, name: str = "lote.geojson"):
    return client.post("/api/history", files={"file": (name, content, "application/geo+json")})


def test_upload_returns_the_full_history(client, parcel_geojson):
    response = upload(client, parcel_geojson)
    assert response.status_code == 200
    body = response.json()

    assert body["parcel"]["name"] == "Lote demo"
    assert body["parcel"]["area_ha"] == pytest.approx(9.0)
    assert body["parcel"]["zone"] == "UTM 18S"
    assert body["parcel"]["geometry"]["type"] == "MultiPolygon"

    campaigns = body["campaigns"]
    assert [c["campaign"] for c in campaigns] == [
        "2018/19",
        "2019/20",
        "2020/21",
        "2021/22",
        "2022/23",
        "2023/24",
    ]
    assert all(c["active_area"] == pytest.approx(23 / 30, abs=1e-4) for c in campaigns)

    summary = body["summary"]
    assert summary["processing_units"] == 2.5
    assert summary["credits"] == 6 * 7.0
    assert summary["cached"] is False
    assert body["series"] and body["observations"]
    assert "NaN" not in response.text  # missing values travel as null, which JSON allows


def test_the_same_file_on_the_same_day_comes_from_the_cache(client, parcel_geojson, fake_fetch):
    upload(client, parcel_geojson)
    second = upload(client, parcel_geojson)
    assert second.status_code == 200
    assert second.json()["summary"]["cached"] is True
    assert len(fake_fetch.calls) == 1  # nothing downloaded the second time


def test_a_new_day_computes_again(client, parcel_geojson, fake_fetch):
    upload(client, parcel_geojson)
    app.dependency_overrides[history_api.today] = lambda: date(2024, 8, 1)
    assert upload(client, parcel_geojson).json()["summary"]["cached"] is False
    assert len(fake_fetch.calls) == 2


def test_an_invalid_file_is_a_422_with_the_reason(client, fake_fetch):
    response = upload(client, b"<kml><Placemark>", "broken.kml")
    assert response.status_code == 422
    assert "not valid XML" in response.json()["detail"]
    assert fake_fetch.calls == []


def test_a_file_over_5_mb_is_refused(client):
    response = upload(client, b" " * (5 * 1024 * 1024 + 1), "huge.geojson")
    assert response.status_code == 413


def test_a_satellite_failure_is_a_502(client, parcel_geojson):
    def failing(*_args):
        raise SentinelError("status 'FAILED'")

    app.dependency_overrides[history_api.daily_fetcher] = lambda: failing
    response = upload(client, parcel_geojson)
    assert response.status_code == 502
    assert "satellite service failed" in response.json()["detail"]
