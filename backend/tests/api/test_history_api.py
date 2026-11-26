"""The parcel history API with fake downloaders, an in-memory parcel store, and a fixed date."""

import os
from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier
from sqlalchemy.exc import OperationalError

from fundoscope.analysis import Flag
from fundoscope.api import history as history_api
from fundoscope.api.main import app
from fundoscope.api.schemas import history_out
from fundoscope.db import MemoryParcelStore
from fundoscope.pipeline import MODEL_FILE, parcel_history
from fundoscope.sentinel import SentinelError

TODAY = date(2024, 7, 31)  # data from Aug 2018 to here: campaigns 2018/19 ... 2023/24


@pytest.fixture
def store():
    return MemoryParcelStore()


@pytest.fixture
def client(tmp_path, fake_fetch, fake_composites, store):
    app.dependency_overrides = {
        history_api.daily_fetcher: lambda: fake_fetch,
        history_api.composites_fetcher: lambda: fake_composites,
        history_api.crop_model: lambda: None,
        history_api.model_version: lambda: None,
        history_api.parcel_store: lambda: store,
        history_api.today: lambda: TODAY,
        history_api.valleys_folder: lambda: tmp_path / "valleys",
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


def test_the_same_parcel_on_the_same_day_downloads_nothing(
    client, parcel_geojson, fake_fetch, fake_composites
):
    first = upload(client, parcel_geojson).json()
    second = upload(client, parcel_geojson)

    assert second.status_code == 200
    body = second.json()
    assert body["summary"]["cached"] is True
    assert body["summary"]["processing_units"] == 0 and body["summary"]["credits"] == 0
    assert body["parcel"]["id"] == first["parcel"]["id"]
    assert body["campaigns"] == first["campaigns"]
    assert len(fake_fetch.calls) == 1 and len(fake_composites.calls) == 6


def test_a_new_day_downloads_only_the_new_dates(
    client, parcel_geojson, fake_fetch, fake_composites
):
    upload(client, parcel_geojson)
    app.dependency_overrides[history_api.today] = lambda: date(2024, 8, 20)

    body = upload(client, parcel_geojson).json()

    assert body["summary"]["cached"] is False
    _, _, start, end = fake_fetch.calls[1]
    assert (start, end) == (date(2024, 8, 1), date(2024, 8, 20))
    # Only the new campaign's composites: the finished ones are stored.
    assert [call[0] for call in fake_composites.calls[6:]] == [date(2024, 8, 1)]
    assert body["campaigns"][-1]["campaign"] == "2024/25"


def test_a_stored_report_is_reopened_without_downloading(client, parcel_geojson, fake_fetch):
    first = upload(client, parcel_geojson).json()

    response = client.get(f"/api/parcels/{first['parcel']['id']}/history")

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["cached"] is True and body["summary"]["processing_units"] == 0
    assert body["campaigns"] == first["campaigns"]
    assert body["observations"] == first["observations"]
    assert len(fake_fetch.calls) == 1


def test_stored_parcels_are_listed(client, parcel_geojson):
    first = upload(client, parcel_geojson).json()

    listed = client.get("/api/parcels").json()

    assert [p["id"] for p in listed] == [first["parcel"]["id"]]
    assert listed[0]["name"] == "Lote demo" and listed[0]["area_ha"] == 9.0
    assert listed[0]["observed_through"] == "2024-07-31"


def test_an_unknown_parcel_is_a_404(client):
    response = client.get("/api/parcels/00000000-0000-0000-0000-000000000000/history")

    assert response.status_code == 404


def test_a_database_that_is_down_is_a_503(client):
    def down():
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    app.dependency_overrides[history_api.parcel_store] = down

    response = client.get("/api/parcels")

    assert response.status_code == 503
    assert "database isn't available" in response.json()["detail"]


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


def test_with_a_model_each_campaign_has_its_crop(client, parcel_geojson, fake_crop_model):
    app.dependency_overrides[history_api.crop_model] = lambda: fake_crop_model

    campaigns = upload(client, parcel_geojson).json()["campaigns"]

    assert [c["crop"] for c in campaigns] == ["grape"] * 6
    assert all(0.7 < c["crop_confidence"] < 0.8 for c in campaigns)


def test_without_a_model_the_crop_is_empty(client, parcel_geojson):
    campaigns = upload(client, parcel_geojson).json()["campaigns"]

    assert {(c["crop"], c["crop_confidence"]) for c in campaigns} == {(None, None)}


def test_the_model_is_loaded_once_and_again_after_retraining(tmp_path):
    assert history_api.cached_model(tmp_path) is None
    joblib.dump(DummyClassifier().fit([[0], [1]], ["grape", "fallow/bare"]), tmp_path / MODEL_FILE)

    first = history_api.cached_model(tmp_path)
    assert list(first.classes_) == ["fallow/bare", "grape"]
    assert history_api.cached_model(tmp_path) is first

    os.utime(tmp_path / MODEL_FILE, (1, 1))  # a retrained model has a new date
    assert history_api.cached_model(tmp_path) is not first


def test_the_model_comes_from_the_models_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(history_api, "get_settings", lambda: SimpleNamespace(models_dir=tmp_path))
    assert history_api.crop_model() is None


def test_flags_are_part_of_the_answer(client, parcel_geojson, fake_fetch):
    assert upload(client, parcel_geojson).json()["flags"] == []  # a steady parcel

    history = parcel_history(parcel_geojson, "lote.geojson", end=TODAY, fetch=fake_fetch)
    flag = Flag(
        "2021/22", "possible abandonment", "Possible abandonment: ...", {"active_area": 0.2}
    )
    body = history_out(replace(history, flags=[flag])).model_dump()

    assert body["flags"] == [
        {
            "campaign": "2021/22",
            "kind": "possible abandonment",
            "message": "Possible abandonment: ...",
            "evidence": {"active_area": 0.2},
        }
    ]


def test_a_parcel_outside_the_pilot_area_gets_a_note(client, parcel_geojson):
    neighbors = upload(client, parcel_geojson).json()["neighbors"]

    assert neighbors["valley"] is None and neighbors["bands"] == []
    assert "only available inside the pilot area" in neighbors["note"]


def test_a_parcel_inside_a_counted_valley_is_compared_with_its_neighbors(
    client, neighbor_valley, fake_fetch
):
    valleys, parcel_file = neighbor_valley
    app.dependency_overrides[history_api.valleys_folder] = lambda: valleys

    body = upload(client, parcel_file, "vecino.geojson").json()

    neighbors = body["neighbors"]
    assert neighbors["valley"] == "demo"
    assert neighbors["ring"]["type"] == "Polygon"
    first = neighbors["bands"][0]
    assert (first["campaign"], first["band"], first["underperforming"]) == (
        "2024/25",
        "below p10",
        True,
    )
    assert first["p10"] < first["p50"] < first["p90"]
    assert [b["campaign"] for b in neighbors["bands"]] == ["2024/25", "2025/26", "2026/27"]
    assert any(flag["kind"] == "underperforming" for flag in body["flags"])


def test_the_model_version_is_the_model_file_time(monkeypatch, tmp_path):
    from fundoscope.config import get_settings

    monkeypatch.setattr(get_settings(), "models_dir", tmp_path)
    assert history_api.model_version() is None
    (tmp_path / MODEL_FILE).write_bytes(b"model")
    assert history_api.model_version().startswith(str(date.today().year))
