"""GET /api/valleys/{slug}: a count of the small synthetic valley near Ica, made offline."""

from datetime import date, datetime

import pytest
from fastapi.testclient import TestClient

from fundoscope.api import valley as valley_api
from fundoscope.api.main import app
from fundoscope.pipeline import plan_refresh, refresh_valley
from fundoscope.sentinel import CreditBudget
from fundoscope.valley import PLANTED


@pytest.fixture
def counted(tmp_path, small_valley, valley_models, valley_fakes):
    plan = plan_refresh(small_valley, tmp_path / "demo", date(2026, 11, 24))
    refresh_valley(
        plan,
        tmp_path / "demo",
        valley_models,
        CreditBudget(1_000),
        now=lambda: datetime(2026, 11, 24, 9, 30),
        **valley_fakes,
    )
    return tmp_path


@pytest.fixture
def client(counted):
    app.dependency_overrides = {valley_api.valleys_folder: lambda: counted}
    yield TestClient(app)
    app.dependency_overrides = {}


def test_the_latest_count_of_a_valley(client):
    response = client.get("/api/valleys/demo")

    assert response.status_code == 200
    body = response.json()
    assert body["slug"] == "demo"
    assert (body["updated_at"], body["as_of"], body["newest_image"]) == (
        "2026-11-24T09:30:00",
        "2026-11-22",
        "2026-11-21",
    )
    assert body["farmland_ha"] == 0.23
    assert len(body["corners"]) == 4
    current = body["counts"][0]
    assert (current["key"], current["months"], current["in_progress"]) == (
        "2026-27_to_2026-11-22",
        4,
        True,
    )
    assert current["farmland_image"] == "/api/valleys/demo/maps/2026-27_to_2026-11-22/farmland.png"
    planted = next(item for item in current["classes"] if item["name"] == PLANTED)
    assert planted["hectares"] == 0.16 and planted["low"] is None
    assert planted["color"] == "#008300"
    assert planted["image"].endswith("/planted-not-yet-identifiable.png")
    grape = next(item for item in current["classes"] if item["name"] == "grape")
    assert grape["low"] <= grape["adjusted"] <= grape["high"]


def test_every_image_is_served(client):
    body = client.get("/api/valleys/demo").json()
    urls = [body["counts"][0]["farmland_image"]] + [
        c["image"] for c in body["counts"][0]["classes"]
    ]

    for url in urls:
        response = client.get(url)
        assert response.status_code == 200, url
        assert response.headers["content-type"] == "image/png"
        assert response.content.startswith(b"\x89PNG")
        assert "max-age=3600" in response.headers["cache-control"]


def test_a_valley_without_a_count(client):
    response = client.get("/api/valleys/ica")

    assert response.status_code == 404
    assert "no count yet" in response.json()["detail"]


def test_a_missing_image(client):
    assert client.get("/api/valleys/demo/maps/2026-27_to_2026-11-22/citrus.png").status_code == 404


@pytest.mark.parametrize(
    "url",
    ["/api/valleys/..%2Fprivate", "/api/valleys/Demo", "/api/valleys/demo/maps/..%2F..%2Fx/y.png"],
)
def test_names_cannot_leave_the_valley_folder(client, url):
    assert client.get(url).status_code in (404, 422)


def test_valleys_live_in_the_data_folder():
    from fundoscope.config import get_settings

    assert valley_api.valleys_folder() == get_settings().data_dir / "valley"
