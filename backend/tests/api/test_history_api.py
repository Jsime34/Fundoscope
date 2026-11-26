"""Parcels and their reports through the API (fixtures in conftest.py)."""

from dataclasses import replace
from datetime import date

import pytest
from sqlalchemy.exc import OperationalError

from fundoscope.analysis import Flag
from fundoscope.api import history as history_api
from fundoscope.api.main import app
from fundoscope.api.schemas import history_out
from fundoscope.pipeline import parcel_history

TODAY = date(2024, 7, 31)


def upload(client, content: bytes, name: str = "lote.geojson"):
    return client.post("/api/parcels", files={"file": (name, content, "application/geo+json")})


def test_uploading_stores_the_parcel_without_downloading(client, parcel_geojson, fake_fetch):
    response = upload(client, parcel_geojson)

    assert response.status_code == 200
    parcel = response.json()
    assert parcel["id"] and parcel["name"] == "Lote demo"
    assert parcel["area_ha"] == pytest.approx(9.0) and parcel["zone"] == "UTM 18S"
    assert parcel["geometry"]["type"] == "MultiPolygon" and len(parcel["bounds"]) == 4
    assert fake_fetch.calls == []


def test_the_same_outline_is_the_same_parcel(client, parcel_geojson):
    first = upload(client, parcel_geojson).json()

    assert upload(client, parcel_geojson, "again.geojson").json()["id"] == first["id"]


def test_an_invalid_file_is_a_422_with_the_reason(client):
    response = upload(client, b"<kml><Placemark>", "broken.kml")

    assert response.status_code == 422
    assert "not valid XML" in response.json()["detail"]
    assert client.get("/api/parcels").json() == []


def test_a_file_over_5_mb_is_refused(client):
    response = upload(client, b" " * (5 * 1024 * 1024 + 1), "huge.geojson")

    assert response.status_code == 413


def test_an_analysis_gives_the_full_report(analyze, parcel_geojson):
    _, response = analyze(parcel_geojson)

    assert response.status_code == 200
    body = response.json()
    assert body["parcel"]["name"] == "Lote demo"
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
    assert (body["summary"]["processing_units"], body["summary"]["credits"]) == (2.5, 6 * 7.0)
    assert body["series"] and body["observations"]
    assert "NaN" not in response.text  # missing values travel as null, which JSON allows


def test_a_repeat_on_the_same_day_downloads_nothing(
    analyze, parcel_geojson, fake_fetch, fake_composites
):
    _, first = analyze(parcel_geojson)

    job, second = analyze(parcel_geojson)

    assert (job["processing_units"], job["credits"]) == (0, 0)
    body = second.json()
    assert body["parcel"]["id"] == first.json()["parcel"]["id"]
    assert body["campaigns"] == first.json()["campaigns"]
    assert (body["summary"]["processing_units"], body["summary"]["credits"]) == (2.5, 42.0)
    assert len(fake_fetch.calls) == 1 and len(fake_composites.calls) == 6


def test_a_new_day_downloads_only_the_new_dates(
    analyze, parcel_geojson, fake_fetch, fake_composites
):
    analyze(parcel_geojson)

    job, response = analyze(parcel_geojson, today=date(2024, 8, 20))

    _, _, start, end = fake_fetch.calls[1]
    assert (start, end) == (date(2024, 8, 1), date(2024, 8, 20))
    # Only the new campaign's composites: the finished ones are stored.
    assert [call[0] for call in fake_composites.calls[6:]] == [date(2024, 8, 1)]
    assert (job["processing_units"], job["credits"]) == (2.5, 7.0)
    body = response.json()
    assert body["campaigns"][-1]["campaign"] == "2024/25"
    # The report's cost adds up both analyses.
    assert (body["summary"]["processing_units"], body["summary"]["credits"]) == (5.0, 49.0)


def test_a_stored_report_is_reopened_without_downloading(
    client, analyze, parcel_geojson, fake_fetch
):
    _, first = analyze(parcel_geojson)

    again = client.get(f"/api/parcels/{first.json()['parcel']['id']}/history")

    assert again.status_code == 200
    assert again.json() == first.json()
    assert len(fake_fetch.calls) == 1


def test_a_parcel_never_analyzed_has_no_report_yet(client, parcel_geojson):
    parcel = upload(client, parcel_geojson).json()

    response = client.get(f"/api/parcels/{parcel['id']}/history")

    assert response.status_code == 404
    assert "start an analysis" in response.json()["detail"]
    unknown = client.get("/api/parcels/00000000-0000-0000-0000-000000000000/history")
    assert unknown.status_code == 404


def test_stored_parcels_are_listed(client, run_jobs, parcel_geojson):
    parcel = upload(client, parcel_geojson).json()
    [listed] = client.get("/api/parcels").json()
    assert (listed["id"], listed["observed_through"]) == (parcel["id"], None)  # not analyzed

    client.post(f"/api/parcels/{parcel['id']}/analyses")
    run_jobs()

    [listed] = client.get("/api/parcels").json()
    assert listed["name"] == "Lote demo" and listed["area_ha"] == 9.0
    assert listed["observed_through"] == "2024-07-31"


def test_a_database_that_is_down_is_a_503(client):
    def down():
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    app.dependency_overrides[history_api.parcel_store] = down

    response = client.get("/api/parcels")

    assert response.status_code == 503
    assert "database isn't available" in response.json()["detail"]


def test_with_a_model_each_campaign_has_its_crop(analyze, parcel_geojson, fake_crop_model):
    _, response = analyze(parcel_geojson, model=fake_crop_model)

    campaigns = response.json()["campaigns"]
    assert [c["crop"] for c in campaigns] == ["grape"] * 6
    assert all(0.7 < c["crop_confidence"] < 0.8 for c in campaigns)


def test_without_a_model_the_crop_is_empty(analyze, parcel_geojson):
    _, response = analyze(parcel_geojson)

    assert {(c["crop"], c["crop_confidence"]) for c in response.json()["campaigns"]} == {
        (None, None)
    }


def test_flags_are_part_of_the_answer(analyze, parcel_geojson, fake_fetch):
    assert analyze(parcel_geojson)[1].json()["flags"] == []  # a steady parcel

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
    assert body["summary"]["processing_units"] == 2.5  # without `spent`: this run's cost


def test_a_parcel_outside_the_pilot_area_gets_a_note(analyze, parcel_geojson):
    neighbors = analyze(parcel_geojson)[1].json()["neighbors"]

    assert neighbors["valley"] is None and neighbors["bands"] == []
    assert "only available inside the pilot area" in neighbors["note"]


def test_a_parcel_inside_a_counted_valley_is_compared_with_its_neighbors(analyze, neighbor_valley):
    valleys, parcel_file = neighbor_valley
    app.dependency_overrides[history_api.valleys_folder] = lambda: valleys

    body = analyze(parcel_file, "vecino.geojson", valleys=valleys)[1].json()

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
