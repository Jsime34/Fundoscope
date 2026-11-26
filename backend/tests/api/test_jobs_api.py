"""Analyses as background jobs through the API, FR-13 (fixtures in conftest.py)."""

import os
import time
from datetime import date
from types import SimpleNamespace
from uuid import uuid4

import joblib
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from fundoscope.api import jobs as jobs_api
from fundoscope.api.main import app
from fundoscope.db import AutocommitParcelStore, SqlJobStore
from fundoscope.pipeline import MODEL_FILE
from fundoscope.pipeline.jobs import Worker
from fundoscope.sentinel import SentinelError, fetch_daily_stats, fetch_monthly_composites


def upload(client, content: bytes) -> str:
    response = client.post(
        "/api/parcels", files={"file": ("lote.geojson", content, "application/geo+json")}
    )
    return response.json()["id"]


def test_starting_an_analysis_answers_at_once_with_a_queued_job(client, parcel_geojson):
    parcel_id = upload(client, parcel_geojson)

    started = time.perf_counter()
    response = client.post(f"/api/parcels/{parcel_id}/analyses")
    elapsed = time.perf_counter() - started

    assert response.status_code == 202 and elapsed < 1.0
    job = response.json()
    assert (job["parcel_id"], job["kind"], job["status"], job["progress"]) == (
        parcel_id,
        "analyze",
        "queued",
        0,
    )
    assert job["step"] == "Waiting for the worker" and job["report"] is None
    assert client.get(f"/api/jobs/{job['id']}").json() == job


def test_an_analysis_already_queued_isnt_queued_twice(client, parcel_geojson):
    parcel_id = upload(client, parcel_geojson)

    first = client.post(f"/api/parcels/{parcel_id}/analyses").json()
    second = client.post(f"/api/parcels/{parcel_id}/analyses").json()

    assert second["id"] == first["id"]


def test_an_unknown_parcel_cant_be_analyzed(client):
    response = client.post(f"/api/parcels/{uuid4()}/analyses")

    assert response.status_code == 404


def test_a_finished_job_is_done_with_its_cost_and_its_report(client, analyze, parcel_geojson):
    job, report = analyze(parcel_geojson)

    assert (job["status"], job["progress"], job["step"], job["error"]) == (
        "done",
        100,
        "Done",
        None,
    )
    assert (job["processing_units"], job["credits"]) == (2.5, 42.0)
    assert job["started_at"] and job["finished_at"]
    assert job["report"] == f"/api/parcels/{job['parcel_id']}/history"
    assert client.get(job["report"]).json() == report.json()


def test_a_failing_analysis_ends_failed_and_the_api_keeps_working(client, analyze, parcel_geojson):
    def copernicus_down(*_args):
        raise SentinelError("HTTP 503 from the Statistical API")

    job, report = analyze(parcel_geojson, fetch=copernicus_down)

    assert job["status"] == "failed"
    assert job["error"] == (
        "The satellite service failed (HTTP 503 from the Statistical API). Try again later."
    )
    assert job["report"] is None and report.status_code == 404
    assert client.get("/health").status_code == 200
    assert client.get("/api/parcels").status_code == 200
    # Starting again queues a new job.
    retry = client.post(f"/api/parcels/{job['parcel_id']}/analyses").json()
    assert retry["id"] != job["id"] and retry["status"] == "queued"


def test_an_unknown_job_is_a_404(client):
    assert client.get(f"/api/jobs/{uuid4()}").status_code == 404


def test_starting_an_analysis_wakes_the_worker(client, parcel_geojson):
    woken = []
    app.state.worker = SimpleNamespace(wake=lambda: woken.append(True))
    try:
        client.post(f"/api/parcels/{upload(client, parcel_geojson)}/analyses")
    finally:
        del app.state.worker

    assert woken == [True]


def test_the_job_endpoints_are_documented(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/api/parcels", "/api/parcels/{parcel_id}/analyses", "/api/jobs/{job_id}"} <= set(paths)


def test_the_worker_starts_and_stops_with_the_app(monkeypatch):
    events = []

    class FakeWorker:
        def start(self):
            events.append("start")

        def stop(self):
            events.append("stop")

    worker = FakeWorker()
    monkeypatch.setattr(jobs_api, "make_worker", lambda: worker)

    with TestClient(app):
        assert app.state.worker is worker and events == ["start"]

    assert events == ["start", "stop"] and not hasattr(app.state, "worker")


def test_the_app_worker_uses_copernicus_the_saved_model_and_postgis(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(jobs_api, "update_parcel", lambda *a, **k: calls.append((a, k)) or "ok")
    monkeypatch.setattr(jobs_api, "crop_model", lambda: "model")
    monkeypatch.setattr(jobs_api, "model_version", lambda: "v1")

    def progress(done, total):
        pass

    assert jobs_api.analyze_with_copernicus("parcel-id", progress) == "ok"

    [(args, options)] = calls
    assert args[0] == "parcel-id" and isinstance(args[1], AutocommitParcelStore)
    assert options["today"] == date.today()
    assert options["fetch"] is fetch_daily_stats
    assert options["fetch_composites"] is fetch_monthly_composites
    assert (options["model"], options["model_version"]) == ("model", "v1")
    assert options["on_progress"] is progress
    worker = jobs_api.make_worker()
    assert isinstance(worker, Worker) and isinstance(worker.jobs, SqlJobStore)
    assert worker.analyze is jobs_api.analyze_with_copernicus
    assert isinstance(jobs_api.job_store(), SqlJobStore)


def test_the_model_is_loaded_once_and_again_after_retraining(tmp_path):
    assert jobs_api.cached_model(tmp_path) is None
    joblib.dump(DummyClassifier().fit([[0], [1]], ["grape", "fallow/bare"]), tmp_path / MODEL_FILE)

    first = jobs_api.cached_model(tmp_path)
    assert list(first.classes_) == ["fallow/bare", "grape"]
    assert jobs_api.cached_model(tmp_path) is first

    os.utime(tmp_path / MODEL_FILE, (1, 1))  # a retrained model has a new date
    assert jobs_api.cached_model(tmp_path) is not first


def test_the_model_comes_from_the_models_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs_api, "get_settings", lambda: SimpleNamespace(models_dir=tmp_path))
    assert jobs_api.crop_model() is None
    assert jobs_api.model_version() is None
    (tmp_path / MODEL_FILE).write_bytes(b"model")
    assert jobs_api.model_version().startswith(str(date.today().year))
