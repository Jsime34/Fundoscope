"""The API with in-memory stores, fake downloaders, and a fixed date: nothing leaves the test.

The app's worker thread doesn't run here (it starts with the app's lifespan, which a plain
`TestClient(app)` skips). `run_jobs` plays its part: it runs the queued jobs at once.
"""

from datetime import date

import pytest
from fastapi.testclient import TestClient

from fundoscope.api import history as history_api
from fundoscope.api import jobs as jobs_api
from fundoscope.api.main import app
from fundoscope.api.valley import valley_store, valleys_folder
from fundoscope.db import MemoryJobStore, MemoryParcelStore, MemoryValleyStore
from fundoscope.pipeline.jobs import run_next
from fundoscope.pipeline.parcel_store import update_parcel
from fundoscope.pipeline.valley_store import CountedValleys

TODAY = date(2024, 7, 31)  # data from Aug 2018 to here: campaigns 2018/19 ... 2023/24


@pytest.fixture
def store():
    return MemoryParcelStore()


@pytest.fixture
def jobs():
    return MemoryJobStore()


@pytest.fixture
def valleys(tmp_path):
    """No counted valley (tests that need one replace it)."""
    return CountedValleys(tmp_path / "valleys", MemoryValleyStore())


@pytest.fixture
def client(store, jobs, valleys):
    app.dependency_overrides = {
        history_api.parcel_store: lambda: store,
        jobs_api.job_store: lambda: jobs,
        valleys_folder: lambda: valleys.folder,
        valley_store: lambda: valleys.runs,
    }
    yield TestClient(app)
    app.dependency_overrides = {}


@pytest.fixture
def run_jobs(store, jobs, fake_fetch, fake_composites, valleys):
    """Run every queued job, as the worker would, with the fakes (options replace them)."""

    def run(**options):
        settings = {
            "today": TODAY,
            "fetch": fake_fetch,
            "fetch_composites": fake_composites,
            "valleys": valleys,
        } | options

        def analyze(parcel_id, on_progress):
            return update_parcel(parcel_id, store, on_progress=on_progress, **settings)

        finished = []
        while (job := run_next(jobs, analyze)) is not None:
            finished.append(job)
        return finished

    return run


@pytest.fixture
def analyze(client, run_jobs):
    """Upload a parcel file, start its analysis, run it, and return (job, report response)."""

    def upload_and_run(content: bytes, name: str = "lote.geojson", **options):
        parcel = client.post(
            "/api/parcels", files={"file": (name, content, "application/geo+json")}
        ).json()
        job = client.post(f"/api/parcels/{parcel['id']}/analyses").json()
        run_jobs(**options)
        finished = client.get(f"/api/jobs/{job['id']}").json()
        return finished, client.get(f"/api/parcels/{parcel['id']}/history")

    return upload_and_run
