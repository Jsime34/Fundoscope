"""Parcel analyses as background jobs (FR-13), with the in-memory stores and fakes."""

import time
from datetime import date

import pytest

from fundoscope.db import DONE, FAILED, QUEUED, RUNNING, MemoryJobStore, MemoryParcelStore
from fundoscope.pipeline.jobs import (
    INTERRUPTED,
    Worker,
    progress_step,
    readable_error,
    run_next,
    start_analysis,
)
from fundoscope.pipeline.parcel_store import store_parcel, update_parcel
from fundoscope.sentinel import SentinelError

TODAY = date(2024, 7, 31)  # campaigns 2018/19 to 2023/24


class RecordingJobs(MemoryJobStore):
    """Also keeps every progress update."""

    def __init__(self):
        super().__init__()
        self.updates = []

    def update(self, job_id, progress, step):
        self.updates.append((progress, step))
        super().update(job_id, progress, step)


@pytest.fixture
def parcels():
    return MemoryParcelStore()


@pytest.fixture
def parcel_id(parcels, parcel_geojson):
    return store_parcel(parcel_geojson, "lote.geojson", parcels).id


def runner(parcels, fetch, composites=None):
    def analyze(parcel_id, on_progress):
        return update_parcel(
            parcel_id,
            parcels,
            today=TODAY,
            fetch=fetch,
            fetch_composites=composites,
            on_progress=on_progress,
        )

    return analyze


def test_an_analysis_is_queued_once(parcel_id):
    jobs = MemoryJobStore()

    job = start_analysis(parcel_id, jobs)

    assert job.status == QUEUED
    assert start_analysis(parcel_id, jobs) == job  # still queued: the same job
    jobs.claim()
    assert start_analysis(parcel_id, jobs).id == job.id  # running: still the same
    jobs.finish(job.id)
    assert start_analysis(parcel_id, jobs).id != job.id  # finished: a new one


def test_nothing_queued_runs_nothing(parcels, fake_fetch):
    assert run_next(MemoryJobStore(), runner(parcels, fake_fetch)) is None
    assert fake_fetch.calls == []


def test_a_job_runs_to_done_with_its_progress_and_cost(
    parcels, parcel_id, fake_fetch, fake_composites
):
    jobs = RecordingJobs()
    job = start_analysis(parcel_id, jobs)

    done = run_next(jobs, runner(parcels, fake_fetch, fake_composites))

    assert (done.id, done.status, done.progress, done.error) == (job.id, DONE, 100, None)
    assert (done.processing_units, done.credits) == (2.5, 6 * 7.0)
    # 7 downloads: the daily statistics and the composites of 6 campaigns.
    assert jobs.updates[0] == (5, "Downloading satellite data: 0 of 7 requests done")
    assert [progress for progress, _ in jobs.updates] == sorted(p for p, _ in jobs.updates)
    assert jobs.updates[-1] == (95, "Building the report")
    assert parcels.get(parcel_id).observed_through == TODAY


def test_a_repeat_on_the_same_day_costs_nothing(parcels, parcel_id, fake_fetch):
    jobs = RecordingJobs()
    start_analysis(parcel_id, jobs)
    run_next(jobs, runner(parcels, fake_fetch))
    start_analysis(parcel_id, jobs)
    jobs.updates.clear()

    again = run_next(jobs, runner(parcels, fake_fetch))

    assert again.status == DONE and (again.processing_units, again.credits) == (0, 0)
    assert jobs.updates == [(95, "Building the report")]  # nothing to download
    assert len(fake_fetch.calls) == 1


def test_a_failing_analysis_ends_failed_in_plain_language(parcels, parcel_id):
    def failing(*_args):
        raise SentinelError("status 'FAILED'")

    jobs = MemoryJobStore()
    start_analysis(parcel_id, jobs)

    failed = run_next(jobs, runner(parcels, failing))

    assert failed.status == FAILED
    assert failed.error == "The satellite service failed (status 'FAILED'). Try again later."
    assert parcels.get(parcel_id).observed_through is None  # nothing saved


def test_progress_runs_from_5_to_95_while_downloading():
    assert progress_step(0, 10) == (5, "Downloading satellite data: 0 of 10 requests done")
    assert progress_step(5, 10)[0] == 50
    assert progress_step(9, 10)[0] == 86
    assert progress_step(10, 10) == (95, "Building the report")
    assert progress_step(0, 0) == (95, "Building the report")


def test_errors_are_one_short_sentence():
    assert readable_error(SentinelError("HTTP 503")) == (
        "The satellite service failed (HTTP 503). Try again later."
    )
    assert readable_error(LookupError("There is no stored parcel 42.\nmore")) == (
        "Unexpected error (LookupError): There is no stored parcel 42."
    )
    assert readable_error(RuntimeError()) == "Unexpected error (RuntimeError)."
    assert len(readable_error(ValueError("x" * 1000))) == 300


def test_the_worker_fails_jobs_cut_off_by_a_restart_once(parcels, parcel_id, fake_fetch):
    jobs = MemoryJobStore()
    cut_off = start_analysis(parcel_id, jobs)
    jobs.claim()  # running when the server stopped
    worker = Worker(jobs, runner(parcels, fake_fetch))

    assert worker.step() is False  # nothing queued
    assert (jobs.get(cut_off.id).status, jobs.get(cut_off.id).error) == (FAILED, INTERRUPTED)

    new = start_analysis(parcel_id, jobs)
    jobs.claim()  # running in this process: not touched by the next step
    assert worker.step() is False
    assert jobs.get(new.id).status == RUNNING


def test_the_worker_runs_one_queued_job_per_step(parcels, parcel_id, fake_fetch):
    jobs = MemoryJobStore()
    job = start_analysis(parcel_id, jobs)
    worker = Worker(jobs, runner(parcels, fake_fetch))

    assert worker.step() is True
    assert jobs.get(job.id).status == DONE
    assert worker.step() is False


def test_a_job_store_that_fails_doesnt_stop_the_worker(parcels, fake_fetch, caplog):
    class Down(MemoryJobStore):
        fail = True

        def fail_running(self, error):
            if self.fail:
                raise ConnectionError("database down")
            return super().fail_running(error)

    jobs = Down()
    worker = Worker(jobs, runner(parcels, fake_fetch))

    assert worker.step() is False
    assert "couldn't reach the job store" in caplog.text
    jobs.fail = False
    assert worker.step() is False  # recovers: the restart check runs again, then polls


def test_the_worker_thread_runs_a_job_as_soon_as_it_is_woken(parcels, parcel_id, fake_fetch):
    jobs = MemoryJobStore()
    worker = Worker(jobs, runner(parcels, fake_fetch), poll_s=60)  # only a wake can start it
    worker.start()
    try:
        time.sleep(0.05)  # the worker is now waiting
        job = start_analysis(parcel_id, jobs)
        worker.wake()
        deadline = time.monotonic() + 10
        while jobs.get(job.id).status != DONE and time.monotonic() < deadline:
            time.sleep(0.02)
        assert jobs.get(job.id).status == DONE
    finally:
        worker.stop()
    assert not worker._thread.is_alive()


def test_stopping_a_worker_that_never_started_is_fine(parcels, fake_fetch):
    Worker(MemoryJobStore(), runner(parcels, fake_fetch)).stop()
