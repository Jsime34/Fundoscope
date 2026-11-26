"""What every job store must do (in memory and in PostGIS), FR-13."""

import pytest
from shapely.geometry import MultiPolygon, box
from sqlalchemy import select

from fundoscope.db import ANALYZE, DONE, FAILED, QUEUED, RUNNING, SqlJobStore, session_scope
from fundoscope.db.jobs import FINISHED, STARTING, WAITING
from fundoscope.db.models import JobRow

SQUARE = MultiPolygon([box(-75.731, -14.071, -75.729, -14.069)])  # public farmland near Ica


def parcel(parcels, key="abc"):
    return parcels.add(key, "Lote demo", SQUARE, 32718, 4.6, False).id


def test_a_new_job_is_queued(any_jobs):
    jobs, parcels = any_jobs
    parcel_id = parcel(parcels)

    job = jobs.create(parcel_id)

    assert (job.parcel_id, job.kind, job.status, job.progress) == (parcel_id, ANALYZE, QUEUED, 0)
    assert job.step == WAITING and job.error is None
    assert (job.processing_units, job.credits) == (0, 0)
    assert job.created_at is not None and job.started_at is None and job.finished_at is None
    assert jobs.get(job.id) == job


def test_an_unknown_job_is_none(any_jobs):
    from uuid import uuid4

    jobs, _ = any_jobs

    assert jobs.get(uuid4()) is None


def test_jobs_are_claimed_oldest_first_and_once(any_jobs):
    jobs, parcels = any_jobs
    first = jobs.create(parcel(parcels, "a"))
    second = jobs.create(parcel(parcels, "b"))

    claimed = jobs.claim()

    assert claimed.id == first.id
    assert (claimed.status, claimed.step) == (RUNNING, STARTING) and claimed.started_at
    assert jobs.get(first.id) == claimed
    assert jobs.claim().id == second.id
    assert jobs.claim() is None


def test_a_parcel_has_one_active_job_at_most(any_jobs):
    jobs, parcels = any_jobs
    parcel_id = parcel(parcels)
    assert jobs.active(parcel_id) is None

    job = jobs.create(parcel_id)
    assert jobs.active(parcel_id) == job
    assert jobs.active(parcel_id, "valley_refresh") is None

    jobs.claim()
    assert jobs.active(parcel_id).status == RUNNING

    jobs.finish(job.id)
    assert jobs.active(parcel_id) is None


def test_progress_is_visible_at_once(any_jobs):
    jobs, parcels = any_jobs
    job = jobs.create(parcel(parcels))
    jobs.claim()

    jobs.update(job.id, 50, "Downloading satellite data: 5 of 10 requests done")

    seen = jobs.get(job.id)
    assert (seen.progress, seen.step) == (50, "Downloading satellite data: 5 of 10 requests done")


def test_a_finished_job_is_done_with_its_cost(any_jobs):
    jobs, parcels = any_jobs
    job = jobs.create(parcel(parcels))
    jobs.claim()

    done = jobs.finish(job.id, processing_units=21.4, credits=63.0)

    assert (done.status, done.progress, done.step, done.error) == (DONE, 100, FINISHED, None)
    assert (done.processing_units, done.credits) == (21.4, 63.0)
    assert done.finished_at is not None and jobs.get(job.id) == done


def test_a_failed_job_keeps_where_it_stopped(any_jobs):
    jobs, parcels = any_jobs
    job = jobs.create(parcel(parcels))
    jobs.claim()
    jobs.update(job.id, 23, "Downloading satellite data: 2 of 10 requests done")

    failed = jobs.finish(job.id, error="The satellite service failed (boom). Try again later.")

    assert (failed.status, failed.progress) == (FAILED, 23)
    assert failed.step == "Downloading satellite data: 2 of 10 requests done"
    assert failed.error.startswith("The satellite service failed")


def test_running_jobs_are_failed_after_a_restart(any_jobs):
    jobs, parcels = any_jobs
    running = jobs.create(parcel(parcels, "a"))
    queued = jobs.create(parcel(parcels, "b"))
    jobs.claim()

    assert jobs.fail_running("The server restarted.") == 1

    assert (jobs.get(running.id).status, jobs.get(running.id).error) == (
        FAILED,
        "The server restarted.",
    )
    assert jobs.get(queued.id).status == QUEUED
    assert jobs.fail_running("again") == 0


def test_the_cost_of_a_parcel_adds_up_its_jobs(any_jobs):
    jobs, parcels = any_jobs
    parcel_id, other = parcel(parcels, "a"), parcel(parcels, "b")
    assert jobs.spent(parcel_id) == (0, 0)
    for units, credits in ((21.4, 63.0), (0.5, 7.0)):
        job = jobs.create(parcel_id)
        jobs.claim()
        jobs.finish(job.id, processing_units=units, credits=credits)

    assert jobs.spent(parcel_id) == pytest.approx((21.9, 70.0))
    assert jobs.spent(other) == (0, 0)


def test_a_job_locked_by_another_worker_is_skipped(db_engine):
    from fundoscope.db import AutocommitParcelStore

    parcels, jobs = AutocommitParcelStore(db_engine), SqlJobStore(db_engine)
    first = jobs.create(parcel(parcels, "a"))
    second = jobs.create(parcel(parcels, "b"))

    with session_scope(db_engine) as other_worker:  # holds the first job's row lock
        other_worker.scalars(select(JobRow).where(JobRow.id == first.id).with_for_update()).one()

        assert jobs.claim().id == second.id
        assert jobs.claim() is None

    assert jobs.claim().id == first.id
