"""Parcel and job stores in memory: for tests and for runs without a database."""

import threading
from dataclasses import replace
from datetime import UTC, date, datetime
from uuid import UUID, uuid4

import pandas as pd
from shapely.geometry.base import BaseGeometry

from fundoscope.db.jobs import (
    ANALYZE,
    DONE,
    FAILED,
    FINISHED,
    QUEUED,
    RUNNING,
    STARTING,
    WAITING,
    Job,
    now,
)
from fundoscope.db.records import (
    OBSERVATION_COLUMNS,
    StoredCampaign,
    StoredHistory,
    StoredParcel,
    empty_observations,
)


class MemoryParcelStore:
    """Keeps everything in dictionaries; lost when the process ends."""

    def __init__(self) -> None:
        self._parcels: dict[UUID, StoredParcel] = {}
        self._observations: dict[UUID, pd.DataFrame] = {}
        self._campaigns: dict[UUID, dict[str, StoredCampaign]] = {}

    def find(self, key: str) -> StoredParcel | None:
        return next((p for p in self._parcels.values() if p.key == key), None)

    def add(
        self,
        key: str,
        name: str | None,
        geometry: BaseGeometry,
        utm_epsg: int,
        area_ha: float,
        repaired: bool,
    ) -> StoredParcel:
        parcel = StoredParcel(
            uuid4(), key, name, geometry, utm_epsg, area_ha, repaired, None, datetime.now(UTC)
        )
        self._parcels[parcel.id] = parcel
        return parcel

    def get(self, parcel_id: UUID) -> StoredParcel | None:
        return self._parcels.get(parcel_id)

    def parcels(self) -> list[StoredParcel]:
        """Newest first (the order they were added: the clock may not tell them apart)."""
        return list(reversed(self._parcels.values()))

    def load(self, parcel_id: UUID) -> StoredHistory:
        parcel = self._parcels[parcel_id]
        observations = self._observations.get(parcel_id, empty_observations())
        campaigns = dict(self._campaigns.get(parcel_id, {}))
        return StoredHistory(observations.copy(), parcel.observed_through, campaigns)

    def save(
        self,
        parcel_id: UUID,
        observations: pd.DataFrame,
        observed_through: date,
        campaigns: dict[str, StoredCampaign],
    ) -> None:
        new = observations[list(OBSERVATION_COLUMNS)]
        old = self._observations.get(parcel_id)
        if old is not None and len(old):
            new = pd.concat([old[~old.index.isin(new.index)], new])
        self._observations[parcel_id] = new.sort_index()
        self._campaigns.setdefault(parcel_id, {}).update(campaigns)
        parcel = self._parcels[parcel_id]
        self._parcels[parcel_id] = replace(parcel, observed_through=observed_through)


class MemoryJobStore:
    """Jobs in a dictionary, safe to share between threads; lost when the process ends."""

    def __init__(self) -> None:
        self._jobs: dict[UUID, Job] = {}
        self._lock = threading.Lock()

    def create(self, parcel_id: UUID, kind: str = ANALYZE) -> Job:
        job = Job(uuid4(), parcel_id, kind, QUEUED, step=WAITING, created_at=now())
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: UUID) -> Job | None:
        return self._jobs.get(job_id)

    def active(self, parcel_id: UUID, kind: str = ANALYZE) -> Job | None:
        return next(
            (
                job
                for job in self._jobs.values()
                if job.parcel_id == parcel_id
                and job.kind == kind
                and job.status in (QUEUED, RUNNING)
            ),
            None,
        )

    def claim(self) -> Job | None:
        with self._lock:
            job = next((job for job in self._jobs.values() if job.status == QUEUED), None)
            if job is None:
                return None
            job = replace(job, status=RUNNING, step=STARTING, started_at=now())
            self._jobs[job.id] = job
            return job

    def update(self, job_id: UUID, progress: int, step: str) -> None:
        with self._lock:
            self._jobs[job_id] = replace(self._jobs[job_id], progress=progress, step=step)

    def finish(
        self,
        job_id: UUID,
        *,
        error: str | None = None,
        processing_units: float = 0.0,
        credits: float = 0.0,
    ) -> Job:
        with self._lock:
            job = self._jobs[job_id]
            done = {} if error else {"progress": 100, "step": FINISHED}
            job = replace(
                job,
                status=FAILED if error else DONE,
                error=error,
                processing_units=processing_units,
                credits=credits,
                finished_at=now(),
                **done,
            )
            self._jobs[job_id] = job
            return job

    def fail_running(self, error: str) -> int:
        with self._lock:
            running = [job for job in self._jobs.values() if job.status == RUNNING]
            for job in running:
                self._jobs[job.id] = replace(job, status=FAILED, error=error, finished_at=now())
            return len(running)

    def spent(self, parcel_id: UUID) -> tuple[float, float]:
        jobs = [job for job in self._jobs.values() if job.parcel_id == parcel_id]
        return sum(job.processing_units for job in jobs), sum(job.credits for job in jobs)
