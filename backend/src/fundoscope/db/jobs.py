"""Background jobs (FR-13): what the `jobs` table keeps, and the job store in PostGIS.

A job goes from queued to running (progress 0 to 100) to done or failed. The API creates
it, the worker claims and runs it, and anyone can read its status in the meantime, so every
call of `SqlJobStore` runs in its own short transaction: a progress update is visible as
soon as it is written.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from sqlalchemy import Engine, func, select, update

from fundoscope.db.models import JobRow
from fundoscope.db.session import session_scope

ANALYZE = "analyze"  # a parcel's history brought up to date (the only kind for now)

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"

WAITING = "Waiting for the worker"
STARTING = "Starting"
FINISHED = "Done"


@dataclass(frozen=True)
class Job:
    """One background job.

    Attributes:
        id: the job's id.
        parcel_id: the parcel it works on.
        kind: what it does (`ANALYZE`).
        status: `QUEUED`, `RUNNING`, `DONE`, or `FAILED`.
        progress: 0 to 100.
        step: what it is doing now, in plain language (or where it failed).
        error: why it failed, in plain language (None unless failed).
        processing_units: Sentinel Hub processing units spent (FR-25).
        credits: openEO credits spent.
        created_at, started_at, finished_at: when it was queued, claimed, and finished.
    """

    id: UUID
    parcel_id: UUID | None
    kind: str
    status: str
    progress: int = 0
    step: str | None = None
    error: str | None = None
    processing_units: float = 0.0
    credits: float = 0.0
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class JobStore(Protocol):
    """Where jobs are kept."""

    def create(self, parcel_id: UUID, kind: str = ANALYZE) -> Job: ...

    def get(self, job_id: UUID) -> Job | None: ...

    def active(self, parcel_id: UUID, kind: str = ANALYZE) -> Job | None:
        """The parcel's queued or running job of this kind, if any."""
        ...

    def claim(self) -> Job | None:
        """Mark the oldest queued job as running and return it (None if none is queued)."""
        ...

    def update(self, job_id: UUID, progress: int, step: str) -> None: ...

    def finish(
        self,
        job_id: UUID,
        *,
        error: str | None = None,
        processing_units: float = 0.0,
        credits: float = 0.0,
    ) -> Job:
        """Done (progress 100) or, with an `error`, failed."""
        ...

    def fail_running(self, error: str) -> int:
        """Mark every running job as failed (after a restart); returns how many."""
        ...

    def spent(self, parcel_id: UUID) -> tuple[float, float]:
        """Processing units and openEO credits spent on the parcel by all its jobs."""
        ...


def now() -> datetime:
    return datetime.now(UTC)


class SqlJobStore:
    """Jobs in the `jobs` table, each call in its own transaction."""

    def __init__(self, bind: Engine | None = None) -> None:
        self.bind = bind

    def create(self, parcel_id: UUID, kind: str = ANALYZE) -> Job:
        with session_scope(self.bind) as session:
            row = JobRow(
                parcel_id=parcel_id,
                kind=kind,
                status=QUEUED,
                progress=0,
                step=WAITING,
                processing_units=0.0,
                credits=0.0,
            )
            session.add(row)
            session.flush()
            session.refresh(row)
            return _job(row)

    def get(self, job_id: UUID) -> Job | None:
        with session_scope(self.bind) as session:
            row = session.get(JobRow, job_id)
            return _job(row) if row else None

    def active(self, parcel_id: UUID, kind: str = ANALYZE) -> Job | None:
        with session_scope(self.bind) as session:
            row = session.scalars(
                select(JobRow)
                .where(
                    JobRow.parcel_id == parcel_id,
                    JobRow.kind == kind,
                    JobRow.status.in_((QUEUED, RUNNING)),
                )
                .order_by(JobRow.created_at)
            ).first()
            return _job(row) if row else None

    def claim(self) -> Job | None:
        with session_scope(self.bind) as session:
            # SKIP LOCKED: two workers never claim the same job.
            row = session.scalars(
                select(JobRow)
                .where(JobRow.status == QUEUED)
                .order_by(JobRow.created_at, JobRow.id)
                .limit(1)
                .with_for_update(skip_locked=True)
            ).first()
            if row is None:
                return None
            row.status, row.step, row.started_at = RUNNING, STARTING, now()
            session.flush()
            return _job(row)

    def update(self, job_id: UUID, progress: int, step: str) -> None:
        with session_scope(self.bind) as session:
            session.execute(
                update(JobRow).where(JobRow.id == job_id).values(progress=progress, step=step)
            )

    def finish(
        self,
        job_id: UUID,
        *,
        error: str | None = None,
        processing_units: float = 0.0,
        credits: float = 0.0,
    ) -> Job:
        with session_scope(self.bind) as session:
            row = session.get_one(JobRow, job_id)
            row.status = FAILED if error else DONE
            if not error:
                row.progress, row.step = 100, FINISHED
            row.error = error
            row.processing_units, row.credits = processing_units, credits
            row.finished_at = now()
            session.flush()
            return _job(row)

    def fail_running(self, error: str) -> int:
        with session_scope(self.bind) as session:
            result = session.execute(
                update(JobRow)
                .where(JobRow.status == RUNNING)
                .values(status=FAILED, error=error, finished_at=now())
            )
            return result.rowcount

    def spent(self, parcel_id: UUID) -> tuple[float, float]:
        with session_scope(self.bind) as session:
            units, credits = session.execute(
                select(
                    func.coalesce(func.sum(JobRow.processing_units), 0.0),
                    func.coalesce(func.sum(JobRow.credits), 0.0),
                ).where(JobRow.parcel_id == parcel_id)
            ).one()
            return float(units), float(credits)


def _job(row: JobRow) -> Job:
    return Job(
        id=row.id,
        parcel_id=row.parcel_id,
        kind=row.kind,
        status=row.status,
        progress=row.progress,
        step=row.step,
        error=row.error,
        processing_units=row.processing_units,
        credits=row.credits,
        created_at=row.created_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
    )
