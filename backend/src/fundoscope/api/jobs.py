"""Background jobs (FR-13): follow a job, and the worker that runs them.

GET /api/jobs/{id} gives a job's status and progress; POST /api/parcels/{id}/analyses (in
`api.history`) queues one. The worker runs inside the API process: `api.main` starts it with
the app and stops it on shutdown. It analyzes parcels with the real downloaders, the saved
crop model, and the PostGIS store.
"""

from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request

from fundoscope.api.schemas import JobOut, job_out
from fundoscope.api.valley import valleys_folder
from fundoscope.config import get_settings
from fundoscope.db import AutocommitParcelStore, JobStore, SqlJobStore
from fundoscope.ml import CropModel
from fundoscope.pipeline import MODEL_FILE, load_crop_model
from fundoscope.pipeline.jobs import Worker
from fundoscope.pipeline.parcel_history import Progress
from fundoscope.pipeline.parcel_store import StoredAnalysis, update_parcel
from fundoscope.sentinel import fetch_daily_stats, fetch_monthly_composites

router = APIRouter(prefix="/api", tags=["jobs"])


def job_store() -> JobStore:
    """The jobs in PostGIS (tests override this with an in-memory store)."""
    return SqlJobStore()


@router.get(
    "/jobs/{job_id}",
    response_model=JobOut,
    responses={404: {"description": "No such job"}},
)
def get_job(job_id: UUID, jobs: Annotated[JobStore, Depends(job_store)]) -> JobOut:
    """A job's status (queued, running, done, failed), progress (0-100), and cost.

    Once it is done, `report` says where to get the parcel's report.
    """
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "There is no such job.")
    return job_out(job)


def wake_worker(request: Request) -> None:
    """Have the app's worker look for queued jobs now (no worker in tests: nothing happens)."""
    worker = getattr(request.app.state, "worker", None)
    if worker is not None:
        worker.wake()


def make_worker() -> Worker:
    """The worker that `api.main` starts with the app."""
    return Worker(SqlJobStore(), analyze_with_copernicus)


def analyze_with_copernicus(parcel_id: UUID, on_progress: Progress) -> StoredAnalysis:
    """One parcel analysis with the real downloaders, the saved model, and PostGIS."""
    return update_parcel(
        parcel_id,
        AutocommitParcelStore(),
        today=date.today(),
        fetch=fetch_daily_stats,
        fetch_composites=fetch_monthly_composites,
        model=crop_model(),
        model_version=model_version(),
        valleys=valleys_folder(),
        on_progress=on_progress,
    )


def crop_model() -> CropModel | None:
    """The saved crop model (None until `fundoscope model train` has run)."""
    return cached_model(get_settings().models_dir)


def model_version() -> str | None:
    """When the saved model was written: crops predicted by another version are redone."""
    path = get_settings().models_dir / MODEL_FILE
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat() if path.exists() else None


def cached_model(folder: Path) -> CropModel | None:
    """Load the model once; a retrained model (newer file) is loaded again."""
    path = folder / MODEL_FILE
    return _load(folder, path.stat().st_mtime if path.exists() else None)


@lru_cache(maxsize=2)
def _load(folder: Path, modified: float | None) -> CropModel | None:
    return load_crop_model(folder)
