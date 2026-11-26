"""Parcels and their reports: upload a parcel, start its analysis, list and reopen parcels.

Uploading stores the parcel in PostGIS (FR-02) without downloading anything. Starting an
analysis queues a background job (FR-13, see `api.jobs`) and answers at once with the job,
which brings the parcel's history up to date: the first analysis downloads 8 years (about
1-3 minutes), later ones only the dates since the last. The report is then read with
GET /api/parcels/{id}/history, rebuilt from the database with no satellite request.
"""

from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from pydantic import BaseModel

from fundoscope.api.jobs import job_store, wake_worker
from fundoscope.api.schemas import HistoryOut, JobOut, ParcelOut, history_out, job_out, parcel_out
from fundoscope.api.valley import valleys_folder
from fundoscope.db import JobStore, ParcelStore, SqlParcelStore, session_scope
from fundoscope.ingest import ParcelError
from fundoscope.pipeline.jobs import start_analysis
from fundoscope.pipeline.parcel_store import reopen_parcel, store_parcel

MAX_UPLOAD_BYTES = 5 * 1024 * 1024

router = APIRouter(prefix="/api", tags=["parcels"])


class StoredParcelOut(BaseModel):
    id: UUID
    name: str | None
    area_ha: float
    observed_through: date | None  # the newest day of satellite data stored (None: not analyzed)
    created_at: datetime


def parcel_store() -> Iterator[ParcelStore]:
    """The PostGIS store, in one transaction per request (committed if the request works)."""
    with session_scope() as session:
        yield SqlParcelStore(session)


@router.post(
    "/parcels",
    response_model=ParcelOut,
    responses={
        413: {"description": "The file is larger than 5 MB"},
        422: {"description": "The file is not a usable parcel"},
    },
)
def upload_parcel(
    file: UploadFile, store: Annotated[ParcelStore, Depends(parcel_store)]
) -> ParcelOut:
    """Validate and store a KML, KMZ, or GeoJSON parcel (nothing is downloaded yet).

    The same outline uploaded again, in any format, is the same parcel, with the same id.
    """
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "The file is larger than 5 MB.")
    try:
        parcel = store_parcel(data, file.filename or "upload", store)
    except ParcelError as exc:
        raise HTTPException(422, str(exc)) from exc
    return parcel_out(parcel, parcel.id)


@router.post(
    "/parcels/{parcel_id}/analyses",
    status_code=202,
    response_model=JobOut,
    responses={404: {"description": "No such parcel"}},
)
def start_parcel_analysis(
    parcel_id: UUID,
    request: Request,
    store: Annotated[ParcelStore, Depends(parcel_store)],
    jobs: Annotated[JobStore, Depends(job_store)],
) -> JobOut:
    """Queue an analysis of a stored parcel and answer at once with the job.

    Follow it with GET /api/jobs/{id}. If an analysis of the parcel is already queued or
    running, that job is returned instead of a new one.
    """
    if store.get(parcel_id) is None:
        raise HTTPException(404, "There is no such parcel.")
    job = start_analysis(parcel_id, jobs)
    wake_worker(request)
    return job_out(job)


@router.get("/parcels", response_model=list[StoredParcelOut])
def list_parcels(store: Annotated[ParcelStore, Depends(parcel_store)]) -> list[StoredParcelOut]:
    """The stored parcels, newest first."""
    return [
        StoredParcelOut(
            id=p.id,
            name=p.name,
            area_ha=round(p.area_ha, 2),
            observed_through=p.observed_through,
            created_at=p.created_at,
        )
        for p in store.parcels()
    ]


@router.get(
    "/parcels/{parcel_id}/history",
    response_model=HistoryOut,
    responses={404: {"description": "No such parcel, or it was never analyzed"}},
)
def get_parcel_history(
    parcel_id: UUID,
    store: Annotated[ParcelStore, Depends(parcel_store)],
    jobs: Annotated[JobStore, Depends(job_store)],
    valleys: Annotated[Path, Depends(valleys_folder)],
) -> HistoryOut:
    """A parcel's report, rebuilt from the database with no satellite request.

    Its cost is what all the parcel's analyses have spent so far (FR-25).
    """
    analysis = reopen_parcel(parcel_id, store, valleys)
    if analysis is None:
        raise HTTPException(404, "There is no report for this parcel yet: start an analysis.")
    return history_out(analysis.history, parcel_id=parcel_id, spent=jobs.spent(parcel_id))
