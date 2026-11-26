"""Parcel histories: upload a parcel file and get its history; list and reopen stored parcels.

POST /api/history runs synchronously (about 1-3 minutes for 8 years the first time). Every
parcel and its history are stored in PostGIS (FR-02): uploading the same outline again only
downloads the dates after the last stored one, and GET /api/parcels/{id}/history reopens a
stored report without any satellite request. Background jobs with progress (FR-13) come next.
"""

from collections.abc import Iterator
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel

from fundoscope.api.schemas import HistoryOut, history_out
from fundoscope.api.valley import valleys_folder
from fundoscope.config import get_settings
from fundoscope.db import ParcelStore, SqlParcelStore, session_scope
from fundoscope.ingest import ParcelError
from fundoscope.ml import CropModel
from fundoscope.pipeline import MODEL_FILE, load_crop_model
from fundoscope.pipeline.parcel_history import CompositeFetcher, Fetcher
from fundoscope.pipeline.parcel_store import analyze_parcel, reopen_parcel
from fundoscope.sentinel import SentinelError, fetch_daily_stats, fetch_monthly_composites

MAX_UPLOAD_BYTES = 5 * 1024 * 1024

router = APIRouter(prefix="/api", tags=["parcel history"])


class StoredParcelOut(BaseModel):
    id: UUID
    name: str | None
    area_ha: float
    observed_through: date | None  # the newest day of satellite data stored
    created_at: datetime


# Dependencies: tests override these with fakes so no request reaches Copernicus or the database.
def daily_fetcher() -> Fetcher:
    return fetch_daily_stats


def composites_fetcher() -> CompositeFetcher:
    return fetch_monthly_composites


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


def parcel_store() -> Iterator[ParcelStore]:
    """The PostGIS store, in one transaction per request (committed if the request works)."""
    with session_scope() as session:
        yield SqlParcelStore(session)


def today() -> date:
    return date.today()


@router.post(
    "/history",
    response_model=HistoryOut,
    responses={
        413: {"description": "The file is larger than 5 MB"},
        422: {"description": "The file is not a usable parcel"},
        502: {"description": "The satellite service failed"},
    },
)
def post_history(
    file: UploadFile,
    fetch: Annotated[Fetcher, Depends(daily_fetcher)],
    fetch_composites: Annotated[CompositeFetcher, Depends(composites_fetcher)],
    model: Annotated[CropModel | None, Depends(crop_model)],
    version: Annotated[str | None, Depends(model_version)],
    store: Annotated[ParcelStore, Depends(parcel_store)],
    day: Annotated[date, Depends(today)],
    valleys: Annotated[Path, Depends(valleys_folder)],
) -> HistoryOut:
    """Validate and store the parcel, bring its history up to today, and summarize it.

    With a trained crop model, each complete campaign also gets its predicted crop. A parcel
    analyzed before only downloads what is new since then.
    """
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "The file is larger than 5 MB.")
    try:
        analysis = analyze_parcel(
            data,
            file.filename or "upload",
            store,
            today=day,
            fetch=fetch,
            fetch_composites=fetch_composites,
            model=model,
            model_version=version,
            valleys=valleys,
        )
    except ParcelError as exc:
        raise HTTPException(422, str(exc)) from exc
    except SentinelError as exc:
        raise HTTPException(502, f"The satellite service failed: {exc}") from exc
    return history_out(
        analysis.history, parcel_id=analysis.parcel_id, cached=not analysis.downloaded
    )


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
    responses={404: {"description": "No stored report for this parcel"}},
)
def get_parcel_history(
    parcel_id: UUID,
    store: Annotated[ParcelStore, Depends(parcel_store)],
    valleys: Annotated[Path, Depends(valleys_folder)],
) -> HistoryOut:
    """Reopen a stored report: rebuilt from the database, with no satellite request."""
    analysis = reopen_parcel(parcel_id, store, valleys)
    if analysis is None:
        raise HTTPException(404, "There is no stored report for this parcel.")
    return history_out(analysis.history, parcel_id=parcel_id, cached=True)
