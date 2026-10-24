"""POST /api/history: upload a parcel file, get its campaign-by-campaign history.

The request runs synchronously (about 1-3 minutes for 8 years on the first run, then
cached). Background jobs with progress (FR-13) replace this once the `db` module exists.
"""

from datetime import date
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from fundoscope.api import cache
from fundoscope.api.schemas import HistoryOut, history_out
from fundoscope.config import get_settings
from fundoscope.ingest import ParcelError
from fundoscope.pipeline import parcel_history
from fundoscope.pipeline.parcel_history import CompositeFetcher, Fetcher
from fundoscope.sentinel import SentinelError, fetch_daily_stats, fetch_monthly_composites

MAX_UPLOAD_BYTES = 5 * 1024 * 1024

router = APIRouter(prefix="/api", tags=["parcel history"])


# Dependencies: tests override these with fakes so no request reaches Copernicus.
def daily_fetcher() -> Fetcher:
    return fetch_daily_stats


def composites_fetcher() -> CompositeFetcher:
    return fetch_monthly_composites


def cache_folder() -> Path:
    return get_settings().data_dir / "private" / "cache"


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
    folder: Annotated[Path, Depends(cache_folder)],
    day: Annotated[date, Depends(today)],
) -> HistoryOut:
    """Validate the parcel, download 8 years of Sentinel-2 data, and summarize each campaign."""
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "The file is larger than 5 MB.")

    key = cache.cache_key(data, day)
    if (cached := cache.load(folder, key)) is not None:
        cached.summary.cached = True
        return cached

    try:
        history = parcel_history(
            data, file.filename or "upload", end=day, fetch=fetch, fetch_composites=fetch_composites
        )
    except ParcelError as exc:
        raise HTTPException(422, str(exc)) from exc
    except SentinelError as exc:
        raise HTTPException(502, f"The satellite service failed: {exc}") from exc

    result = history_out(history)
    cache.save(folder, key, result)
    return result
