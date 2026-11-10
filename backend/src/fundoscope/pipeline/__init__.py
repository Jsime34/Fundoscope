"""Orchestration.

Combines the pure domain modules with the adapters (db, sentinel): e.g. "analyze a parcel" =
check the DB cache, fetch missing Sentinel-2 dates, compute campaigns, classify, flag, store.
Used by both the CLI and the API worker so the logic lives in one place. Weeks 3-10.

Submodules: `parcel_history` (file -> campaign table; no database yet), `training_data`
(labels -> training pixels, within the quota).
"""

from fundoscope.pipeline.parcel_history import ParcelHistory, parcel_history
from fundoscope.pipeline.training_data import (
    DONE,
    FAILED,
    LOG_FILE,
    NO_CLEAR_PIXELS,
    TOO_NARROW,
    Extraction,
    Task,
    TaskResult,
    extract_training_data,
    extraction_tasks,
    pending,
)

__all__ = [
    "DONE",
    "FAILED",
    "LOG_FILE",
    "NO_CLEAR_PIXELS",
    "TOO_NARROW",
    "Extraction",
    "ParcelHistory",
    "Task",
    "TaskResult",
    "extract_training_data",
    "extraction_tasks",
    "parcel_history",
    "pending",
]
