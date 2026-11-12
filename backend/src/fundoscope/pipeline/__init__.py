"""Orchestration.

Combines the pure domain modules with the adapters (db, sentinel): e.g. "analyze a parcel" =
check the DB cache, fetch missing Sentinel-2 dates, compute campaigns, classify, flag, store.
Used by both the CLI and the API worker so the logic lives in one place. Weeks 3-10.

Submodules: `parcel_history` (file -> campaign table; no database yet), `training_data`
(labels -> training pixels, within the quota), `train_model` (training pixels -> saved crop
model, model card, and cross-validation report).
"""

from fundoscope.pipeline.parcel_history import ParcelHistory, parcel_history
from fundoscope.pipeline.train_model import (
    CARD_FILE,
    MODEL_FILE,
    REPORT_FILE,
    TrainedModel,
    format_cv_report,
    train_crop_model,
)
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
    load_training_pixels,
    pending,
)

__all__ = [
    "CARD_FILE",
    "DONE",
    "FAILED",
    "LOG_FILE",
    "MODEL_FILE",
    "NO_CLEAR_PIXELS",
    "REPORT_FILE",
    "TOO_NARROW",
    "Extraction",
    "ParcelHistory",
    "Task",
    "TaskResult",
    "TrainedModel",
    "extract_training_data",
    "extraction_tasks",
    "format_cv_report",
    "load_training_pixels",
    "parcel_history",
    "pending",
    "train_crop_model",
]
