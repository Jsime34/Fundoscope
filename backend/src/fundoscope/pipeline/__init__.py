"""Orchestration.

Combines the pure domain modules with the adapters (db, sentinel): e.g. "analyze a parcel" =
check the DB cache, fetch missing Sentinel-2 dates, compute campaigns, classify, flag, store.
Used by both the CLI and the API worker so the logic lives in one place. Weeks 3-10.

Submodules: `parcel_history` (file -> campaign table; no database yet), `training_data`
(labels -> training pixels, within the quota), `train_model` (training pixels -> saved crop
model and season-to-date models, model card, and cross-validation report), `valley_refresh`
(the valley's new composites -> hectares per crop class, FR-14).
"""

from fundoscope.pipeline.parcel_history import ParcelHistory, parcel_history
from fundoscope.pipeline.train_model import (
    CARD_FILE,
    MODEL_FILE,
    REPORT_FILE,
    SEASON_DIR,
    SEASON_MONTHS,
    TrainedModel,
    format_cv_report,
    load_crop_model,
    season_model_file,
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
from fundoscope.pipeline.valley_refresh import (
    Plan,
    Refresh,
    ValleyError,
    load_valley,
    plan_refresh,
    refresh_valley,
    valley_dir,
)

__all__ = [
    "CARD_FILE",
    "DONE",
    "FAILED",
    "LOG_FILE",
    "MODEL_FILE",
    "NO_CLEAR_PIXELS",
    "REPORT_FILE",
    "SEASON_DIR",
    "SEASON_MONTHS",
    "TOO_NARROW",
    "Extraction",
    "ParcelHistory",
    "Plan",
    "Refresh",
    "Task",
    "TaskResult",
    "TrainedModel",
    "ValleyError",
    "extract_training_data",
    "extraction_tasks",
    "format_cv_report",
    "load_crop_model",
    "load_training_pixels",
    "load_valley",
    "parcel_history",
    "pending",
    "plan_refresh",
    "refresh_valley",
    "season_model_file",
    "train_crop_model",
    "valley_dir",
]
