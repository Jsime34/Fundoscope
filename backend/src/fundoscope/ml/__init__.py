"""Crop classification.

Features, Random Forest, Transformer comparison, evaluation, prediction (FR-06, FR-16). Weeks
5-6, 11.

Submodules: `features` (the 150 features of a pixel and campaign), `label_curves` (compares
every label with its NDVI curve before training), `train_rf` (grouped cross-validation and the
final Random Forest), `evaluate` (macro-F1, F1 per class, confusion matrix).
"""

from fundoscope.ml.evaluate import (
    MIN_MACRO_F1,
    TARGET_MACRO_F1,
    Scores,
    parcel_prediction,
    score,
)
from fundoscope.ml.features import (
    BANDS,
    FEATURE_COLUMNS,
    INTERIOR_BUFFER_M,
    MAX_PIXELS_PER_SAMPLE,
    MONTHS,
    FeatureError,
    pixel_features,
    training_sample,
)
from fundoscope.ml.label_curves import (
    BARE_SOIL_NDVI,
    MAX_BARE_MONTHS,
    REVIEWED_NOTE,
    CurveCheck,
    CurveMismatch,
    campaign_curves,
    check_curves,
    curve_problem,
)
from fundoscope.ml.train_rf import (
    BROADER_CLASS,
    FOREST,
    GRID,
    N_FOLDS,
    CrossValidation,
    TrainingError,
    best,
    class_merges,
    cross_validate,
    fit,
    grouped_folds,
    training_table,
)

__all__ = [
    "BANDS",
    "BARE_SOIL_NDVI",
    "BROADER_CLASS",
    "FEATURE_COLUMNS",
    "FOREST",
    "GRID",
    "INTERIOR_BUFFER_M",
    "MAX_BARE_MONTHS",
    "MAX_PIXELS_PER_SAMPLE",
    "MIN_MACRO_F1",
    "MONTHS",
    "N_FOLDS",
    "REVIEWED_NOTE",
    "TARGET_MACRO_F1",
    "CrossValidation",
    "CurveCheck",
    "CurveMismatch",
    "FeatureError",
    "Scores",
    "TrainingError",
    "best",
    "campaign_curves",
    "class_merges",
    "check_curves",
    "cross_validate",
    "curve_problem",
    "fit",
    "grouped_folds",
    "parcel_prediction",
    "pixel_features",
    "score",
    "training_sample",
    "training_table",
]
