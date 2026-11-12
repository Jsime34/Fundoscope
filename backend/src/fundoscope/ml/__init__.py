"""Crop classification.

Features, Random Forest, Transformer comparison, evaluation, prediction (FR-06, FR-16). Weeks
5-6, 11.

Submodules: `features` (the 150 features of a pixel and campaign), `label_curves` (compares
every label with its NDVI curve before training).
"""

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

__all__ = [
    "BANDS",
    "BARE_SOIL_NDVI",
    "FEATURE_COLUMNS",
    "INTERIOR_BUFFER_M",
    "MAX_BARE_MONTHS",
    "MAX_PIXELS_PER_SAMPLE",
    "MONTHS",
    "REVIEWED_NOTE",
    "CurveCheck",
    "CurveMismatch",
    "FeatureError",
    "campaign_curves",
    "check_curves",
    "curve_problem",
    "pixel_features",
    "training_sample",
]
