"""Crop classification.

Features, Random Forest, Transformer comparison, evaluation, prediction (FR-06, FR-16). Weeks
5-6, 11.

Submodules: `features` (the 150 features of a pixel and campaign).
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

__all__ = [
    "BANDS",
    "FEATURE_COLUMNS",
    "INTERIOR_BUFFER_M",
    "MAX_PIXELS_PER_SAMPLE",
    "MONTHS",
    "FeatureError",
    "pixel_features",
    "training_sample",
]
