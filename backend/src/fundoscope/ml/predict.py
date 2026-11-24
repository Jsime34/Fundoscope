"""The crop of one campaign of a parcel, predicted with the trained model (FR-06).

Input: the campaign's 12 monthly composites (August to July) of the pixels at least 15 m
inside the parcel, NaN elsewhere, as for training. Each pixel gets the 150 features of
`fundoscope.ml.features` and a probability per class; the parcel's class is the one with the
highest mean probability, and the confidence is that probability (0 to 1).

A campaign still in progress has too few months for the features (at most 4 may be
missing), so it gets no prediction.
"""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import xarray as xr

from fundoscope.ml.features import FEATURE_COLUMNS, pixel_features


class CropModel(Protocol):
    """What prediction needs from a trained classifier (a scikit-learn model)."""

    classes_: np.ndarray

    def predict_proba(self, features: Any) -> np.ndarray: ...


@dataclass(frozen=True)
class CropPrediction:
    """The predicted crop of one campaign.

    Attributes:
        crop_class: the model class with the highest mean probability.
        confidence: that mean probability, 0 to 1.
        pixels: pixels that were classified.
        probabilities: mean probability of every class.
    """

    crop_class: str
    confidence: float
    pixels: int
    probabilities: dict[str, float]


def predict_crop(model: CropModel, stack: xr.DataArray) -> CropPrediction | None:
    """Predict the crop of a campaign; None if no pixel has enough clear months.

    Args:
        model: a trained classifier on `FEATURE_COLUMNS`.
        stack: reflectance 0..1 with dims (month, band, y, x), the 12 months of the
            campaign (August first), NaN outside the pixels to classify.

    Raises:
        FeatureError: the stack isn't 12 months starting in August, or lacks a band.
    """
    features = pixel_features(stack)
    if features.empty:
        return None
    mean = model.predict_proba(features[list(FEATURE_COLUMNS)].to_numpy()).mean(axis=0)
    best = int(np.argmax(mean))
    return CropPrediction(
        crop_class=str(model.classes_[best]),
        confidence=float(mean[best]),
        pixels=len(features),
        probabilities={str(name): float(p) for name, p in zip(model.classes_, mean, strict=True)},
    )
