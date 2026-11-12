"""Scores of the crop model (FR-06): macro-F1, F1 per class, and the confusion matrix.

The unit is a parcel and campaign: its class is the one with the highest average probability
over its pixels, and the confidence is that probability (0 to 1).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

MIN_MACRO_F1 = 0.65  # FR-06 minimum
TARGET_MACRO_F1 = 0.75


@dataclass(frozen=True)
class Scores:
    """How well predicted classes match the labels.

    Attributes:
        macro_f1: mean of the per-class F1 scores, so small classes count as much as big ones.
        per_class_f1: F1 of each class (0 when a class is never predicted correctly).
        confusion: counts with the labeled class as rows and the predicted class as columns.
        samples: parcel-campaigns scored.
    """

    macro_f1: float
    per_class_f1: dict[str, float]
    confusion: pd.DataFrame
    samples: int


def parcel_prediction(probabilities: pd.DataFrame) -> pd.DataFrame:
    """The class and confidence of each parcel-campaign.

    Args:
        probabilities: one row per pixel with a probability column per class, indexed by
            the pixel's parcel-campaign (any index; rows with the same index are averaged).

    Returns:
        One row per parcel-campaign with `crop_class` and `confidence`.
    """
    mean = probabilities.groupby(level=list(range(probabilities.index.nlevels))).mean()
    return pd.DataFrame({"crop_class": mean.idxmax(axis=1), "confidence": mean.max(axis=1)})


def score(truth: Sequence[str], predicted: Sequence[str], classes: Sequence[str]) -> Scores:
    """Compare predicted classes with the labeled ones, over `classes`."""
    labels = list(classes)
    per_class = f1_score(truth, predicted, labels=labels, average=None, zero_division=0)
    matrix = confusion_matrix(truth, predicted, labels=labels)
    return Scores(
        macro_f1=float(per_class.mean()),
        per_class_f1={name: float(value) for name, value in zip(labels, per_class, strict=True)},
        confusion=pd.DataFrame(matrix, index=labels, columns=labels),
        samples=len(truth),
    )
