"""Hectares per class with 95 % ranges (FR-14).

The hectares of a class are its pixels x 0.01 ha (one 10 m pixel), so the classes add up to
the farmland. The 95 % range of each model class follows Olofsson et al. (2014), "Good
practices for estimating area and assessing accuracy of land change", Remote Sensing of
Environment 148: 42-57. The model's cross-validated confusion matrix says how often a
parcel-campaign predicted as class i was really class j. Weighting those proportions by the
share of the map in each class gives an error-adjusted area of each class and its standard
error; the range is the adjusted area +- 1.96 standard errors, never below 0.

Caveats, to state wherever the ranges are shown: the confusion matrix counts labeled
parcel-campaigns, not a random sample of the valley's pixels, so the ranges are approximate.
"Planted, not yet identifiable" and "no clear images" come from rules, not from the model, and
have no range.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fundoscope.valley.aoi import PIXEL_HA

Z_95 = 1.96


@dataclass(frozen=True)
class ClassArea:
    """The area of one class.

    Attributes:
        name: the class.
        pixels: mapped pixels.
        hectares: mapped hectares (pixels x 0.01).
        adjusted: error-adjusted hectares (None for classes outside the model).
        low, high: the 95 % range of the adjusted hectares (None outside the model).
    """

    name: str
    pixels: int
    hectares: float
    adjusted: float | None = None
    low: float | None = None
    high: float | None = None


def class_areas(pixels: Mapping[str, int], confusion: pd.DataFrame) -> list[ClassArea]:
    """The area of every class of a map, in the order of `pixels`.

    Args:
        pixels: mapped pixels per class: the model classes and the rule classes.
        confusion: cross-validated counts with the labeled class as rows and the predicted
            class as columns (as in the model card). Its classes are the model classes.

    Raises:
        ValueError: a class of the confusion matrix has no pixel count.
    """
    model_classes = list(confusion.columns)
    missing = [name for name in model_classes if name not in pixels]
    if missing:
        raise ValueError(f"No pixel count for the model classes {', '.join(missing)}.")
    mapped = pd.Series([pixels[name] for name in model_classes], index=model_classes, dtype=float)
    adjusted, error = olofsson(mapped, confusion)
    areas = []
    for name, count in pixels.items():
        if name not in model_classes:
            areas.append(ClassArea(name, count, count * PIXEL_HA))
            continue
        middle = adjusted[name] * PIXEL_HA
        half = Z_95 * error[name] * PIXEL_HA
        areas.append(
            ClassArea(name, count, count * PIXEL_HA, middle, max(middle - half, 0.0), middle + half)
        )
    return areas


def olofsson(mapped: pd.Series, confusion: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Error-adjusted area of each class and its standard error, in the units of `mapped`.

    Args:
        mapped: mapped area per model class.
        confusion: counts, labeled class as rows and predicted class as columns, over the same
            classes as `mapped`.

    A class the cross-validation never predicted gives no information about its own errors:
    its mapped area is taken as correct.
    """
    total = mapped.sum()
    if total == 0:
        zero = pd.Series(0.0, index=mapped.index)
        return zero, zero.copy()
    weight = (mapped / total).to_numpy()  # W_i, by mapped (predicted) class
    counts = confusion.loc[mapped.index, mapped.index].to_numpy(float).T  # n_ij: map i, ref j
    per_map = counts.sum(axis=1)  # n_i.
    never_predicted = per_map == 0
    counts[never_predicted] = np.eye(len(mapped))[never_predicted]
    per_map[never_predicted] = 1
    share = counts / per_map[:, None]  # n_ij / n_i.
    adjusted = total * (weight[:, None] * share).sum(axis=0)
    terms = weight[:, None] ** 2 * share * (1 - share)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(per_map[:, None] > 1, terms / (per_map[:, None] - 1), 0.0)
    error = total * np.sqrt(terms.sum(axis=0))
    return pd.Series(adjusted, index=mapped.index), pd.Series(error, index=mapped.index)
