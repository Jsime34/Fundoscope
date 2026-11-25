"""Classify every farmland pixel of a valley tile (FR-14).

Each farmland pixel gets the class with the highest probability of the crop model for the
months it has: the full model for a finished campaign, or the season-to-date model of the
first n months for a campaign in progress. A pixel missing more than a third of those months
(no clear view) is counted as "no clear images", so that the hectares of all classes still add
up to the farmland.

In a campaign in progress, a field sown a few weeks ago can't be told apart yet. A pixel whose
top probability is below 0.5 and whose NDVI is rising from bare soil is counted as "planted,
not yet identifiable": its NDVI was at most 0.25 (bare soil) in an earlier month of the
campaign and is now at least 0.1 higher. The rule needs two months or more.

The tile is classified a block of rows at a time, so memory stays small.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from fundoscope.ml import BARE_SOIL_NDVI, MONTHS, CropModel, feature_columns, pixel_features

PLANTED = "planted, not yet identifiable"
NO_CLEAR_IMAGES = "no clear images"
NOT_FARMLAND = 0  # the code of pixels outside the farmland
MIN_CONFIDENCE = 0.5
NDVI_RISE = 0.1
CHUNK_ROWS = 100


@dataclass(frozen=True)
class ClassMap:
    """The class of every pixel of a tile.

    Attributes:
        codes: (y, x) uint8: `NOT_FARMLAND` (0) outside the farmland, otherwise 1 + the
            index of the pixel's class in `classes`.
        classes: the class names: the model's classes, then `PLANTED` and `NO_CLEAR_IMAGES`.
    """

    codes: NDArray[np.uint8]
    classes: tuple[str, ...]

    def counts(self) -> dict[str, int]:
        """Pixels of each class, zeros included."""
        found = np.bincount(self.codes.ravel(), minlength=len(self.classes) + 1)
        return {name: int(found[code]) for code, name in enumerate(self.classes, start=1)}


def map_classes(model: CropModel) -> tuple[str, ...]:
    """The classes of a map made with `model`."""
    return (*(str(name) for name in model.classes_), PLANTED, NO_CLEAR_IMAGES)


def classify_tile(
    stack: xr.DataArray,
    farmland: NDArray[np.bool_],
    model: CropModel,
    *,
    in_progress: bool,
    chunk_rows: int = CHUNK_ROWS,
) -> ClassMap:
    """The class of every farmland pixel of a tile.

    Args:
        stack: reflectance 0..1 with dims (month, band, y, x): the first n months of one
            campaign, August first, as the model was trained.
        farmland: (y, x) mask of the pixels to classify.
        model: the crop model for n months (`n_features_in_` must match).
        in_progress: the campaign isn't over: apply the "planted, not yet identifiable" rule.
        chunk_rows: rows classified at a time.

    Raises:
        ValueError: the mask doesn't match the stack, or the model expects other months.
        FeatureError: the stack isn't consecutive months from August, or lacks a band.
    """
    months = stack.sizes["month"]
    columns = list(feature_columns(months))
    expected = getattr(model, "n_features_in_", len(columns))
    if expected != len(columns):
        raise ValueError(
            f"The model expects {expected} features; {months} months give {len(columns)}."
        )
    if farmland.shape != (stack.sizes["y"], stack.sizes["x"]):
        raise ValueError(
            f"The farmland mask {farmland.shape} doesn't match the tile "
            f"{(stack.sizes['y'], stack.sizes['x'])}."
        )
    classes = map_classes(model)
    codes = np.where(farmland, classes.index(NO_CLEAR_IMAGES) + 1, NOT_FARMLAND).astype(np.uint8)
    ndvi_columns = [f"ndvi_{month}" for month in MONTHS[:months]]
    for start in range(0, stack.sizes["y"], chunk_rows):
        rows = slice(start, start + chunk_rows)
        chunk = stack.isel(y=rows).where(xr.DataArray(farmland[rows], dims=("y", "x")))
        features = pixel_features(chunk)
        if features.empty:
            continue
        probabilities = model.predict_proba(features[columns].to_numpy())
        best = probabilities.argmax(axis=1)
        if in_progress:
            curve = features[ndvi_columns].to_numpy().T
            planted = (probabilities.max(axis=1) < MIN_CONFIDENCE) & rising_from_bare(curve)
            best = np.where(planted, classes.index(PLANTED), best)
        row = pd.Index(chunk["y"].to_numpy()).get_indexer(features.index.get_level_values("y"))
        col = pd.Index(chunk["x"].to_numpy()).get_indexer(features.index.get_level_values("x"))
        codes[start + row, col] = best + 1
    return ClassMap(codes, classes)


def rising_from_bare(curve: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Pixels whose NDVI rose from bare soil: at most 0.25 before, now at least 0.1 higher.

    Args:
        curve: monthly NDVI with dims (month, pixel), the latest month last.
    """
    if curve.shape[0] < 2:
        return np.zeros(curve.shape[1], bool)
    low = curve[:-1].min(axis=0)
    return (low <= BARE_SOIL_NDVI) & (curve[-1] - low >= NDVI_RISE)
