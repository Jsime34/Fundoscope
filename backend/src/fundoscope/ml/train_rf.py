"""Random Forest crop model: training table, grouped cross-validation, final fit (FR-06).

Every training pixel has the 150 features of `fundoscope.ml.features` and the class of its
label. Cross-validation keeps all the pixels of a parcel, from every campaign, in the same
fold, so each fold is scored on parcels the model has never seen: pixels of one field are
nearly identical, and splitting them between training and test would inflate the score.

A few settings (`GRID`) are compared by the cross-validated macro-F1 per parcel-campaign,
and the best one is refit on all training parcels. The field-verified parcels never enter
here: they are scored once, with the final model.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

from fundoscope.ingest import CROP_CLASSES, MIN_PARCELS_PER_CLASS
from fundoscope.ml.evaluate import Scores, parcel_prediction, score
from fundoscope.ml.features import FEATURE_COLUMNS

N_FOLDS = 5
SEED = 0
# Fixed for every setting; class weights make up for "other annual crops" being the
# largest class by far.
FOREST: dict[str, Any] = {
    "n_estimators": 500,
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
    "random_state": SEED,
}
# The settings compared by cross-validation.
GRID: tuple[dict[str, Any], ...] = (
    {"max_features": "sqrt", "min_samples_leaf": 1},
    {"max_features": "sqrt", "min_samples_leaf": 5},
    {"max_features": 0.3, "min_samples_leaf": 1},
    {"max_features": 0.3, "min_samples_leaf": 5},
)
TREE_CROPS = "tree crops"
# Where a class with fewer than 15 parcels goes (FR-06). A class missing here can't be merged.
BROADER_CLASS = {
    "avocado": TREE_CROPS,
    "citrus": TREE_CROPS,
    "mango": TREE_CROPS,
    "blueberry": TREE_CROPS,
    "rice": "other annual crops",
}
KEY = ["parcel_id", "campaign"]


class TrainingError(ValueError):
    """The model can't be trained on this data. The message is meant for the user."""


@dataclass(frozen=True)
class CrossValidation:
    """The cross-validated result of one setting.

    Attributes:
        settings: the Random Forest arguments.
        scores: per parcel-campaign, over the classes the model learns.
        predictions: one row per parcel-campaign (index `parcel_id`, `campaign`) with the
            labeled class `truth`, the predicted `crop_class`, and its `confidence`.
    """

    settings: dict[str, Any]
    scores: Scores
    predictions: pd.DataFrame


def class_merges(parcels_per_class: Mapping[str, int]) -> dict[str, str]:
    """The class each model class trains as: itself, or its broader class if it is too small.

    When a small class goes to a group like "tree crops", every class of that group goes
    with it (citrus + mango + avocado -> tree crops).

    Raises:
        TrainingError: a class has fewer than 15 parcels and nowhere to go, or a merged
            class still has fewer than 15.
    """
    present = {name: count for name, count in parcels_per_class.items() if count > 0}
    small = sorted(name for name, count in present.items() if count < MIN_PARCELS_PER_CLASS)
    stuck = [name for name in small if name not in BROADER_CLASS]
    if stuck:
        raise TrainingError(
            f"{', '.join(stuck)} has fewer than {MIN_PARCELS_PER_CLASS} parcels and no broader "
            "class to merge into: label more parcels of it."
        )
    groups = {BROADER_CLASS[name] for name in small}
    merges = {
        name: BROADER_CLASS[name] if BROADER_CLASS.get(name) in groups else name for name in present
    }
    totals: Counter[str] = Counter()
    for name, target in merges.items():
        totals[target] += present[name]
    still_small = sorted(name for name, count in totals.items() if count < MIN_PARCELS_PER_CLASS)
    if still_small:
        raise TrainingError(
            f"{', '.join(still_small)} still has fewer than {MIN_PARCELS_PER_CLASS} parcels "
            "after merging: label more parcels of it."
        )
    return merges


def training_table(pixels: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Join the training pixels with their labels.

    Args:
        pixels: `parcel_id`, `campaign`, and the 150 `FEATURE_COLUMNS`, one row per pixel.
        labels: `parcel_id` and `crop` (a value of `CROP_CLASSES`), one row per parcel.

    Returns:
        `parcel_id`, `campaign`, `crop_class` (the label's model class), `target` (the class
        the model learns, after merging small classes), and the features.

    Raises:
        TrainingError: a pixel's parcel has no label, or a small class can't be merged.
    """
    crops = labels.set_index("parcel_id")["crop"]
    crop_class = pixels["parcel_id"].map(crops).map(CROP_CLASSES)
    if crop_class.isna().any():
        unknown = sorted(set(pixels.loc[crop_class.isna(), "parcel_id"]))
        raise TrainingError(f"Pixels without a usable label: {', '.join(unknown)}.")
    per_class = crop_class.groupby(pixels["parcel_id"]).first().value_counts()
    merges = class_merges(per_class.to_dict())
    table = pixels[[*KEY, *FEATURE_COLUMNS]].reset_index(drop=True)
    table.insert(2, "crop_class", crop_class.to_numpy())
    table.insert(3, "target", crop_class.map(merges).to_numpy())
    return table


def grouped_folds(table: pd.DataFrame, folds: int = N_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Row positions of the training and test pixels of each fold, grouped by parcel.

    Folds keep about the same share of every class (stratified by `target`).
    """
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    return list(splitter.split(np.zeros(len(table)), table["target"], table["parcel_id"]))


def cross_validate(
    table: pd.DataFrame, settings: Mapping[str, Any], folds: int = N_FOLDS
) -> CrossValidation:
    """Score one setting: each fold is predicted by a forest trained on the other folds."""
    classes = sorted(table["target"].unique())
    features = table[list(FEATURE_COLUMNS)].to_numpy()
    target = table["target"].to_numpy()
    probabilities = np.zeros((len(table), len(classes)))
    for train, test in grouped_folds(table, folds):
        model = RandomForestClassifier(**settings).fit(features[train], target[train])
        columns = [classes.index(name) for name in model.classes_]
        probabilities[np.ix_(test, columns)] = model.predict_proba(features[test])
    pixels = pd.DataFrame(
        probabilities, columns=classes, index=pd.MultiIndex.from_frame(table[KEY])
    )
    predictions = parcel_prediction(pixels)
    predictions.insert(0, "truth", table.groupby(KEY)["target"].first())
    scores = score(predictions["truth"], predictions["crop_class"], classes)
    return CrossValidation(dict(settings), scores, predictions)


def best(results: Sequence[CrossValidation]) -> CrossValidation:
    """The setting with the highest macro-F1; the first one on a tie."""
    return max(results, key=lambda result: result.scores.macro_f1)


def fit(table: pd.DataFrame, settings: Mapping[str, Any]) -> RandomForestClassifier:
    """Train the final forest on every training pixel."""
    model = RandomForestClassifier(**settings)
    return model.fit(table[list(FEATURE_COLUMNS)].to_numpy(), table["target"].to_numpy())
