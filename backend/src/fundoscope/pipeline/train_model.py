"""Train the crop model and save it with its model card and report (FR-06).

Before training, the labels must pass `check_labels`, every training pixel must be saved,
and every label that doesn't fit its NDVI curve must be reviewed. Then the settings of
`fundoscope.ml.GRID` are compared by grouped 5-fold cross-validation, the best one is refit
on all training parcels, and three files are written to the output folder (git-ignored):
`rf_v1.joblib` (the model), `model_card.json`, and `rf_cv.md` (the cross-validation report).
The field-verified parcels are not used: they are scored once, with the final model.
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import joblib
import pandas as pd
import sklearn

from fundoscope.ingest import check_labels
from fundoscope.ml import (
    FEATURE_COLUMNS,
    FOREST,
    GRID,
    MIN_MACRO_F1,
    N_FOLDS,
    TARGET_MACRO_F1,
    CrossValidation,
    TrainingError,
    best,
    check_curves,
    cross_validate,
    fit,
    training_table,
)
from fundoscope.pipeline.training_data import (
    extraction_tasks,
    load_training_pixels,
    pending,
    too_narrow,
)

VERSION = "rf_v1"
MODEL_FILE = f"{VERSION}.joblib"
CARD_FILE = "model_card.json"
REPORT_FILE = "rf_cv.md"


@dataclass(frozen=True)
class TrainedModel:
    """What `train_crop_model` did.

    Attributes:
        chosen: the cross-validation of the setting that was refit and saved.
        compared: every setting tried, in `GRID` order.
        card: the model card, as saved in `model_card.json`.
        folder: where the model, card, and report were written.
    """

    chosen: CrossValidation
    compared: list[CrossValidation]
    card: dict[str, Any]
    folder: Path


def train_crop_model(
    labels: gpd.GeoDataFrame,
    features_dir: Path,
    out_dir: Path,
    *,
    grid: tuple[dict[str, Any], ...] = GRID,
    forest: dict[str, Any] = FOREST,
    folds: int = N_FOLDS,
    on_result: Callable[[CrossValidation], None] | None = None,
) -> TrainedModel:
    """Compare the settings, refit the best one, and save model, card, and report.

    Raises:
        TrainingError: the labels or the training pixels aren't ready.
    """
    if not check_labels(labels).ok:
        raise TrainingError(
            "The labels aren't ready: run `fundoscope labels check` and fix what it lists."
        )
    tasks = extraction_tasks(labels)
    missing = [task for task in pending(tasks, features_dir) if not too_narrow(task)]
    if missing:
        raise TrainingError(
            f"{len(missing)} parcel-campaigns have no training pixels yet: "
            "run `fundoscope labels extract`."
        )
    pixels = load_training_pixels(tasks, features_dir)
    curves = check_curves(pixels, labels)
    if curves.unreviewed:
        raise TrainingError(
            f"{len(curves.unreviewed)} labels don't fit their NDVI curve and aren't reviewed: "
            "run `fundoscope labels extract` and review them."
        )
    table = training_table(pixels, labels)

    compared = []
    for setting in grid:
        result = cross_validate(table, forest | setting, folds)
        compared.append(result)
        if on_result:
            on_result(result)
    chosen = best(compared)
    model = fit(table, chosen.settings)

    card = model_card(table, labels, chosen, compared, grid, folds, len(curves.mismatches))
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / MODEL_FILE, compress=3)
    (out_dir / CARD_FILE).write_text(json.dumps(card, indent=2), encoding="utf-8")
    (out_dir / REPORT_FILE).write_text(format_cv_report(card), encoding="utf-8")
    return TrainedModel(chosen, compared, card, out_dir)


def model_card(
    table: pd.DataFrame,
    labels: pd.DataFrame,
    chosen: CrossValidation,
    compared: list[CrossValidation],
    grid: tuple[dict[str, Any], ...],
    folds: int,
    reviewed_mismatches: int,
) -> dict[str, Any]:
    """What the model is, what it was trained on, and how it scored."""
    parcels = table.drop_duplicates("parcel_id")
    used = labels[labels["parcel_id"].isin(parcels["parcel_id"])]
    samples = table.drop_duplicates(["parcel_id", "campaign"])
    scores = chosen.scores
    return {
        "model": "Random Forest (scikit-learn)",
        "version": VERSION,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "scikit_learn": sklearn.__version__,
        "classes": list(scores.per_class_f1),
        "merged_classes": {
            row.crop_class: row.target
            for row in parcels.itertuples()
            if row.crop_class != row.target
        },
        "features": list(FEATURE_COLUMNS),
        "settings": chosen.settings,
        "cross_validation": {
            "folds": folds,
            "grouped_by": "parcel_id",
            "unit": "parcel-campaign: class with the highest mean probability of its pixels",
            "macro_f1": round(scores.macro_f1, 4),
            "per_class_f1": {name: round(value, 4) for name, value in scores.per_class_f1.items()},
            "confusion": {
                "labeled_rows_predicted_columns": list(scores.confusion.columns),
                "matrix": scores.confusion.to_numpy().tolist(),
            },
            "compared": [
                {"setting": setting, "macro_f1": round(result.scores.macro_f1, 4)}
                for setting, result in zip(grid, compared, strict=True)
            ],
        },
        "data": {
            "pixels": len(table),
            "parcel_campaigns": len(samples),
            "parcels": len(parcels),
            "parcels_per_class": _counts(parcels["target"]),
            "parcel_campaigns_per_class": _counts(samples["target"]),
            "parcels_per_source": _counts(used["source"]),
            "parcels_per_region": _counts(used["region"]),
            "reviewed_ndvi_mismatches": reviewed_mismatches,
            "sha256": hashlib.sha256(
                pd.util.hash_pandas_object(table, index=False).to_numpy().tobytes()
            ).hexdigest(),
        },
        "field_verified_parcels": "not used: kept for one final test with the frozen model",
    }


def _counts(values: pd.Series) -> dict[str, int]:
    return {str(name): int(count) for name, count in values.value_counts().items()}


def format_cv_report(card: dict[str, Any]) -> str:
    """The cross-validation report in Markdown, written from the model card."""
    cv, data = card["cross_validation"], card["data"]
    macro = cv["macro_f1"]
    verdict = (
        "meets the target"
        if macro >= TARGET_MACRO_F1
        else ("meets the minimum" if macro >= MIN_MACRO_F1 else "below the minimum")
    )
    satellite = data["parcels_per_source"].get("satellite", 0)
    lines = [
        "# Random Forest: grouped cross-validation",
        "",
        f"Trained {card['trained_at'][:10]} on {data['parcels']} labeled parcels "
        f"({data['parcel_campaigns']} parcel-campaigns, {data['pixels']:,} pixels). "
        f"{cv['folds']}-fold cross-validation grouped by parcel: each fold is scored on "
        "parcels the model never saw. Each parcel-campaign gets the class with the highest "
        "mean probability over its pixels.",
        "",
        f"**Macro-F1: {macro:.2f}** ({verdict}; FR-06 requires at least {MIN_MACRO_F1:.2f}, "
        f"target {TARGET_MACRO_F1:.2f}).",
        "",
        "| Class | Parcels | Parcel-campaigns | F1 |",
        "|---|---:|---:|---:|",
    ]
    for name, value in sorted(cv["per_class_f1"].items(), key=lambda item: -item[1]):
        lines.append(
            f"| {name} | {data['parcels_per_class'][name]} | "
            f"{data['parcel_campaigns_per_class'][name]} | {value:.2f} |"
        )
    names = cv["confusion"]["labeled_rows_predicted_columns"]
    lines += [
        "",
        "Confusion matrix (rows: labeled class, columns: predicted class):",
        "",
        "| | " + " | ".join(names) + " |",
        "|---|" + "---:|" * len(names),
    ]
    for name, row in zip(names, cv["confusion"]["matrix"], strict=True):
        lines.append(f"| **{name}** | " + " | ".join(str(count) for count in row) + " |")
    lines += [
        "",
        f"Settings compared ({card['settings']['n_estimators']} trees, "
        f"class weights `{card['settings']['class_weight']}`):",
        "",
        "| Setting | Macro-F1 |",
        "|---|---:|",
    ]
    for item in cv["compared"]:
        setting = ", ".join(f"{key}={value}" for key, value in item["setting"].items())
        is_chosen = all(
            card["settings"].get(key) == value for key, value in item["setting"].items()
        )
        chosen = " (chosen)" if is_chosen else ""
        lines.append(f"| {setting}{chosen} | {item['macro_f1']:.2f} |")
    if card["merged_classes"]:
        merged = ", ".join(f"{a} -> {b}" for a, b in card["merged_classes"].items())
        lines += ["", f"Classes merged for having fewer than 15 parcels: {merged}."]
    lines += [
        "",
        "Notes:",
        "",
        f"- The score is the best of {len(cv['compared'])} settings, so it is slightly "
        "optimistic. The field-verified parcels are not used here: they are scored once, "
        "with the final model.",
        "- Neighboring fields can fall in different folds. Fields next to each other tend to "
        "look alike, so a new area may score lower.",
    ]
    if satellite:
        lines.append(
            f"- {satellite} fallow/bare parcels come from satellite images (bare all campaign) "
            "instead of Street View, because idle fields next to Street View roads are rare."
        )
    lines.append(
        f"- {data['reviewed_ndvi_mismatches']} labels that didn't fit their NDVI curve were "
        "reviewed by hand and kept."
    )
    return "\n".join(lines) + "\n"
