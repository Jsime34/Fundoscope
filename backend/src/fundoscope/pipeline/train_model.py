"""Train the crop model and save it with its model card and report (FR-06).

Before training, the labels must pass `check_labels`, every training pixel must be saved,
and every label that doesn't fit its NDVI curve must be reviewed. Then the settings of
`fundoscope.ml.GRID` are compared by grouped 5-fold cross-validation, the best one is refit
on all training parcels, and three files are written to the output folder (git-ignored):
`rf_v1.joblib` (the model), `model_card.json`, and `rf_cv.md` (the cross-validation report).
The field-verified parcels are not used: they are scored once, with the final model.

The same setting then trains the season-to-date models of the valley count (FR-14): one for
each campaign length from August alone to August-June, each cross-validated with the same
folds and saved as `season/rf_v1_to_<month>.joblib`. Their scores and confusion matrices (for
the 95 % ranges of the hectares) go into the model card and the report.
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
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
    MONTHS,
    N_FOLDS,
    TARGET_MACRO_F1,
    CropModel,
    CrossValidation,
    Scores,
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
SEASON_DIR = "season"
# Campaign lengths with a season-to-date model: August alone to August-June. All 12 months
# is the full model.
SEASON_MONTHS = tuple(range(1, len(MONTHS)))


def season_model_file(months: int) -> str:
    """The file name of the season-to-date model for the first `months` months.

    Raises:
        ValueError: `months` isn't 1 to 11.
    """
    if months not in SEASON_MONTHS:
        raise ValueError(f"Season-to-date models cover 1 to {len(MONTHS) - 1} months.")
    return f"{VERSION}_to_{MONTHS[months - 1]}.joblib"


@dataclass(frozen=True)
class TrainedModel:
    """What `train_crop_model` did.

    Attributes:
        chosen: the cross-validation of the setting that was refit and saved.
        compared: every setting tried, in `GRID` order.
        card: the model card, as saved in `model_card.json`.
        folder: where the model, card, and report were written.
        season: the cross-validation of each season-to-date model, by number of months.
    """

    chosen: CrossValidation
    compared: list[CrossValidation]
    card: dict[str, Any]
    folder: Path
    season: dict[int, CrossValidation] = field(default_factory=dict)


def train_crop_model(
    labels: gpd.GeoDataFrame,
    features_dir: Path,
    out_dir: Path,
    *,
    grid: tuple[dict[str, Any], ...] = GRID,
    forest: dict[str, Any] = FOREST,
    folds: int = N_FOLDS,
    season_months: tuple[int, ...] = SEASON_MONTHS,
    on_result: Callable[[CrossValidation], None] | None = None,
    on_season: Callable[[int, CrossValidation], None] | None = None,
) -> TrainedModel:
    """Compare the settings, refit the best one, and save model, card, and report.

    Then train a season-to-date model with the chosen setting for each of `season_months`.
    `on_result` and `on_season` are called as each cross-validation finishes.

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
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fit(table, chosen.settings), out_dir / MODEL_FILE, compress=3)

    season = {}
    if season_months:
        (out_dir / SEASON_DIR).mkdir(exist_ok=True)
    for months in season_months:
        partial = training_table(pixels, labels, months)
        season[months] = cross_validate(partial, chosen.settings, folds)
        path = out_dir / SEASON_DIR / season_model_file(months)
        joblib.dump(fit(partial, chosen.settings), path, compress=3)
        if on_season:
            on_season(months, season[months])

    card = model_card(table, labels, chosen, compared, grid, folds, len(curves.mismatches))
    card["season_to_date"] = season_card(season)
    (out_dir / CARD_FILE).write_text(json.dumps(card, indent=2), encoding="utf-8")
    (out_dir / REPORT_FILE).write_text(format_cv_report(card), encoding="utf-8")
    return TrainedModel(chosen, compared, card, out_dir, season)


def load_crop_model(folder: Path, months: int = len(MONTHS)) -> CropModel | None:
    """A crop model saved in `folder` by `train_crop_model`; None if there is none yet.

    Args:
        folder: the models folder.
        months: 12 for the full model; 1 to 11 for the season-to-date model of the first
            `months` months of a campaign.
    """
    if months == len(MONTHS):
        path = folder / MODEL_FILE
    else:
        path = folder / SEASON_DIR / season_model_file(months)
    return joblib.load(path) if path.exists() else None


def load_model_card(folder: Path) -> dict[str, Any] | None:
    """The model card saved in `folder` by `train_crop_model`; None if there is none yet."""
    path = folder / CARD_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def model_confusion(card: dict[str, Any], months: int = len(MONTHS)) -> pd.DataFrame:
    """The cross-validated confusion matrix of the model for `months` months, from its card.

    Rows are the labeled classes and columns the predicted ones.

    Raises:
        KeyError: the card has no season-to-date model for `months` months.
    """
    if months == len(MONTHS):
        entry = card["cross_validation"]
    else:
        found = [item for item in card.get("season_to_date", []) if item["months"] == months]
        if not found:
            raise KeyError(f"The model card has no season-to-date model for {months} months.")
        entry = found[0]
    names = entry["confusion"]["labeled_rows_predicted_columns"]
    return pd.DataFrame(entry["confusion"]["matrix"], index=names, columns=names)


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
            **_scores(scores),
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


def season_card(season: dict[int, CrossValidation]) -> list[dict[str, Any]]:
    """The model card entries of the season-to-date models, shortest first."""
    return [
        {
            "months": months,
            "through": MONTHS[months - 1],
            "file": f"{SEASON_DIR}/{season_model_file(months)}",
            **_scores(result.scores),
        }
        for months, result in sorted(season.items())
    ]


def _scores(scores: Scores) -> dict[str, Any]:
    return {
        "macro_f1": round(scores.macro_f1, 4),
        "per_class_f1": {name: round(value, 4) for name, value in scores.per_class_f1.items()},
        "confusion": {
            "labeled_rows_predicted_columns": list(scores.confusion.columns),
            "matrix": scores.confusion.to_numpy().tolist(),
        },
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
    lines += _season_report(card)
    return "\n".join(lines) + "\n"


def _season_report(card: dict[str, Any]) -> list[str]:
    season = card.get("season_to_date", [])
    if not season:
        return []
    classes = card["classes"]
    lines = [
        "",
        "## Season-to-date models (FR-14)",
        "",
        "A campaign still in progress is classified by a model trained on its first months "
        "only, August to the latest month, with the same setting and folds. Each is scored "
        "against the crop of the whole campaign.",
        "",
        "| Months | August to | Macro-F1 | " + " | ".join(classes) + " |",
        "|---:|---|---:|" + "---:|" * len(classes),
    ]
    full = {
        "months": len(MONTHS),
        "through": MONTHS[-1],
        "macro_f1": card["cross_validation"]["macro_f1"],
        "per_class_f1": card["cross_validation"]["per_class_f1"],
    }
    for entry in [*season, full]:
        per_class = " | ".join(f"{entry['per_class_f1'][name]:.2f}" for name in classes)
        lines.append(
            f"| {entry['months']} | {entry['through'].capitalize()} | "
            f"{entry['macro_f1']:.2f} | {per_class} |"
        )
    lines += [
        "",
        "The last row is the full model. A month in progress is classified with the model of "
        "its length, using the clear images so far.",
    ]
    return lines
