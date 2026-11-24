"""`fundoscope model train`: train the crop model on the labeled parcels (FR-06).

It also trains the season-to-date models used by the valley count (FR-14).
"""

from pathlib import Path
from typing import Any

from fundoscope.cli.labels import default_features_dir, default_labels_file
from fundoscope.config import get_settings
from fundoscope.ingest import LabelError, read_labels
from fundoscope.ml import (
    FOREST,
    GRID,
    MIN_MACRO_F1,
    MONTHS,
    N_FOLDS,
    TARGET_MACRO_F1,
    CrossValidation,
    TrainingError,
)
from fundoscope.pipeline import (
    SEASON_MONTHS,
    TrainedModel,
    format_cv_report,
    train_crop_model,
)


def run_model_train(
    file: Path | None,
    out: Path | None = None,
    report: Path | None = None,
    *,
    features: Path | None = None,
    grid: tuple[dict[str, Any], ...] = GRID,
    forest: dict[str, Any] = FOREST,
    folds: int = N_FOLDS,
    season_months: tuple[int, ...] = SEASON_MONTHS,
) -> int:
    """Train, print the scores, and save the models. Returns the exit code.

    `report`, if given, receives a copy of the cross-validation report (for docs/eval/).
    """
    out = out or default_models_dir()
    print(
        f"Comparing {len(grid)} settings by {folds}-fold cross-validation grouped by parcel "
        "(up to a minute each)..."
    )

    def season_done(months: int, result: CrossValidation) -> None:
        if months == season_months[0]:
            print("Season-to-date models (the chosen setting on the first months only):")
        print(f"  {_months(months)}: macro-F1 {result.scores.macro_f1:.2f}")

    try:
        labels = read_labels(file or default_labels_file())
        trained = train_crop_model(
            labels,
            features or default_features_dir(),
            out,
            grid=grid,
            forest=forest,
            folds=folds,
            season_months=season_months,
            on_result=lambda result: print(f"  {_setting(result, forest)}"),
            on_season=season_done,
        )
    except (LabelError, TrainingError) as exc:
        print(str(exc))
        return 1
    print(format_training(trained, forest))
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(format_cv_report(trained.card), encoding="utf-8")
        print(f"Report also saved to {report}.")
    return 0


def default_models_dir() -> Path:
    """models/ at the repo root: git-ignored."""
    return get_settings().models_dir


def format_training(trained: TrainedModel, forest: dict[str, Any] = FOREST) -> str:
    scores = trained.chosen.scores
    if scores.macro_f1 >= TARGET_MACRO_F1:
        verdict = "meets the FR-06 target"
    elif scores.macro_f1 >= MIN_MACRO_F1:
        verdict = "meets the FR-06 minimum, below the target"
    else:
        verdict = "below the FR-06 minimum"
    per_class = ", ".join(
        f"{name} {value:.2f}"
        for name, value in sorted(scores.per_class_f1.items(), key=lambda item: -item[1])
    )
    data = trained.card["data"]
    lines = [
        f"\nChosen: {_setting(trained.chosen, forest)} ({verdict}: at least "
        f"{MIN_MACRO_F1:.2f}, target {TARGET_MACRO_F1:.2f}).",
        f"F1 per class: {per_class}.",
        f"Trained on {data['parcels']} parcels, {data['parcel_campaigns']} parcel-campaigns, "
        f"{data['pixels']:,} pixels. The field-verified parcels were not used.",
    ]
    if trained.season:
        scores = [result.scores.macro_f1 for result in trained.season.values()]
        lines.append(
            f"Season-to-date models: macro-F1 {min(scores):.2f} to {max(scores):.2f} "
            f"({len(scores)} models, {_months(min(trained.season))} to "
            f"{_months(max(trained.season))})."
        )
    lines.append(f"Saved the models, model card, and report in {trained.folder}.")
    return "\n".join(lines)


def _months(months: int) -> str:
    """'Aug' for 1 month, 'Aug-Nov' for 4."""
    first, last = MONTHS[0].capitalize(), MONTHS[months - 1].capitalize()
    return first if months == 1 else f"{first}-{last}"


def _setting(result: CrossValidation, forest: dict[str, Any]) -> str:
    changed = {key: value for key, value in result.settings.items() if forest.get(key) != value}
    described = ", ".join(f"{key}={value}" for key, value in changed.items()) or "defaults"
    return f"{described}: macro-F1 {result.scores.macro_f1:.2f}"
