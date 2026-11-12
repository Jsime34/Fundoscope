"""`fundoscope model train`: train the crop model on the labeled parcels (FR-06)."""

from pathlib import Path
from typing import Any

from fundoscope.cli.labels import default_features_dir, default_labels_file
from fundoscope.config import get_settings
from fundoscope.ingest import LabelError, read_labels
from fundoscope.ml import (
    FOREST,
    GRID,
    MIN_MACRO_F1,
    N_FOLDS,
    TARGET_MACRO_F1,
    CrossValidation,
    TrainingError,
)
from fundoscope.pipeline import TrainedModel, format_cv_report, train_crop_model


def run_model_train(
    file: Path | None,
    out: Path | None = None,
    report: Path | None = None,
    *,
    features: Path | None = None,
    grid: tuple[dict[str, Any], ...] = GRID,
    forest: dict[str, Any] = FOREST,
    folds: int = N_FOLDS,
) -> int:
    """Train, print the scores, and save the model. Returns the exit code.

    `report`, if given, receives a copy of the cross-validation report (for docs/eval/).
    """
    out = out or default_models_dir()
    print(
        f"Comparing {len(grid)} settings by {folds}-fold cross-validation grouped by parcel "
        "(up to a minute each)..."
    )
    try:
        labels = read_labels(file or default_labels_file())
        trained = train_crop_model(
            labels,
            features or default_features_dir(),
            out,
            grid=grid,
            forest=forest,
            folds=folds,
            on_result=lambda result: print(f"  {_setting(result, forest)}"),
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
    return "\n".join(
        [
            f"\nChosen: {_setting(trained.chosen, forest)} ({verdict}: at least "
            f"{MIN_MACRO_F1:.2f}, target {TARGET_MACRO_F1:.2f}).",
            f"F1 per class: {per_class}.",
            f"Trained on {data['parcels']} parcels, {data['parcel_campaigns']} parcel-campaigns, "
            f"{data['pixels']:,} pixels. The field-verified parcels were not used.",
            f"Saved the model, model card, and report in {trained.folder}.",
        ]
    )


def _setting(result: CrossValidation, forest: dict[str, Any]) -> str:
    changed = {key: value for key, value in result.settings.items() if forest.get(key) != value}
    described = ", ".join(f"{key}={value}" for key, value in changed.items()) or "defaults"
    return f"{described}: macro-F1 {result.scores.macro_f1:.2f}"
