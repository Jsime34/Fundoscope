from pathlib import Path

import pandas as pd

import fundoscope.cli
from fundoscope.cli import main
from fundoscope.cli.model import default_models_dir, format_training, run_model_train
from fundoscope.ml import CrossValidation, Scores
from fundoscope.pipeline import TrainedModel

SMALL_FOREST = {
    "n_estimators": 10,
    "class_weight": "balanced_subsample",
    "random_state": 0,
    "n_jobs": 1,
}


def run(tmp_path, training_inputs, **options):
    labels, features = training_inputs
    path = tmp_path / "labels.gpkg"
    labels.to_file(path, layer="labels", driver="GPKG")
    return run_model_train(
        path,
        tmp_path / "models",
        features=features,
        grid=({"min_samples_leaf": 1},),
        forest=SMALL_FOREST,
        **options,
    )


def test_training_prints_the_scores_and_saves_the_report(tmp_path, training_inputs, capsys):
    report = tmp_path / "docs" / "rf_cv.md"

    assert run(tmp_path, training_inputs, report=report) == 0

    printed = capsys.readouterr().out
    assert "Comparing 1 settings by 5-fold cross-validation grouped by parcel" in printed
    assert "  min_samples_leaf=1: macro-F1 1.00" in printed
    assert "Chosen: min_samples_leaf=1: macro-F1 1.00 (meets the FR-06 target" in printed
    assert "F1 per class: fallow/bare 1.00, grape 1.00." in printed
    assert "Trained on 149 parcels, 149 parcel-campaigns" in printed
    assert (tmp_path / "models" / "rf_v1.joblib").exists()
    assert report.read_text(encoding="utf-8").startswith("# Random Forest")
    assert f"Report also saved to {report}." in printed


def test_problems_are_printed(tmp_path, training_inputs, capsys):
    labels, _ = training_inputs
    path = tmp_path / "labels.gpkg"
    labels.to_file(path, layer="labels", driver="GPKG")

    assert run_model_train(path, tmp_path / "models", features=tmp_path / "nothing") == 1
    assert "have no training pixels yet" in capsys.readouterr().out

    assert run_model_train(tmp_path / "missing.gpkg") == 1
    assert "Can't read layer" in capsys.readouterr().out


def trained_with(macro_f1: float, settings: dict) -> TrainedModel:
    scores = Scores(macro_f1, {"grape": macro_f1}, pd.DataFrame(), 10)
    chosen = CrossValidation(settings, scores, pd.DataFrame())
    data = {"parcels": 10, "parcel_campaigns": 10, "pixels": 500}
    return TrainedModel(chosen, [chosen], {"data": data}, Path("models"))


def test_the_summary_says_where_the_score_stands():
    assert "meets the FR-06 minimum, below the target" in format_training(
        trained_with(0.7, {"n_estimators": 500})
    )
    below = format_training(trained_with(0.5, SMALL_FOREST), SMALL_FOREST)
    assert "Chosen: defaults: macro-F1 0.50 (below the FR-06 minimum" in below


def test_default_models_dir_is_at_the_repo_root():
    assert default_models_dir().name == "models"


def test_main_runs_the_training(monkeypatch, tmp_path):
    received = []
    monkeypatch.setattr(fundoscope.cli, "run_model_train", lambda *args: received.append(args) or 0)

    assert main(["model", "train"]) == 0
    assert main(["model", "train", "x.gpkg", "--out", "m", "--report", "r.md"]) == 0

    assert received == [(None, None, None), (Path("x.gpkg"), Path("m"), Path("r.md"))]
