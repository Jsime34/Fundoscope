import copy
import json

import joblib
import pytest

from fundoscope.ml import FEATURE_COLUMNS, TrainingError, feature_columns
from fundoscope.pipeline import (
    CARD_FILE,
    MODEL_FILE,
    REPORT_FILE,
    SEASON_DIR,
    SEASON_MONTHS,
    extraction_tasks,
    format_cv_report,
    load_crop_model,
    load_training_pixels,
    season_model_file,
    train_crop_model,
)

SMALL_FOREST = {
    "n_estimators": 10,
    "class_weight": "balanced_subsample",
    "random_state": 0,
    "n_jobs": 1,
}
GRID = ({"min_samples_leaf": 1}, {"min_samples_leaf": 5})


def train(labels, features, out, season_months=(), **options):
    return train_crop_model(
        labels,
        features,
        out,
        grid=GRID,
        forest=SMALL_FOREST,
        season_months=season_months,
        **options,
    )


@pytest.fixture(scope="module")
def trained(training_inputs, tmp_path_factory):
    labels, features = training_inputs
    seen = []
    result = train(
        labels,
        features,
        tmp_path_factory.mktemp("models"),
        season_months=(1, 4),
        on_result=seen.append,
        on_season=lambda months, cv: seen.append((months, cv)),
    )
    return result, seen


def test_compares_every_setting_and_saves_the_best(trained):
    result, seen = trained

    assert seen[:2] == result.compared
    assert [r.settings["min_samples_leaf"] for r in result.compared] == [1, 5]
    assert result.chosen is result.compared[0]  # both perfect: the first wins
    assert result.chosen.scores.macro_f1 == 1.0
    assert {path.name for path in result.folder.iterdir()} == {
        MODEL_FILE, CARD_FILE, REPORT_FILE, SEASON_DIR,
    }  # fmt: skip


def test_season_to_date_models_use_the_chosen_setting(trained):
    result, seen = trained

    assert seen[2:] == [(1, result.season[1]), (4, result.season[4])]
    assert {path.name for path in (result.folder / SEASON_DIR).iterdir()} == {
        "rf_v1_to_aug.joblib", "rf_v1_to_nov.joblib",
    }  # fmt: skip
    for months, cv in result.season.items():
        assert cv.settings == result.chosen.settings
        assert cv.scores.macro_f1 == 1.0  # grape is green and fallow bare from August on
        assert load_crop_model(result.folder, months).n_features_in_ == len(feature_columns(months))


def test_season_to_date_models_are_in_the_card(trained):
    result, _ = trained
    [aug, nov] = result.card["season_to_date"]

    assert (aug["months"], aug["through"], aug["file"]) == (1, "aug", "season/rf_v1_to_aug.joblib")
    assert (nov["months"], nov["through"]) == (4, "nov")
    assert nov["macro_f1"] == 1.0
    assert nov["per_class_f1"] == {"fallow/bare": 1.0, "grape": 1.0}
    assert nov["confusion"]["matrix"] == [[15, 0], [0, 134]]


def test_season_to_date_model_files():
    assert SEASON_MONTHS == tuple(range(1, 12))
    assert season_model_file(1) == "rf_v1_to_aug.joblib"
    assert season_model_file(11) == "rf_v1_to_jun.joblib"
    with pytest.raises(ValueError, match="1 to 11 months"):
        season_model_file(12)


def test_the_saved_model_predicts_the_training_classes(trained, training_inputs):
    result, _ = trained
    labels, features = training_inputs
    pixels = load_training_pixels(extraction_tasks(labels), features)

    model = joblib.load(result.folder / MODEL_FILE)

    assert list(model.classes_) == ["fallow/bare", "grape"]
    predicted = model.predict(pixels[list(FEATURE_COLUMNS)].to_numpy())
    assert set(predicted[pixels.parcel_id == "ARE-0136"]) == {"fallow/bare"}


def test_the_model_card_describes_the_data_and_the_scores(trained):
    result, _ = trained
    card = json.loads((result.folder / CARD_FILE).read_text(encoding="utf-8"))

    assert card == result.card
    assert card["version"] == "rf_v1"
    assert card["classes"] == ["fallow/bare", "grape"]
    assert card["merged_classes"] == {}
    assert len(card["features"]) == 150
    assert card["settings"] == SMALL_FOREST | {"min_samples_leaf": 1}
    data = card["data"]
    assert (data["parcels"], data["parcel_campaigns"]) == (149, 149)  # one is too narrow
    assert data["parcels_per_class"] == {"grape": 134, "fallow/bare": 15}
    assert data["parcels_per_source"] == {"photo": 149}
    assert data["parcels_per_region"] == {"ICA": 50, "LLI": 50, "ARE": 49}
    assert len(data["sha256"]) == 64
    assert card["cross_validation"]["confusion"]["matrix"] == [[15, 0], [0, 134]]
    assert [item["macro_f1"] for item in card["cross_validation"]["compared"]] == [1.0, 1.0]


def test_the_report_is_written_from_the_card(trained):
    result, _ = trained
    report = (result.folder / REPORT_FILE).read_text(encoding="utf-8")

    assert report == format_cv_report(result.card)
    assert "**Macro-F1: 1.00** (meets the target" in report
    assert "| grape | 134 | 134 | 1.00 |" in report
    assert "| **fallow/bare** | 15 | 0 |" in report
    assert "| min_samples_leaf=1 (chosen) | 1.00 |" in report
    assert "| min_samples_leaf=5 | 1.00 |" in report
    assert "satellite images" not in report
    assert "0 labels that didn't fit their NDVI curve" in report
    assert "## Season-to-date models (FR-14)" in report
    assert "| Months | August to | Macro-F1 | fallow/bare | grape |" in report
    assert "| 1 | Aug | 1.00 | 1.00 | 1.00 |" in report
    assert "| 4 | Nov | 1.00 | 1.00 | 1.00 |" in report
    assert "| 12 | Jul | 1.00 | 1.00 | 1.00 |" in report  # the full model


def test_the_report_has_no_season_section_without_season_models(trained):
    card = copy.deepcopy(trained[0].card)
    card["season_to_date"] = []
    assert "Season-to-date" not in format_cv_report(card)


@pytest.mark.parametrize(
    ("macro_f1", "verdict"),
    [(0.70, "meets the minimum"), (0.50, "below the minimum")],
)
def test_the_report_states_where_the_score_stands(trained, macro_f1, verdict):
    card = copy.deepcopy(trained[0].card)
    card["cross_validation"]["macro_f1"] = macro_f1

    assert f"**Macro-F1: {macro_f1:.2f}** ({verdict};" in format_cv_report(card)


def test_the_report_mentions_satellite_labels_and_merged_classes(trained):
    card = copy.deepcopy(trained[0].card)
    card["data"]["parcels_per_source"] = {"photo": 134, "satellite": 15}
    card["merged_classes"] = {"citrus": "tree crops"}

    report = format_cv_report(card)

    assert "- 15 fallow/bare parcels come from satellite images" in report
    assert "Classes merged for having fewer than 15 parcels: citrus -> tree crops." in report


def test_labels_that_fail_the_check_are_refused(training_inputs, tmp_path):
    labels, features = training_inputs
    without_test_set = labels[labels.source != "field"]

    with pytest.raises(TrainingError, match="labels aren't ready"):
        train(without_test_set, features, tmp_path)


def test_every_training_pixel_must_be_downloaded(training_inputs, tmp_path):
    labels, _ = training_inputs

    with pytest.raises(TrainingError, match="149 parcel-campaigns have no training pixels yet"):
        train(labels, tmp_path / "empty", tmp_path / "models")


def test_unreviewed_ndvi_mismatches_are_refused(training_inputs, tmp_path):
    labels, features = training_inputs
    relabeled = labels.copy()
    relabeled.loc[relabeled.parcel_id == "ICA-0002", "crop"] = "fallow"  # it is green

    with pytest.raises(TrainingError, match="1 labels don't fit their NDVI curve"):
        train(relabeled, features, tmp_path)

    relabeled.loc[relabeled.parcel_id == "ICA-0002", "notes"] = "NDVI ok: checked"
    assert train(relabeled, features, tmp_path).card["data"]["reviewed_ndvi_mismatches"] == 1


def test_the_saved_model_can_be_loaded_back(trained, tmp_path):
    result, _ = trained

    assert list(load_crop_model(result.folder).classes_) == ["fallow/bare", "grape"]
    assert load_crop_model(tmp_path) is None  # not trained yet
    assert load_crop_model(result.folder, 2) is None  # no model for August-September
    with pytest.raises(ValueError, match="1 to 11 months"):
        load_crop_model(result.folder, 13)


def test_the_confusion_matrices_come_from_the_saved_card(trained, tmp_path):
    from fundoscope.pipeline.train_model import load_model_card, model_confusion

    result, _ = trained
    card = load_model_card(result.folder)

    full = model_confusion(card)
    assert list(full.index) == list(full.columns) == ["fallow/bare", "grape"]
    assert full.to_numpy().sum() == card["data"]["parcel_campaigns"]
    to_november = model_confusion(card, 4)
    assert to_november.to_numpy().sum() == full.to_numpy().sum()
    with pytest.raises(KeyError, match="no season-to-date model for 2 months"):
        model_confusion(card, 2)
    assert load_model_card(tmp_path) is None
