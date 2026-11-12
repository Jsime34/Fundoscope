import numpy as np
import pandas as pd
import pytest

from fundoscope.ml import (
    FEATURE_COLUMNS,
    CrossValidation,
    Scores,
    TrainingError,
    best,
    class_merges,
    cross_validate,
    fit,
    grouped_folds,
    training_table,
)

SMALL_FOREST = {"n_estimators": 10, "random_state": 0, "n_jobs": 1}
# crop of the QGIS drop-down -> feature level that sets it apart
LEVELS = {"grape": 0.0, "onion": 1.0, "fallow": 2.0}


def pixels_and_labels(parcels_per_crop=6, campaigns=("2024/25",), pixels=4, crops=LEVELS):
    """Pixels whose features depend only on the crop, plus their labels."""
    rng = np.random.default_rng(0)
    rows, labels = [], []
    number = 0
    for crop, level in crops.items():
        for _ in range(parcels_per_crop):
            number += 1
            parcel_id = f"ARE-{number:04d}"
            labels.append({"parcel_id": parcel_id, "crop": crop})
            for campaign in campaigns:
                for _ in range(pixels):
                    values = level + rng.normal(0, 0.05, len(FEATURE_COLUMNS))
                    rows.append(
                        {"parcel_id": parcel_id, "campaign": campaign}
                        | dict(zip(FEATURE_COLUMNS, values, strict=True))
                    )
    return pd.DataFrame(rows), pd.DataFrame(labels)


def test_the_table_has_the_label_class_and_the_trained_class():
    pixels, labels = pixels_and_labels(parcels_per_crop=15)

    table = training_table(pixels, labels)

    assert list(table.columns[:4]) == ["parcel_id", "campaign", "crop_class", "target"]
    assert list(table.columns[4:]) == list(FEATURE_COLUMNS)
    assert table.groupby("crop_class").parcel_id.nunique().to_dict() == {
        "fallow/bare": 15, "grape": 15, "other annual crops": 15,
    }  # fmt: skip
    assert (table["crop_class"] == table["target"]).all()


def test_pixels_without_a_label_are_refused():
    pixels, labels = pixels_and_labels(parcels_per_crop=15)

    with pytest.raises(TrainingError, match="ARE-0001"):
        training_table(pixels, labels[labels.parcel_id != "ARE-0001"])


def test_small_classes_are_merged_into_their_broader_class():
    pixels, labels = pixels_and_labels(parcels_per_crop=15, crops={"onion": 0, "rice": 1})
    keep = labels.iloc[:20]  # 15 onion parcels and 5 rice parcels

    table = training_table(pixels[pixels.parcel_id.isin(keep.parcel_id)], keep)

    assert table.groupby("crop_class").parcel_id.nunique().to_dict() == {
        "other annual crops": 15, "rice": 5,
    }  # fmt: skip
    assert set(table["target"]) == {"other annual crops"}


TREES = "tree crops"
ANNUAL = "other annual crops"


@pytest.mark.parametrize(
    ("counts", "merges"),
    [
        ({"grape": 20, "avocado": 15}, {"grape": "grape", "avocado": "avocado"}),
        (  # a small tree crop takes the whole group with it
            {"citrus": 5, "mango": 4, "avocado": 20, "grape": 30},
            {"citrus": TREES, "mango": TREES, "avocado": TREES, "grape": "grape"},
        ),
        ({"rice": 3, ANNUAL: 50}, {"rice": ANNUAL, ANNUAL: ANNUAL}),
        ({"grape": 20, "mango": 0}, {"grape": "grape"}),  # absent classes are ignored
    ],
)
def test_class_merges(counts, merges):
    assert class_merges(counts) == merges


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        ({"grape": 10, "avocado": 20}, "grape has fewer than 15 parcels and no broader class"),
        ({"citrus": 5, "grape": 20}, "tree crops still has fewer than 15"),
    ],
)
def test_classes_that_stay_too_small_are_refused(counts, message):
    with pytest.raises(TrainingError, match=message):
        class_merges(counts)


def test_no_parcel_is_in_both_the_training_and_the_test_folds():
    pixels, labels = pixels_and_labels(parcels_per_crop=15, campaigns=("2023/24", "2024/25"))
    table = training_table(pixels, labels)

    folds = grouped_folds(table)

    assert len(folds) == 5
    for train, test in folds:
        assert not set(table.parcel_id.iloc[train]) & set(table.parcel_id.iloc[test])
    tested = np.sort(np.concatenate([test for _, test in folds]))
    assert (tested == np.arange(len(table))).all()  # every pixel is tested exactly once


def test_cross_validation_predicts_each_parcel_campaign():
    pixels, labels = pixels_and_labels(parcels_per_crop=15, campaigns=("2023/24", "2024/25"))
    table = training_table(pixels, labels)

    result = cross_validate(table, SMALL_FOREST)

    assert result.settings == SMALL_FOREST
    assert len(result.predictions) == 90  # 45 parcels x 2 campaigns
    assert result.predictions.index.names == ["parcel_id", "campaign"]
    assert (result.predictions["truth"] == result.predictions["crop_class"]).all()
    assert result.scores.macro_f1 == 1.0
    assert result.predictions["confidence"].between(0, 1).all()


def test_a_class_missing_from_a_training_fold_gets_no_probability():
    pixels, labels = pixels_and_labels(parcels_per_crop=15, crops={"grape": 0, "fallow": 2})
    table = training_table(pixels, labels)
    # One alfalfa parcel that looks like grape: when it is tested, no training pixel is
    # alfalfa, so the forest can only answer grape or fallow.
    lone = table[table.parcel_id == "ARE-0001"].assign(
        parcel_id="ARE-0999", crop_class="alfalfa", target="alfalfa"
    )
    table = pd.concat([table, lone], ignore_index=True)

    result = cross_validate(table, SMALL_FOREST, folds=3)

    alfalfa = result.predictions.loc[("ARE-0999", "2024/25")]
    assert (alfalfa["truth"], alfalfa["crop_class"]) == ("alfalfa", "grape")
    assert result.scores.per_class_f1["alfalfa"] == 0.0


def outcome(macro_f1: float) -> CrossValidation:
    scores = Scores(macro_f1, {}, pd.DataFrame(), 0)
    return CrossValidation({"macro": macro_f1}, scores, pd.DataFrame())


def test_the_best_setting_wins_and_the_first_wins_a_tie():
    first, second, third = outcome(0.8), outcome(0.9), outcome(0.9)
    assert best([first, second, third]) is second


def test_the_final_forest_learns_every_class():
    pixels, labels = pixels_and_labels(parcels_per_crop=15)
    table = training_table(pixels, labels)

    model = fit(table, SMALL_FOREST)

    assert list(model.classes_) == ["fallow/bare", "grape", "other annual crops"]
    assert (model.predict(table[list(FEATURE_COLUMNS)].to_numpy()) == table["target"]).all()
