import pandas as pd
import pytest

from fundoscope.ml import parcel_prediction, score


def test_macro_f1_is_the_mean_of_the_class_scores():
    truth = ["grape", "grape", "avocado", "avocado"]
    predicted = ["grape", "avocado", "avocado", "avocado"]

    scores = score(truth, predicted, ["avocado", "grape"])

    assert scores.per_class_f1 == pytest.approx({"avocado": 0.8, "grape": 2 / 3})
    assert scores.macro_f1 == pytest.approx((0.8 + 2 / 3) / 2)
    assert scores.samples == 4


def test_the_confusion_matrix_has_labels_as_rows_and_predictions_as_columns():
    scores = score(
        ["grape", "grape", "avocado"], ["avocado", "grape", "avocado"], ["avocado", "grape"]
    )

    assert scores.confusion.loc["grape"].to_dict() == {"avocado": 1, "grape": 1}
    assert scores.confusion.loc["avocado"].to_dict() == {"avocado": 1, "grape": 0}


def test_a_class_never_predicted_scores_zero():
    scores = score(["alfalfa", "grape"], ["grape", "grape"], ["alfalfa", "grape"])

    assert scores.per_class_f1["alfalfa"] == 0.0
    assert scores.confusion.loc["alfalfa", "grape"] == 1


def test_a_parcel_takes_the_class_with_the_highest_mean_probability():
    index = pd.MultiIndex.from_tuples(
        [("ARE-0001", "2024/25")] * 3 + [("ARE-0002", "2024/25")], names=["parcel_id", "campaign"]
    )
    pixels = pd.DataFrame(
        {"avocado": [0.9, 0.2, 0.3, 0.1], "grape": [0.1, 0.8, 0.7, 0.9]}, index=index
    )

    result = parcel_prediction(pixels)

    assert result.loc[("ARE-0001", "2024/25"), "crop_class"] == "grape"  # mean 0.53 vs 0.47
    assert result.loc[("ARE-0001", "2024/25"), "confidence"] == pytest.approx(1.6 / 3)
    assert result.loc[("ARE-0002", "2024/25"), "confidence"] == pytest.approx(0.9)
