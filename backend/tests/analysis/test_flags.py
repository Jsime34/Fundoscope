import numpy as np
import pandas as pd
import pytest

from fundoscope.analysis import (
    ABANDONMENT,
    ANOMALY,
    CROP_CHANGE,
    LOW_DATA,
    PERENNIAL_ESTABLISHED,
    PERENNIAL_REMOVED,
    PERENNIAL_REPLACED,
    abandonment_flags,
    anomaly_flags,
    campaign_flags,
    crop_change_flags,
    low_data_flags,
)


def campaigns(**columns) -> pd.DataFrame:
    """A campaign table like `parcel_history` makes; unspecified columns are steady values."""
    n = len(next(iter(columns.values())))
    labels = [f"{2018 + i}/{(19 + i) % 100:02d}" for i in range(n)]
    table = {
        "complete": [True] * n,
        "active_area": [0.9] * n,
        "crop": [None] * n,
        "crop_confidence": [np.nan] * n,
        "integrated_ndvi": [150.0] * n,
        "n_obs": [60] * n,
    }
    return pd.DataFrame(table | columns, index=pd.Index(labels, name="campaign"))


# --- FR-07 possible abandonment -------------------------------------------------------------


def test_a_steady_parcel_is_not_flagged():
    assert abandonment_flags(campaigns(active_area=[0.91, 0.93, 0.92, 0.91, 0.93, 0.92])) == []


def test_a_gradual_decline_is_not_flagged():
    assert abandonment_flags(campaigns(active_area=[0.9, 0.85, 0.8, 0.75, 0.7, 0.65])) == []


def test_a_collapse_is_flagged_with_its_evidence():
    [flag] = abandonment_flags(
        campaigns(active_area=[0.9, 0.88, 0.92, 0.2], n_obs=[60, 61, 62, 58])
    )

    assert (flag.campaign, flag.kind) == ("2021/22", ABANDONMENT)
    assert flag.evidence == {
        "active_area": 0.2, "median_before": 0.9, "threshold": 0.45, "clear_dates": 58,
    }  # fmt: skip
    assert flag.message == (
        "Possible abandonment: only 20% of the parcel was farmed, less than half of its usual 90%."
    )


def test_abandonment_needs_two_earlier_campaigns():
    assert abandonment_flags(campaigns(active_area=[0.9, 0.2])) == []
    assert len(abandonment_flags(campaigns(active_area=[0.9, 0.9, 0.2]))) == 1


def test_a_parcel_that_stays_idle_keeps_its_flag():
    flags = abandonment_flags(campaigns(active_area=[0.9, 0.9, 0.2, 0.2]))
    assert [flag.campaign for flag in flags] == ["2020/21", "2021/22"]


def test_campaigns_in_progress_or_unknown_are_skipped():
    table = campaigns(
        active_area=[0.9, np.nan, 0.9, 0.1],
        complete=[True, True, True, False],  # the last campaign has only begun
    )
    assert abandonment_flags(table) == []


def test_a_parcel_never_farmed_is_not_abandoned():
    assert abandonment_flags(campaigns(active_area=[0.0, 0.0, 0.0])) == []


# --- FR-27 crop change ----------------------------------------------------------------------


def changes(crops, confidences):
    return crop_change_flags(campaigns(crop=crops, crop_confidence=confidences))


def test_a_sure_change_is_flagged_with_both_predictions():
    [flag] = changes(["onion", "alfalfa"], [0.8, 0.7])

    assert (flag.campaign, flag.kind) == ("2019/20", CROP_CHANGE)
    assert flag.evidence == {
        "previous_campaign": "2018/19", "previous_crop": "onion", "previous_confidence": 0.8,
        "crop": "alfalfa", "confidence": 0.7,
    }  # fmt: skip
    assert (
        flag.message == "Crop change: onion in 2018/19 (80% sure), alfalfa in 2019/20 (70% sure)."
    )


@pytest.mark.parametrize("confidences", [[0.59, 0.9], [0.9, 0.59]])
def test_a_change_with_an_unsure_prediction_is_not_flagged(confidences):
    assert changes(["onion", "alfalfa"], confidences) == []


def test_a_change_at_exactly_the_threshold_is_flagged():
    assert len(changes(["onion", "alfalfa"], [0.6, 0.6])) == 1


@pytest.mark.parametrize(
    ("crops", "kind"),
    [
        (["grape", "other annual crops"], PERENNIAL_REMOVED),
        (["fallow/bare", "avocado"], PERENNIAL_ESTABLISHED),
        (["asparagus", "avocado"], PERENNIAL_REPLACED),
        (["tree crops", "alfalfa"], PERENNIAL_REMOVED),  # a merged class of perennials
        (["alfalfa", "other annual crops"], CROP_CHANGE),  # alfalfa is not on the list
    ],
)
def test_perennials_are_named(crops, kind):
    [flag] = changes(crops, [0.9, 0.9])
    assert flag.kind == kind
    assert flag.message.startswith(kind.capitalize())


def test_the_same_crop_or_a_missing_prediction_is_not_a_change():
    assert changes(["grape", "grape", None, "onion"], [0.9, 0.9, np.nan, 0.9]) == []


# --- FR-09 anomalous campaign and low data --------------------------------------------------


def test_a_halved_campaign_is_flagged_with_the_normal_range():
    [flag] = anomaly_flags(campaigns(integrated_ndvi=[150, 160, 140, 155, 75]))

    assert (flag.campaign, flag.kind) == ("2022/23", ANOMALY)
    assert flag.evidence["integrated_ndvi"] == 75
    assert flag.evidence["median_others"] == 152.5
    assert flag.evidence["normal_low"] > 75
    assert flag.evidence["robust_z"] < -2.5
    assert "well below the parcel's usual 152" in flag.message


def test_ten_percent_variation_is_not_flagged():
    assert anomaly_flags(campaigns(integrated_ndvi=[150, 165, 135, 160, 140, 150])) == []


def test_an_unusually_strong_campaign_is_flagged_too():
    [flag] = anomaly_flags(campaigns(integrated_ndvi=[150, 160, 140, 155, 300]))
    assert "well above" in flag.message


def test_nearly_identical_campaigns_tolerate_small_differences():
    # the MAD of 150, 150, 150 is 0: a 4 % difference must not look extreme
    assert anomaly_flags(campaigns(integrated_ndvi=[150, 150, 150, 156])) == []


def test_fewer_than_four_complete_campaigns_are_not_tested():
    table = campaigns(integrated_ndvi=[150, 160, 140, 20], complete=[True, True, True, False])
    assert anomaly_flags(table) == []


def test_fewer_than_15_clear_dates_is_low_data():
    [flag] = low_data_flags(campaigns(n_obs=[60, 14, 15]))

    assert (flag.campaign, flag.kind, flag.evidence) == (
        "2019/20", LOW_DATA, {"clear_dates": 14, "minimum": 15},
    )  # fmt: skip
    assert flag.message.startswith("Low data: only 14 clear dates")


# --- all together ---------------------------------------------------------------------------


def test_all_flags_come_in_campaign_order():
    table = campaigns(
        active_area=[0.9, 0.9, 0.9, 0.1],
        crop=["grape", "grape", "grape", "fallow/bare"],
        crop_confidence=[0.9, 0.9, 0.9, 0.95],
        integrated_ndvi=[150, 155, 145, 20],
        n_obs=[60, 10, 60, 60],
    )

    flags = campaign_flags(table)

    assert [(flag.campaign, flag.kind) for flag in flags] == [
        ("2019/20", LOW_DATA),
        ("2021/22", ABANDONMENT),
        ("2021/22", PERENNIAL_REMOVED),
        ("2021/22", ANOMALY),
    ]


def test_an_empty_table_has_no_flags():
    empty = campaigns(complete=[]).astype(object)
    assert campaign_flags(empty) == []
