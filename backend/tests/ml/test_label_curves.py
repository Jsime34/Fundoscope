import numpy as np
import pandas as pd
import pytest

from fundoscope.ml import (
    FEATURE_COLUMNS,
    MONTHS,
    REVIEWED_NOTE,
    campaign_curves,
    check_curves,
    curve_problem,
)

GREEN = [0.7] * 12
BARE = [0.1] * 12


def curve(*, bare_months=(), low=0.1, high=0.7):
    """Green all campaign except the given months (0 = August)."""
    return [low if month in bare_months else high for month in range(12)]


def pixels(parcel_id, campaign, ndvi, count=3):
    """`count` pixels of one parcel and campaign, all with the same monthly NDVI."""
    row = dict.fromkeys(FEATURE_COLUMNS, 0.0)
    row |= {f"ndvi_{month}": value for month, value in zip(MONTHS, ndvi, strict=True)}
    return pd.DataFrame([{"parcel_id": parcel_id, "campaign": campaign, **row}] * count)


def labels(*rows):
    columns = ("parcel_id", "crop", "notes")
    return pd.DataFrame([dict(zip(columns, row, strict=True)) for row in rows])


def bare_for(months, first, last):
    return f"near bare soil for {months} months in a row ({first} to {last})"


@pytest.mark.parametrize(
    ("crop_class", "ndvi", "problem"),
    [
        ("fallow/bare", BARE, None),
        ("fallow/bare", [0.1, 0.1, 0.1, 0.4, *[0.1] * 8], None),  # 0.4 is not above 0.4
        ("fallow/bare", [0.1, 0.1, 0.1, 0.8, *[0.1] * 8], "a crop grew: NDVI 0.80 in Nov"),
        ("grape", GREEN, None),
        ("grape", [0.4] * 12, "never green: NDVI peaked at 0.40"),
        ("other annual crops", curve(bare_months=range(9)), None),  # annuals have no limit
        ("avocado", curve(bare_months=(4, 5)), None),
        ("avocado", curve(bare_months=(4, 5, 6)), bare_for(3, "Dec", "Feb")),
        ("citrus", curve(bare_months=(0, 1, 2)), bare_for(3, "Aug", "Oct")),
        ("alfalfa", curve(bare_months=(9, 10, 11)), bare_for(3, "May", "Jul")),
        ("grape", curve(bare_months=(2, 3, 4, 5)), None),  # pruning
        ("grape", curve(bare_months=(2, 3, 4, 5, 6)), bare_for(5, "Oct", "Feb")),
        ("avocado", curve(bare_months=(1, 2, 3, 6), low=0.24), bare_for(3, "Sep", "Nov")),
        ("avocado", curve(bare_months=(1, 2), low=0.25), None),  # 0.25 is not below 0.25
    ],
)
def test_curve_rules(crop_class, ndvi, problem):
    assert curve_problem(crop_class, ndvi) == problem


def test_the_longest_bare_spell_is_reported():
    ndvi = curve(bare_months=(0, 1, 2, 5, 6, 7, 8))
    assert curve_problem("mango", ndvi) == "near bare soil for 4 months in a row (Jan to Apr)"


def test_the_curve_is_the_median_of_the_pixels():
    one_odd_pixel = pixels("ARE-0001", "2024/25", BARE, count=1)
    table = pd.concat([pixels("ARE-0001", "2024/25", GREEN), one_odd_pixel])

    curves = campaign_curves(table)

    assert list(curves.columns) == list(MONTHS)
    assert curves.loc[("ARE-0001", "2024/25")].tolist() == GREEN


def test_lists_only_the_campaigns_that_do_not_fit():
    table = pd.concat(
        [
            pixels("ARE-0002", "2024/25", BARE),  # fallow: fits
            pixels("ARE-0001", "2024/25", GREEN),  # grape: fits
            pixels("ARE-0001", "2023/24", BARE),  # grape before it was planted
            pixels("ARE-0003", "2024/25", curve(bare_months=range(4))),  # fallow, then a crop
        ]
    )
    rows = labels(
        ("ARE-0001", "grape", None), ("ARE-0002", "fallow", None), ("ARE-0003", "bare", "")
    )

    check = check_curves(table, rows)

    assert check.checked == 4
    assert [(m.parcel_id, m.campaign, m.crop_class) for m in check.mismatches] == [
        ("ARE-0001", "2023/24", "grape"),
        ("ARE-0003", "2024/25", "fallow/bare"),
    ]
    first = check.mismatches[0]
    assert first.crop == "grape"
    assert first.problem == "never green: NDVI peaked at 0.10"
    assert first.ndvi == tuple(BARE)
    assert check.unreviewed == check.mismatches


def test_the_crop_drop_down_value_decides_the_rule():
    table = pixels("ARE-0001", "2024/25", curve(bare_months=(0, 1, 2)))

    mandarine = check_curves(table, labels(("ARE-0001", "mandarine", None)))
    onion = check_curves(table, labels(("ARE-0001", "onion", None)))

    assert [m.crop_class for m in mandarine.mismatches] == ["citrus"]
    assert onion.mismatches == []


@pytest.mark.parametrize(
    ("notes", "reviewed"),
    [
        (f"{REVIEWED_NOTE} young orchard, planted 2024", True),
        ("Planted 2024. ndvi OK: young orchard", True),
        (REVIEWED_NOTE, False),  # a reason is required
        ("young orchard", False),
        (None, False),
        (np.nan, False),
    ],
)
def test_a_mismatch_is_reviewed_when_the_notes_say_so(notes, reviewed):
    table = pixels("ARE-0001", "2024/25", BARE)

    check = check_curves(table, labels(("ARE-0001", "avocado", notes)))

    assert [m.reviewed for m in check.mismatches] == [reviewed]
    assert len(check.unreviewed) == (0 if reviewed else 1)


def test_labels_without_notes_are_not_reviewed():
    table = pixels("ARE-0001", "2024/25", BARE)
    rows = pd.DataFrame([{"parcel_id": "ARE-0001", "crop": "avocado"}])

    assert check_curves(table, rows).unreviewed[0].parcel_id == "ARE-0001"


def test_no_pixels_means_nothing_to_check():
    empty = pd.DataFrame(columns=["parcel_id", "campaign", *FEATURE_COLUMNS])

    check = check_curves(empty, labels(("ARE-0001", "grape", None)))

    assert (check.checked, check.mismatches) == (0, [])
