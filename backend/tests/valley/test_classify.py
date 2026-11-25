import numpy as np
import pytest

from fundoscope.ml import FeatureError
from fundoscope.valley import (
    NO_CLEAR_IMAGES,
    NOT_FARMLAND,
    PLANTED,
    ClassMap,
    classify_tile,
    map_classes,
    rising_from_bare,
)

GREEN, UNSURE, BARE = 0.8, 0.5, 0.09
CLASSES = ("alfalfa", "fallow/bare", "grape", "other annual crops", PLANTED, NO_CLEAR_IMAGES)


def code(name: str) -> int:
    return CLASSES.index(name) + 1


@pytest.fixture
def season(make_stack):
    """August to November over 2 rows x 4 columns:
    column 0 green all along, column 1 bare then sown (unsure in November), column 2 unsure
    all along (never bare), column 3 bare all along."""
    ndvi = np.empty((4, 2, 4))
    ndvi[:, :, 0] = GREEN
    ndvi[:, :, 1] = [[BARE] * 2] * 3 + [[UNSURE] * 2]
    ndvi[:, :, 2] = UNSURE
    ndvi[:, :, 3] = BARE
    return lambda: make_stack(ndvi)


def test_a_season_in_progress_counts_newly_sown_fields_as_planted(peak_model, season):
    farmland = np.ones((2, 4), bool)

    classes = classify_tile(season(), farmland, peak_model(4), in_progress=True)

    assert classes.classes == CLASSES
    assert classes.codes[0].tolist() == [
        code("grape"),
        code(PLANTED),  # unsure and rising from bare soil
        code("other annual crops"),  # unsure but never bare: the model's best guess
        code("fallow/bare"),
    ]
    assert classes.codes.dtype == np.uint8


def test_a_finished_campaign_has_no_planted_class(peak_model, season):
    classes = classify_tile(season(), np.ones((2, 4), bool), peak_model(4), in_progress=False)

    assert classes.codes[0, 1] == code("other annual crops")


def test_only_farmland_is_classified_and_cloudy_pixels_are_counted_apart(peak_model, season):
    data = season()
    data[:2, :, 1, 0] = np.nan  # two of four months missing: more than a third
    farmland = np.array([[True, True, True, False], [True, True, True, False]])

    classes = classify_tile(data, farmland, peak_model(4), in_progress=True)

    assert classes.codes[:, 3].tolist() == [NOT_FARMLAND, NOT_FARMLAND]
    assert classes.codes[1, 0] == code(NO_CLEAR_IMAGES)
    assert classes.counts() == {
        "alfalfa": 0,
        "fallow/bare": 0,
        "grape": 1,
        "other annual crops": 2,
        PLANTED: 2,
        NO_CLEAR_IMAGES: 1,
    }


def test_a_tile_without_any_usable_pixel(peak_model, season):
    data = season()
    data[:] = np.nan

    classes = classify_tile(data, np.ones((2, 4), bool), peak_model(4), in_progress=True)

    assert classes.counts()[NO_CLEAR_IMAGES] == 8


def test_blocks_of_rows_give_the_same_map(peak_model, season):
    whole = classify_tile(season(), np.ones((2, 4), bool), peak_model(4), in_progress=True)
    by_row = classify_tile(
        season(), np.ones((2, 4), bool), peak_model(4), in_progress=True, chunk_rows=1
    )

    assert np.array_equal(whole.codes, by_row.codes)


def test_the_model_must_match_the_number_of_months(peak_model, season):
    with pytest.raises(ValueError, match="expects 150 features; 4 months give 54"):
        classify_tile(season(), np.ones((2, 4), bool), peak_model(12), in_progress=True)


def test_the_mask_must_match_the_tile(peak_model, season):
    with pytest.raises(ValueError, match="farmland mask"):
        classify_tile(season(), np.ones((3, 4), bool), peak_model(4), in_progress=True)


def test_the_months_must_start_in_august(peak_model, make_stack):
    data = make_stack(np.full((4, 2, 4), GREEN), start="2026-09-01")

    with pytest.raises(FeatureError, match="August first"):
        classify_tile(data, np.ones((2, 4), bool), peak_model(4), in_progress=True)


def test_rising_from_bare_soil():
    curve = np.array(
        [
            [0.1, 0.8, 0.30, 0.1],
            [0.2, 0.8, 0.30, 0.1],
            [0.5, 0.8, 0.45, 0.15],
        ]
    )  # sown, perennial, never bare, still bare

    assert rising_from_bare(curve).tolist() == [True, False, False, False]


def test_one_month_cannot_rise():
    assert rising_from_bare(np.array([[0.1, 0.9]])).tolist() == [False, False]


def test_counts_include_every_class():
    classes = ClassMap(np.array([[0, 1, 1, 3]], np.uint8), ("a", "b", "c"))

    assert classes.counts() == {"a": 2, "b": 0, "c": 1}


def test_map_classes_add_the_rule_classes(peak_model):
    assert map_classes(peak_model()) == CLASSES
