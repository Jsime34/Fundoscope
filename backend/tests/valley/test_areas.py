import numpy as np
import pandas as pd
import pytest

from fundoscope.valley import NO_CLEAR_IMAGES, PLANTED, class_areas, olofsson

CLASSES = ["grape", "other"]


def confusion(rows):
    """Labeled class as rows, predicted class as columns."""
    return pd.DataFrame(rows, index=CLASSES, columns=CLASSES)


def test_a_perfect_model_keeps_the_mapped_area_with_no_spread():
    adjusted, error = olofsson(
        pd.Series([300.0, 700.0], index=CLASSES), confusion([[10, 0], [0, 30]])
    )

    assert adjusted.tolist() == pytest.approx([300, 700])
    assert error.tolist() == pytest.approx([0, 0])


def test_the_worked_example_of_olofsson_et_al_2014():
    """The paper's example (section 5, Table 8, results on p. 54): forest change in four
    classes, mapped from 30 m Landsat pixels (0.09 ha), with 640 reference samples stratified
    by map class. The same matrix is in the R package mapaccuracy (`olofsson()` examples)."""
    classes = ["deforestation", "forest gain", "stable forest", "stable non-forest"]
    pixels = pd.Series([200_000, 150_000, 3_200_000, 6_450_000], index=classes, dtype=float)
    # Table 8, sample counts: the map class as rows and the reference class as columns.
    table_8 = [
        [66, 0, 5, 4],
        [0, 55, 8, 12],
        [1, 0, 153, 11],
        [2, 1, 9, 313],
    ]
    # Ours is the other way around: the labeled (reference) class as rows.
    matrix = pd.DataFrame(table_8, index=classes, columns=classes).T

    adjusted, error = olofsson(pixels * 0.09, matrix)

    # As printed in the paper: 21,158 +- 6,158 ha of deforestation, and so on.
    assert adjusted.round().tolist() == [21_158, 11_686, 285_770, 581_386]
    assert (1.96 * error).round().tolist() == [6_158, 3_756, 15_510, 16_282]
    assert adjusted.sum() == pytest.approx(900_000)  # the map's total area is kept


def test_errors_move_area_between_classes():
    # Of 20 parcels predicted "other", 4 were grape; every predicted grape was grape.
    mapped = pd.Series([200.0, 800.0], index=CLASSES)
    matrix = confusion([[10, 4], [0, 16]])

    adjusted, error = olofsson(mapped, matrix)

    # p(grape) = 0.2 * 1 + 0.8 * 4/20 = 0.36; p(other) = 0.8 * 16/20 = 0.64
    assert adjusted.tolist() == pytest.approx([360, 640])
    # SE(grape) = sqrt(0.8^2 * 0.2 * 0.8 / 19) = 0.0734; both classes share it here
    assert error.tolist() == pytest.approx([1000 * np.sqrt(0.64 * 0.16 / 19)] * 2)
    assert adjusted.sum() == pytest.approx(mapped.sum())


def test_a_class_never_predicted_in_cross_validation_keeps_its_mapped_area():
    adjusted, error = olofsson(
        pd.Series([100.0, 900.0], index=CLASSES), confusion([[0, 5], [0, 20]])
    )

    assert adjusted["grape"] == pytest.approx(100 + 900 * 5 / 25)
    assert error["grape"] > 0


def test_a_class_predicted_once_has_no_spread_estimate():
    _, error = olofsson(pd.Series([100.0, 900.0], index=CLASSES), confusion([[1, 0], [0, 20]]))

    assert error.tolist() == pytest.approx([0, 0])


def test_an_empty_map_has_no_area():
    adjusted, error = olofsson(pd.Series([0.0, 0.0], index=CLASSES), confusion([[1, 0], [0, 1]]))

    assert adjusted.tolist() == [0, 0] and error.tolist() == [0, 0]


def test_class_areas_in_hectares_with_95_percent_ranges():
    pixels = {"grape": 20_000, "other": 80_000, PLANTED: 5_000, NO_CLEAR_IMAGES: 10}

    areas = class_areas(pixels, confusion([[10, 4], [0, 16]]))

    assert [area.name for area in areas] == ["grape", "other", PLANTED, NO_CLEAR_IMAGES]
    assert [area.hectares for area in areas] == pytest.approx([200, 800, 50, 0.1])
    grape = areas[0]
    half = 1.96 * 1000 * np.sqrt(0.64 * 0.16 / 19)
    assert (grape.adjusted, grape.low, grape.high) == pytest.approx((360, 360 - half, 360 + half))
    assert (areas[2].adjusted, areas[2].low, areas[2].high) == (None, None, None)


def test_ranges_never_go_below_zero():
    # All the map is "other", and 1 of the 3 parcels predicted "other" was grape: a wide range.
    areas = class_areas({"grape": 0, "other": 100_000}, confusion([[2, 1], [0, 2]]))

    assert areas[0].low == 0.0


def test_every_model_class_needs_a_count():
    with pytest.raises(ValueError, match="other"):
        class_areas({"grape": 10}, confusion([[1, 0], [0, 1]]))
