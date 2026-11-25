import numpy as np
import pytest

from fundoscope.valley import farmland_mask

CROPLAND, BUILT_UP, WATER, BARE = 40, 50, 80, 60


def test_farmland_is_green_once_outside_towns_and_water_and_inside_the_valley():
    peak = np.array([[0.8, 0.8, 0.8, 0.3, np.nan, 0.41]])
    cover = np.array([[CROPLAND, BUILT_UP, WATER, CROPLAND, CROPLAND, BARE]])
    inside = np.array([[True, True, True, True, True, True]])

    assert farmland_mask(peak, cover, inside).tolist() == [[True, False, False, False, False, True]]


def test_pixels_outside_the_valley_are_not_farmland():
    mask = farmland_mask(np.array([[0.8, 0.8]]), np.array([[CROPLAND] * 2]), np.array([[1, 0]]))

    assert mask.tolist() == [[True, False]]


def test_the_grids_must_match():
    with pytest.raises(ValueError, match="same shape"):
        farmland_mask(np.zeros((2, 2)), np.zeros((2, 3)), np.ones((2, 2), bool))
