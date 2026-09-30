import numpy as np
import pytest

from fundoscope.timeseries import ndvi, ndwi, normalized_difference


def test_ndvi_of_healthy_vegetation():
    # Plants reflect little red and a lot of near-infrared.
    assert ndvi(red=0.05, nir=0.45) == pytest.approx(0.8)


def test_ndvi_of_bare_soil_is_low():
    assert ndvi(red=0.25, nir=0.30) == pytest.approx(0.0909, abs=1e-4)


def test_ndwi_uses_nir_and_swir():
    assert ndwi(nir=0.40, swir=0.20) == pytest.approx(1 / 3)


def test_equal_bands_give_zero_and_empty_bands_give_nan():
    assert normalized_difference(0.3, 0.3) == 0
    assert np.isnan(normalized_difference(0.0, 0.0))


def test_works_on_arrays_without_warnings():
    red = np.array([[0.05, 0.0], [0.2, 0.1]])
    nir = np.array([[0.45, 0.0], [0.2, 0.5]])
    with np.errstate(all="raise"):  # any hidden division warning would fail the test
        result = ndvi(red, nir)
    assert result.shape == (2, 2)
    assert result[0, 0] == pytest.approx(0.8)
    assert np.isnan(result[0, 1])
    assert result[1, 0] == 0


def test_values_stay_between_minus_one_and_one():
    rng = np.random.default_rng(42)
    red, nir = rng.uniform(0, 1, 1_000), rng.uniform(0, 1, 1_000)
    values = ndvi(red, nir)
    assert np.all((values >= -1) & (values <= 1))
