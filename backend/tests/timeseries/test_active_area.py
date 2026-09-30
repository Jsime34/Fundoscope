import numpy as np
import pytest

from fundoscope.timeseries import active_fraction, active_mask, max_ndvi_composite

NAN = np.nan


def test_composite_keeps_each_pixels_peak_and_ignores_clouds():
    # 3 dates of a 2x2 parcel; NaN = cloud that day.
    stack = np.array(
        [
            [[0.2, NAN], [0.1, NAN]],
            [[0.7, 0.3], [NAN, NAN]],
            [[0.5, 0.6], [0.15, NAN]],
        ]
    )
    composite = max_ndvi_composite(stack)
    assert composite[0, 0] == pytest.approx(0.7)
    assert composite[0, 1] == pytest.approx(0.6)
    assert composite[1, 0] == pytest.approx(0.15)
    assert np.isnan(composite[1, 1])  # never seen clear: stays NaN, and no warning escapes


def test_composite_needs_a_time_axis():
    with pytest.raises(ValueError, match="time, y, x"):
        max_ndvi_composite(np.zeros((4, 4)))


def test_half_of_the_parcel_farmed():
    peak = np.array([[0.8, 0.75], [0.2, 0.18]])  # top row cropped, bottom row fallow
    assert active_fraction(peak) == pytest.approx(0.5)


def test_pixels_outside_the_parcel_do_not_count():
    peak = np.array([[0.8, NAN, NAN], [0.2, 0.7, NAN]])  # NaN = outside the polygon
    assert active_fraction(peak) == pytest.approx(2 / 3)
    assert active_mask(peak).tolist() == [[True, False, False], [False, True, False]]


def test_threshold_is_exclusive_and_adjustable():
    peak = np.array([0.4, 0.41, 0.35])
    assert active_fraction(peak) == pytest.approx(1 / 3)  # exactly 0.4 is not active
    assert active_fraction(peak, threshold=0.3) == pytest.approx(1.0)


def test_empty_composite_is_rejected():
    with pytest.raises(ValueError, match="No observed pixels"):
        active_fraction(np.full((3, 3), NAN))
