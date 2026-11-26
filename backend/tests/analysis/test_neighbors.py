import numpy as np
import pytest
from shapely.geometry import box

from fundoscope.analysis import (
    BANDS,
    UNDERPERFORMING,
    band_of,
    campaign_ndvi,
    compare,
    neighbor_ring,
    underperforming_flags,
)

PERCENTILES = {10: 0.3, 25: 0.4, 50: 0.5, 75: 0.6, 90: 0.7}


def test_the_ring_runs_from_50_m_to_2_km():
    parcel = box(0, 0, 100, 100)

    ring = neighbor_ring(parcel)

    assert not ring.contains(parcel.centroid)
    assert ring.distance(parcel) == pytest.approx(50, abs=0.5)
    assert ring.bounds == pytest.approx((-2000, -2000, 2100, 2100), abs=1)


def test_a_pixel_campaign_value_is_its_mean_over_clear_months():
    monthly = np.array([[[0.2, np.nan]], [[0.4, np.nan]], [[np.nan, np.nan]]])

    values = campaign_ndvi(monthly)

    assert values[0, 0] == pytest.approx(0.3)
    assert np.isnan(values[0, 1])  # never clear


@pytest.mark.parametrize(
    ("value", "band"),
    [(0.1, "below p10"), (0.3, "p10-p25"), (0.45, "p25-p50"), (0.55, "p50-p75"), (0.65, "p75-p90"),
     (0.7, "above p90"), (0.95, "above p90")],
)  # fmt: skip
def test_bands(value, band):
    assert band_of(value, PERCENTILES) == band
    assert band in BANDS


def test_compare_places_the_parcel_among_its_neighbors():
    neighbors = np.linspace(0.2, 0.8, 101)

    band = compare("2025/26", 12, [0.21, 0.23, np.nan], neighbors, min_neighbors=100)

    assert (band.campaign, band.months, band.band) == ("2025/26", 12, "below p10")
    assert band.parcel_ndvi == pytest.approx(0.22)
    assert (band.parcel_pixels, band.neighbor_pixels) == (2, 101)
    assert band.percentiles[50] == pytest.approx(0.5)
    assert band.underperforming


def test_no_comparison_without_parcel_pixels_or_enough_neighbors():
    assert compare("2025/26", 12, [np.nan], np.linspace(0, 1, 200)) is None
    assert compare("2025/26", 12, [0.5], np.linspace(0, 1, 99)) is None


def test_underperforming_campaigns_get_a_flag_with_their_evidence():
    low = compare("2024/25", 12, [0.1], np.linspace(0.2, 0.8, 101), min_neighbors=1)
    fine = compare("2025/26", 12, [0.5], np.linspace(0.2, 0.8, 101), min_neighbors=1)

    flags = underperforming_flags([low, fine])

    assert [(flag.campaign, flag.kind) for flag in flags] == [("2024/25", UNDERPERFORMING)]
    assert "Less green than 90 %" in flags[0].message
    assert flags[0].evidence["neighbor_pixels"] == 101
    assert flags[0].evidence["months_compared"] == 12
