import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fundoscope.ml import (
    BANDS,
    FEATURE_COLUMNS,
    MAX_PIXELS_PER_SAMPLE,
    FeatureError,
    feature_columns,
    pixel_features,
    season_to_date,
    training_sample,
)
from fundoscope.sentinel import COMPOSITE_BANDS

# NDVI over the campaign, August to July: low, peak in October (index 2), low again.
CURVE = np.array([0.2, 0.5, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2])
RED = 0.05


def nir_for(target_ndvi):
    """B08 reflectance that gives `target_ndvi` with red = 0.05."""
    return RED * (1 + target_ndvi) / (1 - target_ndvi)


def make_stack(ny=2, nx=3, start="2023-08-01", months=12, curve=CURVE) -> xr.DataArray:
    """Every pixel has the same NDVI curve; B8A = 0.3 and B11 = 0.1 give NDWI 0.5."""
    data = np.full((months, len(BANDS), ny, nx), 0.1)
    band = {name: i for i, name in enumerate(BANDS)}
    data[:, band["B04"]] = RED
    data[:, band["B08"]] = nir_for(np.asarray(curve)[:months])[:, None, None]
    data[:, band["B8A"]] = 0.3
    return xr.DataArray(
        data,
        dims=("month", "band", "y", "x"),
        coords={
            "month": pd.date_range(start, periods=months, freq="MS"),
            "band": list(BANDS),
            "y": 8_400_005.0 - 10 * np.arange(ny),
            "x": 400_005.0 + 10 * np.arange(nx),
        },
    )


def test_bands_match_the_composites():
    assert BANDS == COMPOSITE_BANDS


def test_every_pixel_gets_the_150_features():
    features = pixel_features(make_stack())

    assert features.shape == (6, 150)
    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert FEATURE_COLUMNS[:2] == ("B02_aug", "B02_sep")
    assert FEATURE_COLUMNS[-7:] == ("ndwi_jul", *FEATURE_COLUMNS[-6:])


def test_monthly_values_and_indices():
    row = pixel_features(make_stack()).iloc[0]

    assert row["B04_jan"] == pytest.approx(RED)
    assert row["ndvi_oct"] == pytest.approx(0.8)
    assert row["ndvi_jul"] == pytest.approx(0.2)
    assert row["ndwi_mar"] == pytest.approx(0.5)  # (B8A - B11) / (B8A + B11) = 0.2 / 0.4


def test_phenology_of_the_ndvi_curve():
    row = pixel_features(make_stack()).iloc[0]

    assert row["ndvi_peak"] == pytest.approx(0.8)
    assert row["ndvi_peak_month"] == 2  # October
    assert row["ndvi_min"] == pytest.approx(0.2)
    assert row["ndvi_amplitude"] == pytest.approx(0.6)
    assert row["ndvi_sum"] == pytest.approx(CURVE.sum())
    assert row["ndvi_months_active"] == 5  # 0.5, 0.8, 0.7, 0.6, 0.5 exceed 0.4


def test_pixels_are_indexed_by_their_utm_centers():
    index = pixel_features(make_stack()).index
    assert index.names == ["x", "y"]
    assert index[0] == (400_005.0, 8_400_005.0)
    assert index[-1] == (400_025.0, 8_399_995.0)


def test_a_missing_month_is_interpolated():
    stack = make_stack()
    stack[1, :, 0, 0] = np.nan  # September is cloudy for the first pixel

    row = pixel_features(stack).loc[(400_005.0, 8_400_005.0)]

    assert row["ndvi_sep"] == pytest.approx((0.2 + 0.8) / 2)
    assert row["B04_sep"] == pytest.approx(RED)


def test_missing_months_at_the_ends_take_the_nearest_value():
    stack = make_stack()
    stack[[0, 11], :, 0, 0] = np.nan

    row = pixel_features(stack).loc[(400_005.0, 8_400_005.0)]

    assert row["ndvi_aug"] == pytest.approx(0.5)
    assert row["ndvi_jul"] == pytest.approx(0.2)


def test_one_cloudy_band_makes_the_whole_month_missing():
    stack = make_stack()
    stack[2, BANDS.index("B11"), 0, 0] = np.nan  # October: only B11 is missing

    row = pixel_features(stack).loc[(400_005.0, 8_400_005.0)]

    assert row["ndvi_oct"] == pytest.approx((0.5 + 0.7) / 2)  # NDVI is not kept from a gap


def test_a_pixel_missing_more_than_4_months_is_left_out():
    stack = make_stack()
    stack[:4, :, 0, 0] = np.nan  # 4 months: kept
    stack[:5, :, 0, 1] = np.nan  # 5 months: left out

    index = pixel_features(stack).index

    assert (400_005.0, 8_400_005.0) in index
    assert (400_015.0, 8_400_005.0) not in index
    assert len(index) == 5


def test_pixels_outside_the_area_drop_out():
    stack = make_stack()
    stack[:, :, 1, :] = np.nan  # the caller masked the southern row
    features = pixel_features(stack)
    assert len(features) == 3
    assert set(features.index.get_level_values("y")) == {8_400_005.0}


def test_no_usable_pixel_gives_an_empty_table():
    stack = make_stack() * np.nan
    features = pixel_features(stack)
    assert features.shape == (0, 150)


def test_extra_bands_are_ignored():
    stack = make_stack()
    extra = stack.isel(band=[0]).assign_coords(band=["SCL"])
    features = pixel_features(xr.concat([stack, extra], dim="band"))
    assert features.shape == (6, 150)


@pytest.mark.parametrize(
    ("stack", "message"),
    [
        (make_stack(months=13, curve=np.resize(CURVE, 13)), "got 13 months starting 2023-08"),
        (make_stack(start="2023-09-01"), "starting 2023-09"),
        (make_stack(months=0), "no months"),
        (make_stack().isel(band=slice(0, 9)), "missing the bands B12"),
        (make_stack().isel(y=0), "Expected dims"),
    ],
)
def test_unusable_composites_are_rejected(stack, message):
    with pytest.raises(FeatureError, match=message):
        pixel_features(stack)


def test_months_must_be_consecutive():
    stack = make_stack()
    months = pd.date_range("2023-08-01", periods=13, freq="MS").delete(5)  # skips January
    with pytest.raises(FeatureError, match="consecutive"):
        pixel_features(stack.assign_coords(month=months))


# --- the first months of a campaign (season to date) -------------------------------------


def test_feature_columns_of_the_first_months():
    assert feature_columns() == FEATURE_COLUMNS
    assert feature_columns(1) == (
        *(f"{name}_aug" for name in (*BANDS, "ndvi", "ndwi")), *FEATURE_COLUMNS[-6:],
    )  # fmt: skip
    assert len(feature_columns(4)) == 4 * 12 + 6


@pytest.mark.parametrize("months", [0, 13])
def test_a_campaign_has_1_to_12_months(months):
    with pytest.raises(ValueError, match=f"not {months}"):
        feature_columns(months)


def test_the_first_months_of_a_campaign_get_their_own_features():
    features = pixel_features(make_stack(months=4))  # August to November

    assert list(features.columns) == list(feature_columns(4))
    row = features.iloc[0]
    assert row["ndvi_nov"] == pytest.approx(0.7)
    assert row["ndvi_peak"] == pytest.approx(0.8)
    assert row["ndvi_peak_month"] == 2
    assert row["ndvi_sum"] == pytest.approx(CURVE[:4].sum())
    assert row["ndvi_months_active"] == 3  # 0.5, 0.8, 0.7


def test_a_short_campaign_may_miss_a_third_of_its_months():
    stack = make_stack(months=3)
    stack[0, :, 0, 0] = np.nan  # 1 of 3 months: kept
    stack[:2, :, 0, 1] = np.nan  # 2 of 3 months: left out

    index = pixel_features(stack).index

    assert (400_005.0, 8_400_005.0) in index
    assert (400_015.0, 8_400_005.0) not in index
    stack = make_stack(months=2)
    stack[0, :, 0, 0] = np.nan  # with 2 months, none may be missing
    assert len(pixel_features(stack)) == 5


def test_season_to_date_matches_the_features_of_the_first_months():
    full = pixel_features(make_stack())

    for months in (1, 4, 11):
        pd.testing.assert_frame_equal(
            season_to_date(full, months), pixel_features(make_stack(months=months))
        )
    pd.testing.assert_frame_equal(season_to_date(full, 12), full)


def test_season_to_date_keeps_only_the_features():
    full = pixel_features(make_stack()).assign(parcel_id="ICA-0001")
    assert list(season_to_date(full, 4).columns) == list(feature_columns(4))


def test_a_small_parcel_keeps_all_its_pixels():
    features = pixel_features(make_stack())
    assert training_sample(features, "ICA-0001 2023/24") is features


def test_a_large_parcel_is_sampled_reproducibly():
    features = pixel_features(make_stack(ny=10, nx=10))

    first = training_sample(features, "ICA-0001 2023/24")
    again = training_sample(features, "ICA-0001 2023/24")
    other = training_sample(features, "ICA-0002 2023/24")

    assert len(first) == MAX_PIXELS_PER_SAMPLE
    assert first.index.equals(again.index)
    assert not first.index.equals(other.index)
    chosen = set(first.index)
    assert list(first.index) == [pixel for pixel in features.index if pixel in chosen]
