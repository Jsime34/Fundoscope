import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fundoscope.ml import BANDS, FeatureError, predict_crop


def stack(green_columns: int, columns: int = 4, rows: int = 3, start: str = "2023-08-01"):
    """12 months over a small grid: the first `green_columns` columns are a growing crop
    (NDVI 0.8), the rest bare soil (NDVI ~0.09)."""
    data = np.full((12, len(BANDS), rows, columns), 0.1)
    band = {name: i for i, name in enumerate(BANDS)}
    data[:, band["B04"]] = 0.05
    data[:, band["B08"]] = 0.06
    data[:, band["B08"], :, :green_columns] = 0.45
    return xr.DataArray(
        data,
        dims=("month", "band", "y", "x"),
        coords={
            "month": pd.date_range(start, periods=12, freq="MS"),
            "band": list(BANDS),
            "y": 8_400_005.0 - 10 * np.arange(rows),
            "x": 400_005.0 + 10 * np.arange(columns),
        },
    )


def test_the_class_with_the_highest_mean_probability_wins(fake_crop_model):
    prediction = predict_crop(fake_crop_model, stack(green_columns=3))

    assert prediction.crop_class == "grape"
    assert prediction.confidence == pytest.approx((3 * 0.9 + 1 * 0.1) / 4)
    assert prediction.pixels == 12
    assert prediction.probabilities == pytest.approx({"fallow/bare": 0.3, "grape": 0.7})


def test_mostly_bare_soil_is_fallow(fake_crop_model):
    prediction = predict_crop(fake_crop_model, stack(green_columns=1))

    assert (prediction.crop_class, prediction.confidence) == ("fallow/bare", pytest.approx(0.7))


def test_only_pixels_inside_the_parcel_count(fake_crop_model):
    data = stack(green_columns=2)
    data[:, :, :, :2] = np.nan  # the green columns are outside the parcel

    prediction = predict_crop(fake_crop_model, data)

    assert (prediction.crop_class, prediction.pixels) == ("fallow/bare", 6)


def test_a_campaign_in_progress_gets_no_prediction(fake_crop_model):
    data = stack(green_columns=4)
    data[5:] = np.nan  # only August to December so far: 7 months missing

    assert predict_crop(fake_crop_model, data) is None


def test_a_stack_that_does_not_start_in_august_is_refused(fake_crop_model):
    with pytest.raises(FeatureError):
        predict_crop(fake_crop_model, stack(green_columns=4, start="2023-09-01"))
