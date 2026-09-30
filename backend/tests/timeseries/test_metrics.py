import numpy as np
import pandas as pd
import pytest

from fundoscope.timeseries import campaign_metrics, clean_series


def test_crop_campaign_versus_fallow_campaign(make_observations, crop_curve):
    crop = crop_curve(peak_day=150, peak=0.8, base=0.15)
    obs = make_observations(
        start="2022-08-01",
        end="2024-07-31",
        ndvi=lambda days: np.where(days < 365, crop(days), 0.15),  # crop, then a fallow year
    )
    table = campaign_metrics(clean_series(obs))

    assert list(table.index) == ["2022/23", "2023/24"]
    grown, fallow = table.loc["2022/23"], table.loc["2023/24"]
    assert grown["peak_ndvi"] == pytest.approx(0.8, abs=0.03)
    assert fallow["peak_ndvi"] == pytest.approx(0.15, abs=0.01)
    assert grown["integrated_ndvi"] > 3 * fallow["integrated_ndvi"]
    assert grown["mean_ndwi"] == pytest.approx(0.1)
    assert grown["n_obs"] == 73  # 365 days / 5
    assert fallow["n_obs"] == 74  # 2023/24 includes Feb 29, 2024: 366 days, ending on a 5th day
    assert grown["coverage"] > 0.95 and bool(grown["complete"]) and bool(fallow["complete"])


def test_integrated_ndvi_of_a_constant_curve(make_observations):
    # A flat NDVI of 0.5 for a whole year: (0.5 - 0.1) x 365 days = 146 NDVI-days.
    table = campaign_metrics(clean_series(make_observations(ndvi=0.5)))
    assert table.loc["2022/23", "integrated_ndvi"] == pytest.approx(146, rel=0.03)


def test_current_campaign_is_marked_incomplete(make_observations):
    table = campaign_metrics(clean_series(make_observations(start="2025-08-01", end="2025-10-31")))
    assert list(table.index) == ["2025/26"]
    assert not table.loc["2025/26", "complete"]


def test_series_starting_mid_campaign(make_observations):
    table = campaign_metrics(clean_series(make_observations(start="2023-01-01", end="2023-12-31")))
    assert list(table.index) == ["2022/23", "2023/24"]
    assert not table["complete"].any()


def test_long_cloudy_stretch_lowers_coverage(make_observations):
    obs = make_observations()
    winter = (obs.index >= "2022-11-01") & (obs.index < "2023-03-01")  # ~4 months of clouds
    obs.loc[winter, "valid_fraction"] = 0.1
    row = campaign_metrics(clean_series(obs)).loc["2022/23"]
    assert row["coverage"] == pytest.approx(245 / 365, abs=0.05)
    assert row["n_obs"] < 73


def test_campaign_without_any_data(make_observations):
    obs = pd.concat(
        [
            make_observations(start="2022-08-01", end="2022-09-30"),
            make_observations(start="2024-08-10", end="2024-09-30"),
        ]
    )
    table = campaign_metrics(clean_series(obs))
    assert list(table.index) == ["2022/23", "2023/24", "2024/25"]
    empty = table.loc["2023/24"]
    assert np.isnan(empty["peak_ndvi"]) and empty["n_obs"] == 0 and empty["coverage"] == 0


def test_no_usable_data_gives_an_empty_table(make_observations):
    table = campaign_metrics(clean_series(make_observations(valid_fraction=0.0)))
    assert table.empty
    assert "peak_ndvi" in table.columns
