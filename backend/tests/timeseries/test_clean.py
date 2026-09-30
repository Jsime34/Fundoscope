import numpy as np
import pandas as pd
import pytest

from fundoscope.timeseries import (
    GRID_ORIGIN,
    TimeSeriesError,
    clean_series,
    drop_cloudy,
    drop_dips,
    regularize,
    smooth,
)

# --- step 1: drop_cloudy -------------------------------------------------------------------


def test_cloudy_and_empty_dates_are_dropped(make_observations):
    obs = make_observations(end="2022-08-31")  # 7 dates
    obs.loc[obs.index[2], "valid_fraction"] = 0.3  # mostly cloud
    obs.loc[obs.index[4], "ndvi"] = np.nan  # no clear pixel at all
    kept = drop_cloudy(obs)
    assert list(kept.index) == list(obs.index.delete([2, 4]))


# --- step 2: drop_dips ---------------------------------------------------------------------


def test_isolated_dip_is_removed_but_a_harvest_is_kept(observations_from_values):
    #         0    1    2 (dip)  3    4    5 (harvest: stays down)
    values = [0.7, 0.7, 0.30, 0.7, 0.7, 0.2, 0.2, 0.2]
    obs = observations_from_values(values)
    kept = drop_dips(obs)
    assert list(kept.index) == list(obs.index.delete(2))


def test_dip_between_distant_neighbors_is_kept(observations_from_values):
    # 20 days between dates: too far apart to tell a cloud from a real change.
    obs = observations_from_values([0.7, 0.3, 0.7], step_days=20)
    assert len(drop_dips(obs)) == 3


def test_first_and_last_dates_are_never_dropped(observations_from_values):
    obs = observations_from_values([0.1, 0.7, 0.7, 0.1])
    assert len(drop_dips(obs)) == 4


# --- step 3: regularize --------------------------------------------------------------------


def test_grid_is_anchored_so_every_parcel_shares_the_same_dates(make_observations):
    obs = make_observations(start="2022-08-03", end="2022-12-31", step_days=7)
    grid = regularize(obs)
    offsets = (grid.index - GRID_ORIGIN).days
    assert all(offsets % 5 == 0)
    assert grid.index[0] >= obs.index[0] and grid.index[-1] <= obs.index[-1]  # no extrapolation


def test_values_between_close_observations_are_interpolated():
    start = GRID_ORIGIN + pd.Timedelta(days=1000)  # on the grid
    obs = pd.DataFrame(
        {"ndvi": [0.2, 0.6], "ndwi": [0.0, 0.2], "valid_fraction": 1.0},
        index=pd.DatetimeIndex([start, start + pd.Timedelta(days=10)], name="date"),
    )
    grid = regularize(obs)
    assert list(grid["ndvi"]) == pytest.approx([0.2, 0.4, 0.6])
    assert list(grid["ndwi"]) == pytest.approx([0.0, 0.1, 0.2])


def test_long_gaps_stay_empty():
    start = GRID_ORIGIN + pd.Timedelta(days=1000)
    obs = pd.DataFrame(
        {"ndvi": [0.2, 0.6], "ndwi": 0.1, "valid_fraction": 1.0},
        index=pd.DatetimeIndex([start, start + pd.Timedelta(days=40)], name="date"),
    )
    grid = regularize(obs)
    assert grid["ndvi"].iloc[0] == pytest.approx(0.2)
    assert grid["ndvi"].iloc[-1] == pytest.approx(0.6)
    assert grid["ndvi"].iloc[1:-1].isna().all()  # 40 days > 30-day limit: not invented


# --- step 4: smooth ------------------------------------------------------------------------


def test_smoothing_brings_a_noisy_curve_closer_to_the_truth(make_observations, crop_curve):
    truth_fn = crop_curve(peak_day=150)
    grid = regularize(make_observations(ndvi=truth_fn))
    truth = grid["ndvi"].to_numpy()
    rng = np.random.default_rng(7)
    grid["ndvi"] = truth + rng.normal(0, 0.04, len(truth))

    smoothed = smooth(grid)["ndvi_smooth"].to_numpy()

    def rmse(a):
        return np.sqrt(np.mean((a - truth) ** 2))

    assert rmse(smoothed) < 0.7 * rmse(grid["ndvi"].to_numpy())


def test_gaps_stay_empty_and_short_runs_are_copied():
    values = [0.5] * 10 + [np.nan] * 5 + [0.3, 0.6, 0.4]  # the last run is too short to fit
    grid = pd.DataFrame({"ndvi": values, "ndwi": values})
    smoothed = smooth(grid)["ndvi_smooth"].to_numpy()
    assert np.isnan(smoothed[10:15]).all()
    assert smoothed[15:] == pytest.approx([0.3, 0.6, 0.4])


def test_smoothed_values_stay_within_the_index_range():
    rng = np.random.default_rng(1)
    values = np.clip(0.98 + rng.normal(0, 0.05, 40), -1, 1)
    grid = pd.DataFrame({"ndvi": values, "ndwi": values})
    assert smooth(grid)["ndvi_smooth"].max() <= 1.0


# --- all steps -----------------------------------------------------------------------------


def test_clean_series_end_to_end(make_observations, crop_curve):
    obs = make_observations(ndvi=crop_curve(peak_day=150, peak=0.8))
    cloudy_dates = obs.index[[10, 40, 60]]
    obs.loc[cloudy_dates, "valid_fraction"] = 0.2
    obs.loc[obs.index[20], "ndvi"] -= 0.3  # leftover cloud: an isolated dip

    result = clean_series(obs)

    assert result.dropped_cloudy == 3
    assert result.dropped_dips == 1
    assert result.step_days == 5
    assert {"ndvi", "ndwi", "ndvi_smooth", "ndwi_smooth"} <= set(result.grid.columns)
    assert result.grid["ndvi_smooth"].max() == pytest.approx(0.8, abs=0.03)


def test_clean_series_accepts_unsorted_input(make_observations):
    obs = make_observations(end="2022-10-31")
    result = clean_series(obs.iloc[::-1])
    assert result.observations.index.is_monotonic_increasing


def test_all_cloudy_gives_an_empty_grid(make_observations):
    result = clean_series(make_observations(end="2022-09-30", valid_fraction=0.1))
    assert result.observations.empty
    assert result.grid.empty


@pytest.mark.parametrize(
    "broken",
    [
        pd.DataFrame({"ndvi": [0.5], "ndwi": [0.1], "valid_fraction": [1.0]}),  # no dates
        pd.DataFrame({"ndvi": [0.5]}, index=pd.DatetimeIndex(["2022-08-01"])),  # missing columns
        pd.DataFrame(
            {"ndvi": [0.5, 0.6], "ndwi": 0.1, "valid_fraction": 1.0},
            index=pd.DatetimeIndex(["2022-08-01", "2022-08-01"]),
        ),  # same date twice
    ],
)
def test_bad_input_is_rejected(broken):
    with pytest.raises(TimeSeriesError):
        clean_series(broken)


def test_a_column_without_values_stays_empty_on_the_grid(make_observations):
    obs = make_observations(end="2022-10-31")
    obs["ndwi"] = np.nan  # e.g. the SWIR band was unavailable
    grid = clean_series(obs).grid
    assert grid["ndvi"].notna().all()
    assert grid["ndwi"].isna().all() and grid["ndwi_smooth"].isna().all()
