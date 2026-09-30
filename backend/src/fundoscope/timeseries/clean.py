"""Clean a parcel's Sentinel-2 time series before computing campaign metrics (FR-04).

Input: a DataFrame indexed by acquisition date (DatetimeIndex, one row per date) with columns
`ndvi` and `ndwi` (parcel means over clear pixels) and `valid_fraction` (0..1, the share of
the parcel's pixels that were clear on that date).

Steps (thresholds are starting values, to be calibrated on partner parcels):

1. `drop_cloudy`: remove dates where less than 60 % of the parcel was clear.
2. `drop_dips`: remove isolated NDVI drops, i.e. a date more than 0.15 below BOTH of its
   neighbors, each within 15 days. Crops don't lose and regain that much greenness within
   two weeks, so these are leftover clouds, haze, or shadows. A harvest (a drop that stays
   down) is kept.
3. `regularize`: put the series on a 5-day grid by linear interpolation in time, filling
   only gaps of up to 30 days. Longer gaps stay empty (NaN) instead of being invented.
4. `smooth`: Savitzky-Golay filter (7 grid points = 30 days, polynomial order 2) on each
   unbroken stretch of the grid; adds `<column>_smooth` columns.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

MIN_VALID_FRACTION = 0.6
DIP_MIN_DROP = 0.15
DIP_WINDOW_DAYS = 15
GRID_STEP_DAYS = 5
MAX_GAP_DAYS = 30
SMOOTH_WINDOW = 7
SMOOTH_POLYORDER = 2

# Every grid is anchored here, so all parcels share the same grid dates and can be compared.
GRID_ORIGIN = pd.Timestamp("2018-08-01")

REQUIRED_COLUMNS = ("ndvi", "ndwi", "valid_fraction")
SERIES_COLUMNS = ("ndvi", "ndwi")


class TimeSeriesError(ValueError):
    """The input series can't be processed (wrong shape or content)."""


@dataclass(frozen=True)
class CleanSeries:
    """Result of `clean_series`.

    Attributes:
        observations: the dates kept after steps 1-2 (irregular, as acquired).
        grid: the 5-day grid with `ndvi`, `ndwi`, `ndvi_smooth`, `ndwi_smooth` (NaN in long gaps).
        step_days: grid spacing in days.
        dropped_cloudy: number of dates removed in step 1.
        dropped_dips: number of dates removed in step 2.
    """

    observations: pd.DataFrame
    grid: pd.DataFrame
    step_days: int
    dropped_cloudy: int
    dropped_dips: int


def clean_series(observations: pd.DataFrame) -> CleanSeries:
    """Run all four steps with the default thresholds."""
    _check_input(observations)
    raw = observations.sort_index()
    clear = drop_cloudy(raw)
    kept = drop_dips(clear)
    grid = smooth(regularize(kept))
    return CleanSeries(
        observations=kept,
        grid=grid,
        step_days=GRID_STEP_DAYS,
        dropped_cloudy=len(raw) - len(clear),
        dropped_dips=len(clear) - len(kept),
    )


def drop_cloudy(
    observations: pd.DataFrame, min_valid_fraction: float = MIN_VALID_FRACTION
) -> pd.DataFrame:
    """Step 1: keep dates with enough clear pixels and a usable NDVI value."""
    keep = (observations["valid_fraction"] >= min_valid_fraction) & observations["ndvi"].notna()
    return observations[keep]


def drop_dips(
    observations: pd.DataFrame,
    column: str = "ndvi",
    min_drop: float = DIP_MIN_DROP,
    window_days: float = DIP_WINDOW_DAYS,
) -> pd.DataFrame:
    """Step 2: remove dates that are much lower than both close neighbors.

    Neighbors are compared as acquired (not after earlier removals), and the first and last
    dates are never removed because they only have one neighbor.
    """
    if len(observations) < 3:
        return observations
    values = observations[column].to_numpy(dtype=np.float64)
    days = _days_since(observations.index, observations.index[0])

    middle = values[1:-1]
    close_neighbors = (days[1:-1] - days[:-2] <= window_days) & (
        days[2:] - days[1:-1] <= window_days
    )
    dip = close_neighbors & (middle < values[:-2] - min_drop) & (middle < values[2:] - min_drop)

    keep = np.ones(len(values), dtype=bool)
    keep[1:-1] = ~dip
    return observations[keep]


def regularize(
    observations: pd.DataFrame,
    columns: tuple[str, ...] = SERIES_COLUMNS,
    step_days: int = GRID_STEP_DAYS,
    max_gap_days: float = MAX_GAP_DAYS,
) -> pd.DataFrame:
    """Step 3: resample onto a regular grid anchored at GRID_ORIGIN.

    A grid date gets a value only if it matches an observation or lies between two
    observations at most `max_gap_days` apart. Nothing is extrapolated past the first or
    last observation.
    """
    if observations.empty:
        return pd.DataFrame(columns=list(columns), index=pd.DatetimeIndex([], name="date"))

    first = _days_since(observations.index[:1], GRID_ORIGIN)[0]
    last = _days_since(observations.index[-1:], GRID_ORIGIN)[0]
    k = np.arange(np.ceil(first / step_days), np.floor(last / step_days) + 1)
    grid_days = k * step_days
    grid_index = pd.DatetimeIndex(GRID_ORIGIN + pd.to_timedelta(grid_days, unit="D"), name="date")

    obs_days = _days_since(observations.index, GRID_ORIGIN)
    data = {
        col: _interpolate_with_gaps(
            obs_days, observations[col].to_numpy(dtype=np.float64), grid_days, max_gap_days
        )
        for col in columns
    }
    return pd.DataFrame(data, index=grid_index)


def smooth(
    grid: pd.DataFrame,
    columns: tuple[str, ...] = SERIES_COLUMNS,
    window: int = SMOOTH_WINDOW,
    polyorder: int = SMOOTH_POLYORDER,
) -> pd.DataFrame:
    """Step 4: add `<column>_smooth`, smoothing each unbroken run of grid values.

    Runs shorter than `window` are copied unchanged (too short to fit the filter). Values
    are clipped to the valid index range [-1, 1].
    """
    result = grid.copy()
    for col in columns:
        values = grid[col].to_numpy(dtype=np.float64)
        smoothed = values.copy()
        for start, end in _runs_of_values(values):
            if end - start >= window:
                smoothed[start:end] = savgol_filter(values[start:end], window, polyorder)
        result[f"{col}_smooth"] = np.clip(smoothed, -1.0, 1.0)
    return result


# --- helpers ------------------------------------------------------------------------------


def _check_input(observations: pd.DataFrame) -> None:
    if not isinstance(observations.index, pd.DatetimeIndex):
        raise TimeSeriesError("Observations must be indexed by date (a DatetimeIndex).")
    missing = [c for c in REQUIRED_COLUMNS if c not in observations.columns]
    if missing:
        raise TimeSeriesError(f"Observations are missing the columns: {', '.join(missing)}.")
    if observations.index.has_duplicates:
        raise TimeSeriesError("Observations contain the same date more than once.")


def _days_since(index: pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    return ((index - origin) / pd.Timedelta(days=1)).to_numpy(dtype=np.float64)


def _interpolate_with_gaps(
    obs_days: np.ndarray, obs_values: np.ndarray, grid_days: np.ndarray, max_gap: float
) -> np.ndarray:
    known = ~np.isnan(obs_values)
    t, v = obs_days[known], obs_values[known]
    out = np.full(len(grid_days), np.nan)
    if len(t) == 0:
        return out

    after = np.searchsorted(t, grid_days, side="left")  # index of the first obs >= grid day
    exact = (after < len(t)) & (t[np.minimum(after, len(t) - 1)] == grid_days)
    between = (after > 0) & (after < len(t))
    gap = np.full(len(grid_days), np.inf)
    gap[between] = t[after[between]] - t[after[between] - 1]

    use = exact | (between & (gap <= max_gap))
    out[use] = np.interp(grid_days[use], t, v)
    return out


def _runs_of_values(values: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) index pairs of consecutive non-NaN values; end is exclusive."""
    has_value = np.concatenate(([0], (~np.isnan(values)).astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(has_value))
    return list(zip(edges[::2], edges[1::2], strict=True))
