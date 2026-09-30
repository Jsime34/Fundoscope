"""Per-campaign metrics for a parcel (FR-05), computed from a cleaned series.

One row per campaign (index: label such as "2023/24"), with these columns:

    peak_ndvi        highest smoothed NDVI in the campaign
    mean_ndvi        average smoothed NDVI over the campaign's grid dates that have data
    mean_ndwi        average smoothed NDWI (canopy water; irrigation shows up here)
    integrated_ndvi  area under the smoothed NDVI curve above a 0.1 baseline, in NDVI x days;
                     a proxy for how much the parcel produced that campaign
    n_obs            satellite dates kept after cloud and dip removal
    coverage         share of the campaign's grid dates (within the period the series covers)
                     that have data, 0..1; low values mean the numbers above are less reliable
    complete         False if the series starts after or ends before the campaign (e.g. the
                     current campaign), so its numbers cover only part of it

% active area is not here: it needs per-pixel data (see `active_area`).
"""

import numpy as np
import pandas as pd

from fundoscope.timeseries.campaigns import campaign_bounds, campaign_labels, campaigns_between
from fundoscope.timeseries.clean import CleanSeries

BASELINE_NDVI = 0.1

# The first image after August 1 can arrive a few days late; that still counts as complete.
COMPLETENESS_TOLERANCE_DAYS = 15

METRIC_COLUMNS = [
    "peak_ndvi",
    "mean_ndvi",
    "mean_ndwi",
    "integrated_ndvi",
    "n_obs",
    "coverage",
    "complete",
]


def campaign_metrics(clean: CleanSeries, baseline: float = BASELINE_NDVI) -> pd.DataFrame:
    """Summarize a cleaned series per campaign. Returns an empty table if there is no data."""
    observations, grid = clean.observations, clean.grid
    if observations.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS, index=pd.Index([], name="campaign"))

    first, last = observations.index[0], observations.index[-1]
    grid_campaign = campaign_labels(grid.index)
    obs_campaign = campaign_labels(observations.index)
    tolerance = pd.Timedelta(days=COMPLETENESS_TOLERANCE_DAYS)

    rows = {}
    for label in campaigns_between(first, last):
        start, end = (pd.Timestamp(d) for d in campaign_bounds(label))
        in_campaign = grid[grid_campaign == label]
        ndvi = in_campaign["ndvi_smooth"].to_numpy(dtype=np.float64)
        ndwi = in_campaign["ndwi_smooth"].to_numpy(dtype=np.float64)
        has_ndvi = ~np.isnan(ndvi)

        rows[label] = {
            "peak_ndvi": _nan_if_empty(np.max, ndvi[has_ndvi]),
            "mean_ndvi": _nan_if_empty(np.mean, ndvi[has_ndvi]),
            "mean_ndwi": _nan_if_empty(np.mean, ndwi[~np.isnan(ndwi)]),
            "integrated_ndvi": float(
                np.sum(np.clip(ndvi[has_ndvi] - baseline, 0.0, None)) * clean.step_days
            ),
            "n_obs": int(np.sum(obs_campaign == label)),
            "coverage": _coverage(has_ndvi.sum(), max(start, first), min(end, last), clean),
            "complete": bool(first <= start + tolerance and last >= end - tolerance),
        }

    table = pd.DataFrame.from_dict(rows, orient="index", columns=METRIC_COLUMNS)
    table.index.name = "campaign"
    return table


def _nan_if_empty(reducer, values: np.ndarray) -> float:
    return float(reducer(values)) if values.size else float("nan")


def _coverage(
    n_with_data: int, start: pd.Timestamp, end: pd.Timestamp, clean: CleanSeries
) -> float:
    """Grid dates with data / grid dates the series could have filled in [start, end]."""
    span_days = (end - start) / pd.Timedelta(days=1)
    expected = int(span_days // clean.step_days) + 1
    return min(1.0, n_with_data / expected) if expected > 0 else 0.0
