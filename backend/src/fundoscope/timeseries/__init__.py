"""Time-series processing.

Campaign grouping (Aug-Jul), spectral indices, cleaning and smoothing, campaign metrics, and
% active area (FR-04, FR-05). Pure: works on pandas/numpy data; no database or network.
"""

from fundoscope.timeseries.active_area import (
    ACTIVE_NDVI_THRESHOLD,
    active_fraction,
    active_mask,
    max_ndvi_composite,
)
from fundoscope.timeseries.campaigns import (
    campaign_bounds,
    campaign_labels,
    campaign_of,
    campaigns_between,
    parse_campaign,
)
from fundoscope.timeseries.clean import (
    GRID_ORIGIN,
    CleanSeries,
    TimeSeriesError,
    clean_series,
    drop_cloudy,
    drop_dips,
    regularize,
    smooth,
)
from fundoscope.timeseries.indices import ndvi, ndwi, normalized_difference
from fundoscope.timeseries.metrics import METRIC_COLUMNS, campaign_metrics

__all__ = [
    "ACTIVE_NDVI_THRESHOLD",
    "GRID_ORIGIN",
    "METRIC_COLUMNS",
    "CleanSeries",
    "active_fraction",
    "active_mask",
    "TimeSeriesError",
    "campaign_bounds",
    "campaign_labels",
    "campaign_metrics",
    "campaign_of",
    "campaigns_between",
    "clean_series",
    "drop_cloudy",
    "drop_dips",
    "max_ndvi_composite",
    "ndvi",
    "ndwi",
    "normalized_difference",
    "parse_campaign",
    "regularize",
    "smooth",
]
