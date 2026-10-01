"""Parcel history: the first end-to-end job (milestone M2).

uploaded file -> `ingest.parse_parcel` -> `sentinel.fetch_daily_stats`
-> `timeseries.clean_series` -> `timeseries.campaign_metrics`

There is no database yet, so every run downloads the series again (about 20 processing
units for 8 years of a small parcel). Caching in PostGIS comes with the `db` module.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import pandas as pd

from fundoscope.ingest import Parcel, parse_parcel
from fundoscope.sentinel import ARCHIVE_START, DailyStats, fetch_daily_stats
from fundoscope.timeseries import CleanSeries, campaign_metrics, clean_series

# Same call signature as `fetch_daily_stats`; tests pass a fake that doesn't use the network.
Fetcher = Callable[..., DailyStats]


@dataclass(frozen=True)
class ParcelHistory:
    """Everything computed for one parcel.

    Attributes:
        parcel: the validated parcel (geometry, UTM zone, area).
        daily: the downloaded daily statistics, with the processing units they cost.
        clean: the cleaned and smoothed series.
        campaigns: one row per campaign (see `timeseries.metrics`).
    """

    parcel: Parcel
    daily: DailyStats
    clean: CleanSeries
    campaigns: pd.DataFrame


def parcel_history(
    data: bytes,
    filename: str,
    start: date = ARCHIVE_START,
    end: date | None = None,
    *,
    fetch: Fetcher = fetch_daily_stats,
) -> ParcelHistory:
    """Build a parcel's history from an uploaded KML/KMZ/GeoJSON file.

    Raises:
        ingest.ParcelError: the file isn't a usable parcel (nothing is downloaded then).
        sentinel.SentinelError: the satellite service returned unusable data.
    """
    parcel = parse_parcel(data, filename)
    daily = fetch(parcel.geometry_utm, parcel.utm_epsg, start, end)
    clean = clean_series(daily.observations)
    return ParcelHistory(parcel, daily, clean, campaign_metrics(clean))
