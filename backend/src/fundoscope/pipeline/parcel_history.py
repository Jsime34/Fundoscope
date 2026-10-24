"""Parcel history: the first end-to-end job (milestone M2).

uploaded file -> `ingest.parse_parcel` -> `sentinel.fetch_daily_stats`
-> `timeseries.clean_series` -> `timeseries.campaign_metrics`
(+ optionally `sentinel.fetch_monthly_composites` -> % active area per campaign)

The daily statistics and the monthly composites are downloaded at the same time. Composites
are requested one campaign at a time, two in parallel: one request for all 8 years failed
on the openEO side, while per-campaign requests have been reliable. A campaign whose request
fails twice gets no active area (NaN) instead of failing the whole history.

There is no database yet, so every run downloads again (about 20 processing units plus about
60 openEO credits for 8 years of a small parcel). Caching comes with the `db` module.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date

import pandas as pd

from fundoscope.ingest import Parcel, parse_parcel
from fundoscope.sentinel import (
    ARCHIVE_START,
    DailyStats,
    MonthlyComposites,
    SentinelError,
    fetch_daily_stats,
)
from fundoscope.timeseries import (
    CleanSeries,
    active_fraction,
    campaign_bounds,
    campaign_metrics,
    campaigns_between,
    clean_series,
    max_ndvi_composite,
    ndvi,
)

# Same call signatures as the sentinel functions; tests pass fakes that use no network.
Fetcher = Callable[..., DailyStats]
CompositeFetcher = Callable[..., MonthlyComposites]

COMPOSITE_WORKERS = 2  # openEO's free tier allows 2 requests at a time
COMPOSITE_ATTEMPTS = 2


@dataclass(frozen=True)
class ParcelHistory:
    """Everything computed for one parcel.

    Attributes:
        parcel: the validated parcel (geometry, UTM zone, area).
        daily: the downloaded daily statistics, with the processing units they cost.
        clean: the cleaned and smoothed series.
        campaigns: one row per campaign (see `timeseries.metrics`), plus `active_area`
            (0..1, NaN if not computed).
        credits: openEO credits spent on composites (0 if active area was not requested).
        active_area_failures: campaigns whose composites could not be downloaded.
    """

    parcel: Parcel
    daily: DailyStats
    clean: CleanSeries
    campaigns: pd.DataFrame
    credits: float = 0.0
    active_area_failures: int = 0


def parcel_history(
    data: bytes,
    filename: str,
    start: date = ARCHIVE_START,
    end: date | None = None,
    *,
    fetch: Fetcher = fetch_daily_stats,
    fetch_composites: CompositeFetcher | None = None,
) -> ParcelHistory:
    """Build a parcel's history from an uploaded KML/KMZ/GeoJSON file.

    Pass `fetch_composites` (e.g. `sentinel.fetch_monthly_composites`) to also compute
    % active area per campaign.

    Raises:
        ingest.ParcelError: the file isn't a usable parcel (nothing is downloaded then).
        sentinel.SentinelError: the daily statistics could not be downloaded.
    """
    parcel = parse_parcel(data, filename)
    end = end or date.today()
    labels = campaigns_between(start, end) if fetch_composites else []

    with ThreadPoolExecutor(max_workers=1 + COMPOSITE_WORKERS) as pool:
        daily_job = pool.submit(fetch, parcel.geometry_utm, parcel.utm_epsg, start, end)
        area_jobs = {
            label: pool.submit(_active_area, fetch_composites, parcel, label, start, end)
            for label in labels
        }
        daily = daily_job.result()
        areas = {label: job.result() for label, job in area_jobs.items()}

    clean = clean_series(daily.observations)
    campaigns = campaign_metrics(clean)
    no_value = _ActiveArea(float("nan"), 0.0, failed=False)
    campaigns["active_area"] = [areas.get(label, no_value).value for label in campaigns.index]
    return ParcelHistory(
        parcel=parcel,
        daily=daily,
        clean=clean,
        campaigns=campaigns,
        credits=sum(area.credits for area in areas.values()),
        active_area_failures=sum(area.failed for area in areas.values()),
    )


@dataclass(frozen=True)
class _ActiveArea:
    value: float  # 0..1, NaN if unknown
    credits: float
    failed: bool  # True if openEO could not deliver the composites (vs. simply no clear pixel)


def _active_area(
    fetch_composites: CompositeFetcher, parcel: Parcel, label: str, start: date, end: date
) -> _ActiveArea:
    """% active area of one campaign, clipped to [start, end]."""
    first, last = campaign_bounds(label)
    first, last = max(first, start), min(last, end)
    for _ in range(COMPOSITE_ATTEMPTS):
        try:
            composites = fetch_composites(
                parcel.geometry_utm, parcel.utm_epsg, first, last, bands=("B04", "B08")
            ).masked_to(parcel.geometry_utm)
        except SentinelError:
            continue  # openEO occasionally fails server-side; try once more
        peak = max_ndvi_composite(ndvi(composites.band("B04"), composites.band("B08")))
        try:
            return _ActiveArea(active_fraction(peak), composites.credits, failed=False)
        except ValueError:  # not a single clear pixel in the campaign: unknown, not a failure
            return _ActiveArea(float("nan"), composites.credits, failed=False)
    return _ActiveArea(float("nan"), 0.0, failed=True)
