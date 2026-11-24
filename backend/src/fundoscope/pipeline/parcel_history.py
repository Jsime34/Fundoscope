"""Parcel history: the first end-to-end job (milestone M2).

uploaded file -> `ingest.parse_parcel` -> `sentinel.fetch_daily_stats`
-> `timeseries.clean_series` -> `timeseries.campaign_metrics`
(+ optionally `sentinel.fetch_monthly_composites` -> % active area per campaign, and with a
trained model, the crop of each campaign)

The daily statistics and the monthly composites are downloaded at the same time. Composites
are requested one campaign at a time, two in parallel: one request for all 8 years failed
on the openEO side, while per-campaign requests have been reliable. A campaign whose request
fails twice gets no active area (NaN) instead of failing the whole history.

There is no database yet, so every run downloads again (about 20 processing units plus about
60 openEO credits for 8 years of a small parcel). Caching comes with the `db` module.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import xarray as xr
from shapely.geometry.base import BaseGeometry

from fundoscope.analysis import Flag, campaign_flags
from fundoscope.ingest import Parcel, parse_parcel
from fundoscope.ml import (
    INTERIOR_BUFFER_M,
    MONTHS,
    CropModel,
    CropPrediction,
    FeatureError,
    predict_crop,
)
from fundoscope.sentinel import (
    ARCHIVE_START,
    COMPOSITE_BANDS,
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
            (0..1, NaN if not computed), `crop` (the predicted model class, None if not
            predicted, e.g. a campaign in progress), and `crop_confidence` (0..1, or NaN).
        credits: openEO credits spent on composites (0 if active area was not requested).
        active_area_failures: campaigns whose composites could not be downloaded.
        flags: abandonment, crop change, anomaly, and low-data flags, in campaign order
            (see `analysis.flags`; abandonment and crop change need the composites).
    """

    parcel: Parcel
    daily: DailyStats
    clean: CleanSeries
    campaigns: pd.DataFrame
    credits: float = 0.0
    active_area_failures: int = 0
    flags: list[Flag] = field(default_factory=list)


def parcel_history(
    data: bytes,
    filename: str,
    start: date = ARCHIVE_START,
    end: date | None = None,
    *,
    fetch: Fetcher = fetch_daily_stats,
    fetch_composites: CompositeFetcher | None = None,
    model: CropModel | None = None,
) -> ParcelHistory:
    """Build a parcel's history from an uploaded KML/KMZ/GeoJSON file.

    Pass `fetch_composites` (e.g. `sentinel.fetch_monthly_composites`) to also compute
    % active area per campaign, and also `model` (the trained crop model) to predict the
    crop of each campaign from the same composites, with all their bands.

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
            label: pool.submit(
                _campaign_composites, fetch_composites, parcel, label, start, end, model
            )
            for label in labels
        }
        daily = daily_job.result()
        areas = {label: job.result() for label, job in area_jobs.items()}

    clean = clean_series(daily.observations)
    campaigns = campaign_metrics(clean)
    no_value = _CampaignResult(float("nan"), 0.0, failed=False)
    results = [areas.get(label, no_value) for label in campaigns.index]
    campaigns["active_area"] = [result.active_area for result in results]
    campaigns["crop"] = [result.crop.crop_class if result.crop else None for result in results]
    campaigns["crop_confidence"] = [
        result.crop.confidence if result.crop else float("nan") for result in results
    ]
    return ParcelHistory(
        parcel=parcel,
        daily=daily,
        clean=clean,
        campaigns=campaigns,
        credits=sum(area.credits for area in areas.values()),
        active_area_failures=sum(area.failed for area in areas.values()),
        flags=campaign_flags(campaigns),
    )


def interior_stack(
    composites: MonthlyComposites, geometry_utm: BaseGeometry, campaign: str
) -> xr.DataArray:
    """A campaign as the crop model sees it, for training and prediction alike.

    The 12 months from August (a month missing from the composites is all NaN), with every
    pixel less than 15 m inside the parcel (UTM meters) set to NaN.
    """
    first, _ = campaign_bounds(campaign)
    months = pd.date_range(first, periods=len(MONTHS), freq="MS")
    interior = geometry_utm.buffer(-INTERIOR_BUFFER_M)
    return composites.masked_to(interior).data.reindex(month=months)


@dataclass(frozen=True)
class _CampaignResult:
    active_area: float  # 0..1, NaN if unknown
    credits: float
    failed: bool  # True if openEO could not deliver the composites (vs. simply no clear pixel)
    crop: CropPrediction | None = None


def _campaign_composites(
    fetch_composites: CompositeFetcher,
    parcel: Parcel,
    label: str,
    start: date,
    end: date,
    model: CropModel | None,
) -> _CampaignResult:
    """% active area and, with a model, the crop of one campaign, clipped to [start, end]."""
    first, last = campaign_bounds(label)
    first, last = max(first, start), min(last, end)
    bands = COMPOSITE_BANDS if model is not None else ("B04", "B08")
    for _ in range(COMPOSITE_ATTEMPTS):
        try:
            composites = fetch_composites(
                parcel.geometry_utm, parcel.utm_epsg, first, last, bands=bands
            )
        except SentinelError:
            continue  # openEO occasionally fails server-side; try once more
        crop = _crop(model, composites, parcel, label) if model is not None else None
        whole = composites.masked_to(parcel.geometry_utm)
        peak = max_ndvi_composite(ndvi(whole.band("B04"), whole.band("B08")))
        try:
            area = active_fraction(peak)
        except ValueError:  # not a single clear pixel in the campaign: unknown, not a failure
            area = float("nan")
        return _CampaignResult(area, composites.credits, failed=False, crop=crop)
    return _CampaignResult(float("nan"), 0.0, failed=True)


def _crop(
    model: CropModel, composites: MonthlyComposites, parcel: Parcel, label: str
) -> CropPrediction | None:
    try:
        return predict_crop(model, interior_stack(composites, parcel.geometry_utm, label))
    except FeatureError:  # e.g. a band missing from the composites: no prediction
        return None
