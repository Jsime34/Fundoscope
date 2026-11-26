"""Parcel history: the first end-to-end job (milestone M2).

uploaded file -> `ingest.parse_parcel` -> `sentinel.fetch_daily_stats`
-> `timeseries.clean_series` -> `timeseries.campaign_metrics`
(+ optionally `sentinel.fetch_monthly_composites` -> % active area per campaign, and with a
trained model, the crop of each campaign)
(+ optionally, for a parcel inside a counted valley, the neighbor comparison, FR-08)

The daily statistics and the monthly composites are downloaded at the same time. Composites
are requested one campaign at a time, two in parallel: one request for all 8 years failed
on the openEO side, while per-campaign requests have been reliable. A campaign whose request
fails twice gets no active area (NaN) instead of failing the whole history.

`history_of` starts from what is stored about the parcel (FR-02) and downloads only what is
missing; `pipeline.parcel_store` reads and saves the store. A first 8-year history of a
small parcel costs about 20 processing units plus about 60 openEO credits.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import xarray as xr
from shapely.geometry.base import BaseGeometry

from fundoscope.analysis import Flag, campaign_flags
from fundoscope.analysis.neighbors import underperforming_flags
from fundoscope.db.records import StoredCampaign, StoredHistory, empty_observations
from fundoscope.ingest import Parcel, parse_parcel
from fundoscope.ml import (
    INTERIOR_BUFFER_M,
    MONTHS,
    CropModel,
    CropPrediction,
    FeatureError,
    predict_crop,
)
from fundoscope.pipeline.neighbors import NeighborReport, neighbor_report
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
# Called with (requests done, requests in all) as the downloads finish (FR-13 progress).
Progress = Callable[[int, int], None]

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
        flags: abandonment, crop change, anomaly, low-data, and underperforming flags, in
            campaign order (see `analysis.flags`; abandonment and crop change need the
            composites, underperforming needs the neighbor comparison).
        neighbors: the neighbor comparison (None when it wasn't asked for).
        campaign_results: what the composites gave for each campaign, reused or computed in
            this run (failed requests are left out, so they are tried again), to store.
    """

    parcel: Parcel
    daily: DailyStats
    clean: CleanSeries
    campaigns: pd.DataFrame
    credits: float = 0.0
    active_area_failures: int = 0
    flags: list[Flag] = field(default_factory=list)
    neighbors: NeighborReport | None = None
    campaign_results: dict[str, StoredCampaign] = field(default_factory=dict)


def parcel_history(
    data: bytes,
    filename: str,
    start: date = ARCHIVE_START,
    end: date | None = None,
    *,
    fetch: Fetcher = fetch_daily_stats,
    fetch_composites: CompositeFetcher | None = None,
    model: CropModel | None = None,
    valleys: Path | None = None,
) -> ParcelHistory:
    """Build a parcel's history from an uploaded KML/KMZ/GeoJSON file.

    Pass `fetch_composites` (e.g. `sentinel.fetch_monthly_composites`) to also compute
    % active area per campaign, and also `model` (the trained crop model) to predict the
    crop of each campaign from the same composites, with all their bands. Pass `valleys`
    (data/valley) for the neighbor comparison of a parcel inside a counted valley; it reads
    the valley's saved data and makes no satellite request.

    Raises:
        ingest.ParcelError: the file isn't a usable parcel (nothing is downloaded then).
        sentinel.SentinelError: the daily statistics could not be downloaded.
    """
    return history_of(
        parse_parcel(data, filename),
        start,
        end,
        fetch=fetch,
        fetch_composites=fetch_composites,
        model=model,
        valleys=valleys,
    )


def history_of(
    parcel: Parcel,
    start: date = ARCHIVE_START,
    end: date | None = None,
    *,
    fetch: Fetcher | None = fetch_daily_stats,
    fetch_composites: CompositeFetcher | None = None,
    model: CropModel | None = None,
    model_version: str | None = None,
    valleys: Path | None = None,
    stored: StoredHistory | None = None,
    on_progress: Progress | None = None,
) -> ParcelHistory:
    """The history of a validated parcel, starting from what is `stored` about it (FR-02).

    Only the dates after the stored ones are downloaded, and only the campaigns whose
    composites aren't stored are requested: a finished campaign is computed once, the one in
    progress once a day (and again if the crop model changed: `model_version`). With
    `fetch=None` and `fetch_composites=None`, nothing is downloaded: the history is rebuilt
    from the stored data alone.

    `on_progress` is called with (0, n) before the n downloads start (n may be 0), then
    with (1, n), (2, n), ... as each one finishes.

    Raises:
        sentinel.SentinelError: the daily statistics could not be downloaded.
    """
    end = end or date.today()
    stored = stored or StoredHistory(empty_observations())
    known_until = stored.observed_through
    fetch_from = max(start, known_until + timedelta(days=1)) if known_until else start
    labels = campaigns_between(start, end)
    reuse = {
        label: stored.campaigns[label]
        for label in labels
        if label in stored.campaigns
        and (
            fetch_composites is None
            or _reusable(stored.campaigns[label], label, end, model, model_version)
        )
    }
    to_compute = [label for label in labels if label not in reuse] if fetch_composites else []

    with ThreadPoolExecutor(max_workers=1 + COMPOSITE_WORKERS) as pool:
        daily_job = (
            pool.submit(fetch, parcel.geometry_utm, parcel.utm_epsg, fetch_from, end)
            if fetch is not None and fetch_from <= end
            else None
        )
        area_jobs = {
            label: pool.submit(
                _campaign_composites, fetch_composites, parcel, label, start, end, model
            )
            for label in to_compute
        }
        downloads = [job for job in (daily_job, *area_jobs.values()) if job is not None]
        if on_progress:
            on_progress(0, len(downloads))
            for done, _ in enumerate(as_completed(downloads), start=1):
                on_progress(done, len(downloads))
        new = daily_job.result() if daily_job else DailyStats(empty_observations(), 0.0, 0, 0)
        fresh = {label: job.result() for label, job in area_jobs.items()}

    observations = _merge(stored.observations, new.observations, start, end)
    daily = DailyStats(observations, new.processing_units, new.requests, new.failed_intervals)
    areas = {label: _from_stored(record) for label, record in reuse.items()} | fresh
    clean = clean_series(daily.observations)
    campaigns = campaign_metrics(clean)
    neighbors = neighbor_report(parcel, valleys) if valleys is not None else None
    no_value = _CampaignResult(float("nan"), 0.0, failed=False)
    results = [areas.get(label, no_value) for label in campaigns.index]
    campaigns["active_area"] = [result.active_area for result in results]
    campaigns["crop"] = [result.crop for result in results]
    campaigns["crop_confidence"] = [result.crop_confidence for result in results]
    computed = {
        label: StoredCampaign(
            result.active_area,
            result.crop,
            result.crop_confidence,
            min(campaign_bounds(label)[1], end),
            model_version if model is not None else None,
        )
        for label, result in fresh.items()
        if not result.failed
    }
    return ParcelHistory(
        parcel=parcel,
        daily=daily,
        clean=clean,
        campaigns=campaigns,
        credits=sum(area.credits for area in fresh.values()),
        active_area_failures=sum(area.failed for area in fresh.values()),
        flags=sorted(
            [
                *campaign_flags(campaigns),
                *underperforming_flags(neighbors.bands if neighbors else []),
            ],
            key=lambda flag: flag.campaign,
        ),
        neighbors=neighbors,
        campaign_results=reuse | computed,
    )


def _reusable(
    record: StoredCampaign,
    label: str,
    end: date,
    model: CropModel | None,
    model_version: str | None,
) -> bool:
    """A stored campaign covers the period (whole, or up to `end`) with today's crop model."""
    covers = record.computed_through >= min(campaign_bounds(label)[1], end)
    return covers and (model is None or record.crop_model == model_version)


def _from_stored(record: StoredCampaign) -> "_CampaignResult":
    return _CampaignResult(record.active_area, 0.0, False, record.crop, record.crop_confidence)


def _merge(old: pd.DataFrame, new: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """The stored and the new daily statistics, one row per date, within [start, end]."""
    frames = [frame for frame in (old, new) if len(frame)]
    if not frames:
        return empty_observations()
    merged = pd.concat(frames) if len(frames) > 1 else frames[0]
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return merged[(merged.index >= pd.Timestamp(start)) & (merged.index <= pd.Timestamp(end))]


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
    crop: str | None = None  # the predicted class
    crop_confidence: float = float("nan")


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
        return _CampaignResult(
            area,
            composites.credits,
            failed=False,
            crop=crop.crop_class if crop else None,
            crop_confidence=crop.confidence if crop else float("nan"),
        )
    return _CampaignResult(float("nan"), 0.0, failed=True)


def _crop(
    model: CropModel, composites: MonthlyComposites, parcel: Parcel, label: str
) -> CropPrediction | None:
    try:
        return predict_crop(model, interior_stack(composites, parcel.geometry_utm, label))
    except FeatureError:  # e.g. a band missing from the composites: no prediction
        return None
