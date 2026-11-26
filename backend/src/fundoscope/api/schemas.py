"""JSON shapes returned by the API (what the front end receives)."""

import math
from datetime import date

from pydantic import BaseModel
from shapely.geometry import mapping

from fundoscope.ingest import zone_label
from fundoscope.pipeline import ParcelHistory
from fundoscope.pipeline.neighbors import NeighborReport


class ParcelOut(BaseModel):
    name: str | None
    area_ha: float
    utm_epsg: int
    zone: str
    repaired: bool
    geometry: dict  # GeoJSON MultiPolygon, WGS84 lon/lat
    bounds: tuple[float, float, float, float]  # west, south, east, north


class SummaryOut(BaseModel):
    satellite_dates: int
    first_date: date | None
    last_date: date | None
    dropped_cloudy: int
    dropped_dips: int
    processing_units: float
    credits: float
    active_area_failures: int
    cached: bool = False


class CampaignOut(BaseModel):
    campaign: str
    peak_ndvi: float | None
    mean_ndvi: float | None
    mean_ndwi: float | None
    integrated_ndvi: float | None
    n_obs: int
    coverage: float
    complete: bool
    active_area: float | None
    crop: str | None = None  # predicted model class; None for a campaign in progress
    crop_confidence: float | None = None  # 0..1


class FlagOut(BaseModel):
    campaign: str
    kind: str  # "possible abandonment", "crop change", "perennial removed", ...
    message: str  # one plain-language sentence
    evidence: dict[str, float | int | str]  # the values behind the flag


class SeriesPoint(BaseModel):
    date: date
    ndvi: float | None
    ndwi: float | None


class ObservationPoint(BaseModel):
    date: date
    ndvi: float


class NeighborBandOut(BaseModel):
    campaign: str
    months: int  # months compared: 12 once the campaign is over
    parcel_ndvi: float  # mean NDVI of the parcel's pixels
    parcel_pixels: int
    neighbor_pixels: int
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    band: str  # "below p10", "p10-p25", ..., "above p90"
    underperforming: bool


class NeighborsOut(BaseModel):
    valley: str | None  # None: the parcel is outside every counted valley
    note: str  # why there are no bands, if there are none
    ring: dict | None  # GeoJSON of the 50 m-2 km ring, WGS84 lon/lat
    bands: list[NeighborBandOut]


class HistoryOut(BaseModel):
    parcel: ParcelOut
    summary: SummaryOut
    campaigns: list[CampaignOut]
    flags: list[FlagOut] = []
    neighbors: NeighborsOut | None = None
    series: list[SeriesPoint]  # cleaned and smoothed, every 5 days
    observations: list[ObservationPoint]  # clear satellite dates actually used


def history_out(history: ParcelHistory) -> HistoryOut:
    """Convert a pipeline result into the API's JSON shape (NaN becomes null)."""
    parcel, daily, clean = history.parcel, history.daily, history.clean
    obs = daily.observations
    return HistoryOut(
        parcel=ParcelOut(
            name=parcel.name,
            area_ha=round(parcel.area_ha, 2),
            utm_epsg=parcel.utm_epsg,
            zone=zone_label(parcel.utm_epsg),
            repaired=parcel.repaired,
            geometry=mapping(parcel.geometry),
            bounds=parcel.geometry.bounds,
        ),
        summary=SummaryOut(
            satellite_dates=len(obs),
            first_date=obs.index[0].date() if len(obs) else None,
            last_date=obs.index[-1].date() if len(obs) else None,
            dropped_cloudy=clean.dropped_cloudy,
            dropped_dips=clean.dropped_dips,
            processing_units=round(daily.processing_units, 2),
            credits=round(history.credits, 2),
            active_area_failures=history.active_area_failures,
        ),
        campaigns=[
            CampaignOut(
                campaign=label,
                peak_ndvi=_num(row["peak_ndvi"]),
                mean_ndvi=_num(row["mean_ndvi"]),
                mean_ndwi=_num(row["mean_ndwi"]),
                integrated_ndvi=_num(row["integrated_ndvi"]),
                n_obs=int(row["n_obs"]),
                coverage=float(row["coverage"]),
                complete=bool(row["complete"]),
                active_area=_num(row["active_area"]),
                crop=row["crop"] if isinstance(row["crop"], str) else None,
                crop_confidence=_num(row["crop_confidence"]),
            )
            for label, row in history.campaigns.iterrows()
        ],
        flags=[
            FlagOut(
                campaign=flag.campaign,
                kind=flag.kind,
                message=flag.message,
                evidence=flag.evidence,
            )
            for flag in history.flags
        ],
        neighbors=neighbors_out(history.neighbors),
        series=[
            SeriesPoint(
                date=day.date(), ndvi=_num(row["ndvi_smooth"]), ndwi=_num(row["ndwi_smooth"])
            )
            for day, row in clean.grid.iterrows()
        ],
        observations=[
            ObservationPoint(date=day.date(), ndvi=round(float(ndvi), 4))
            for day, ndvi in clean.observations["ndvi"].items()
        ],
    )


def neighbors_out(report: NeighborReport | None) -> NeighborsOut | None:
    """The neighbor comparison in the API's shape (None when it wasn't computed)."""
    if report is None:
        return None
    return NeighborsOut(
        valley=report.valley,
        note=report.note,
        ring=mapping(report.ring) if report.ring is not None else None,
        bands=[
            NeighborBandOut(
                campaign=band.campaign,
                months=band.months,
                parcel_ndvi=round(band.parcel_ndvi, 4),
                parcel_pixels=band.parcel_pixels,
                neighbor_pixels=band.neighbor_pixels,
                **{f"p{p}": round(value, 4) for p, value in band.percentiles.items()},
                band=band.band,
                underperforming=band.underperforming,
            )
            for band in report.bands
        ],
    )


def _num(value: float) -> float | None:
    value = float(value)
    return None if math.isnan(value) else round(value, 4)
