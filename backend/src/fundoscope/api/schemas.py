"""JSON shapes returned by the API (what the front end receives)."""

import math
from datetime import date

from pydantic import BaseModel
from shapely.geometry import mapping

from fundoscope.ingest import zone_label
from fundoscope.pipeline import ParcelHistory


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


class SeriesPoint(BaseModel):
    date: date
    ndvi: float | None
    ndwi: float | None


class ObservationPoint(BaseModel):
    date: date
    ndvi: float


class HistoryOut(BaseModel):
    parcel: ParcelOut
    summary: SummaryOut
    campaigns: list[CampaignOut]
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
            )
            for label, row in history.campaigns.iterrows()
        ],
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


def _num(value: float) -> float | None:
    value = float(value)
    return None if math.isnan(value) else round(value, 4)
