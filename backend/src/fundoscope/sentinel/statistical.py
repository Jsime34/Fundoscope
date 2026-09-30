"""Daily parcel statistics from the Sentinel Hub Statistical API (FR-03).

For every Sentinel-2 L2A acquisition over a geometry, the API computes NDVI and NDWI on the
clear pixels (clouds, shadows, and cirrus are masked by the evalscript) and returns their
mean and spread plus how many pixels were clear. The result is a DataFrame in the format
`fundoscope.timeseries.clean_series` expects.

Requests cover at most one year each. sentinelhub-py already retries when the service is
busy (HTTP 429) or fails temporarily, so there is no retry loop here. Every response reports
the processing units (PU) it cost; they are summed so callers can track the monthly quota.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd
from sentinelhub import CRS, Geometry, SentinelHubStatistical, SHConfig
from shapely.geometry.base import BaseGeometry

from fundoscope.sentinel.cdse import s2_l2a, sh_config
from fundoscope.sentinel.evalscripts import DAILY_INDICES

ARCHIVE_START = date(2018, 8, 1)  # first day of campaign 2018/19
MAX_CHUNK_DAYS = 365
RESOLUTION_M = 10
PU_HEADER = "x-processingunits-spent"

COLUMNS = ["ndvi", "ndwi", "ndvi_std", "valid_fraction", "pixels"]

# (request) -> (decoded JSON payload, processing units spent). Replaced by a fake in tests.
Sender = Callable[[SentinelHubStatistical], tuple[dict, float]]


class SentinelError(RuntimeError):
    """The Sentinel Hub service returned something we can't use."""


@dataclass(frozen=True)
class DailyStats:
    """Result of `fetch_daily_stats`.

    Attributes:
        observations: one row per acquisition date (DatetimeIndex, UTC dates) with
            ndvi, ndwi (means over clear pixels), ndvi_std, valid_fraction (0..1, clear share
            of the pixels inside the geometry), pixels (pixels inside the geometry).
        processing_units: total PU spent by the requests.
        requests: number of API requests made.
        failed_intervals: dates the service could not process (skipped).
    """

    observations: pd.DataFrame
    processing_units: float
    requests: int
    failed_intervals: int


def fetch_daily_stats(
    geometry_utm: BaseGeometry,
    epsg: int,
    start: date = ARCHIVE_START,
    end: date | None = None,
    *,
    config: SHConfig | None = None,
    send: Sender | None = None,
) -> DailyStats:
    """Download daily NDVI/NDWI statistics for a geometry, from `start` to `end` inclusive.

    Args:
        geometry_utm: parcel (or ring) geometry in UTM meters, e.g. `Parcel.geometry_utm`.
        epsg: its UTM EPSG code (32717 or 32718 in coastal Peru).
        start, end: first and last day to include; `end` defaults to today.
        config: Sentinel Hub configuration; defaults to our CDSE settings.
        send: function that performs a request; only tests replace it.
    """
    end = end or date.today()
    if end < start:
        raise ValueError(f"The end date {end} is before the start date {start}")
    config = config or sh_config()
    send = send or _send
    geometry = Geometry(geometry_utm, CRS(epsg))

    frames, spent, failed, requests = [], 0.0, 0, 0
    for chunk_start, chunk_end in date_chunks(start, end):
        payload, units = send(build_request(geometry, chunk_start, chunk_end, config))
        observations, chunk_failed = parse_response(payload)
        frames.append(observations)
        spent += units
        failed += chunk_failed
        requests += 1

    observations = pd.concat(frames).sort_index()
    observations = observations[~observations.index.duplicated(keep="first")]
    return DailyStats(observations, spent, requests, failed)


def date_chunks(
    start: date, end: date, max_days: int = MAX_CHUNK_DAYS
) -> Iterator[tuple[date, date]]:
    """Split [start, end] (inclusive) into consecutive chunks of at most `max_days` days."""
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=max_days - 1), end)
        yield chunk_start, chunk_end
        chunk_start = chunk_end + timedelta(days=1)


def build_request(
    geometry: Geometry, first_day: date, last_day: date, config: SHConfig
) -> SentinelHubStatistical:
    """One Statistical API request with one aggregation interval per day.

    The time range is whole days (midnight to the midnight after `last_day`), so it divides
    exactly into 1-day intervals. With plain dates sentinelhub-py would end at 23:59:59, and
    the API skips a partial last interval by default.
    """
    time_range = (
        datetime.combine(first_day, datetime.min.time()),
        datetime.combine(last_day + timedelta(days=1), datetime.min.time()),
    )
    return SentinelHubStatistical(
        aggregation=SentinelHubStatistical.aggregation(
            evalscript=DAILY_INDICES,
            time_interval=time_range,
            aggregation_interval="P1D",
            resolution=(RESOLUTION_M, RESOLUTION_M),
        ),
        input_data=[SentinelHubStatistical.input_data(s2_l2a())],
        geometry=geometry,
        config=config,
    )


def parse_response(payload: dict) -> tuple[pd.DataFrame, int]:
    """Turn a Statistical API response into observation rows.

    Returns the rows (one per acquisition date) and the number of failed intervals.
    Dates where every pixel was cloudy are kept with NaN indices and valid_fraction 0.

    Pixel counts: `sampleCount` covers the geometry's whole bounding box, and pixels outside
    the geometry are counted as "no data". So the clear share must be measured against
    `geometryPixelCount` (pixels inside the geometry), not against `sampleCount`; otherwise a
    diagonal parcel filling half its bounding box would never look more than ~50 % clear.
    """
    if payload.get("status") != "OK":
        raise SentinelError(f"Statistical API returned status {payload.get('status')!r}")

    rows, failed = [], 0
    for item in payload.get("data", []):
        if "error" in item:
            failed += 1
            continue
        ndvi = item["outputs"]["ndvi"]["bands"]["B0"]["stats"]
        ndwi = item["outputs"]["ndwi"]["bands"]["B0"]["stats"]
        inside = int(payload.get("geometryPixelCount") or ndvi["sampleCount"])
        clear = int(ndvi["sampleCount"]) - int(ndvi["noDataCount"])
        rows.append(
            {
                "date": pd.Timestamp(item["interval"]["from"]).tz_convert(None),
                "ndvi": float(ndvi["mean"]),  # float() also parses the API's "NaN"
                "ndwi": float(ndwi["mean"]),
                "ndvi_std": float(ndvi["stDev"]),
                "valid_fraction": min(1.0, clear / inside) if inside else 0.0,
                "pixels": inside,
            }
        )

    if payload.get("data") and failed == len(payload["data"]):
        raise SentinelError("The Statistical API could not process any date in the request.")
    frame = pd.DataFrame(rows, columns=["date", *COLUMNS]).set_index("date")
    frame.index = pd.DatetimeIndex(frame.index, name="date")
    return frame, failed


def _send(request: SentinelHubStatistical) -> tuple[dict, float]:
    [response] = request.get_data(decode_data=False)
    return response.decode(), float(response.headers.get(PU_HEADER, 0.0))
