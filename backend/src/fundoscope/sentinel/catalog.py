"""The date of the newest Sentinel-2 image over an area, from the CDSE catalog (FR-15).

The valley dashboard shows the date of the newest image its count used. Monthly composites
don't say which images went into them, so the date comes from the Sentinel Hub Catalog API
(STAC) of Copernicus Data Space: the newest Sentinel-2 L2A acquisition over the area up to a
given day. Catalog searches are free (no processing units).
"""

from collections.abc import Callable, Iterable
from datetime import date, datetime, timedelta

from requests import RequestException
from sentinelhub import CRS, BBox, SentinelHubCatalog
from sentinelhub.exceptions import BaseSentinelHubException

from fundoscope.sentinel.cdse import s2_l2a, sh_config
from fundoscope.sentinel.statistical import SentinelError

SEARCH_DAYS = 31  # Sentinel-2 passes every 5 days or so: a month is plenty

# (bbox in WGS84, first day, last day) -> STAC items. Replaced by a fake in tests.
Searcher = Callable[[tuple[float, float, float, float], date, date], Iterable[dict]]


def newest_image_date(
    bounds_wgs84: tuple[float, float, float, float],
    until: date,
    *,
    search: Searcher | None = None,
) -> date | None:
    """The day of the newest Sentinel-2 L2A image over the area up to `until` (included).

    Returns None when no image was taken in the 31 days before `until`.

    Raises:
        SentinelError: the catalog couldn't be searched.
    """
    start = until - timedelta(days=SEARCH_DAYS)
    try:
        items = list((search or _search)(bounds_wgs84, start, until))
    except (BaseSentinelHubException, RequestException) as exc:
        raise SentinelError(f"Could not search the Copernicus catalog: {exc}") from exc
    days = [datetime.fromisoformat(item["properties"]["datetime"]).date() for item in items]
    days = [day for day in days if day <= until]
    return max(days, default=None)


def _search(
    bounds_wgs84: tuple[float, float, float, float], start: date, end: date
) -> Iterable[dict]:
    catalog = SentinelHubCatalog(config=sh_config())
    return catalog.search(
        s2_l2a(),
        bbox=BBox(bounds_wgs84, crs=CRS.WGS84),
        time=(start.isoformat(), end.isoformat()),
        fields={"include": ["properties.datetime"], "exclude": []},
    )
