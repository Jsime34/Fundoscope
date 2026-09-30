"""Copernicus Data Space access.

Adapter to the Copernicus Data Space Ecosystem: authentication settings (`cdse`), evalscripts,
and daily parcel statistics from the Statistical API (`statistical`). Monthly composites
(openEO) and quota tracking come next (FR-03). Weeks 2-3, 5.
"""

from fundoscope.sentinel.cdse import s2_l2a, sh_config
from fundoscope.sentinel.statistical import (
    ARCHIVE_START,
    DailyStats,
    SentinelError,
    date_chunks,
    fetch_daily_stats,
    parse_response,
)

__all__ = [
    "ARCHIVE_START",
    "DailyStats",
    "SentinelError",
    "date_chunks",
    "fetch_daily_stats",
    "parse_response",
    "s2_l2a",
    "sh_config",
]
