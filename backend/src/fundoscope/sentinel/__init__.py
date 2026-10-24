"""Copernicus Data Space access.

Adapter to the Copernicus Data Space Ecosystem: authentication settings (`cdse`), evalscripts,
daily parcel statistics from the Statistical API (`statistical`), and monthly cloud-free
composites from openEO (`composites`). Quota tracking comes next (FR-03). Weeks 2-3, 5.
"""

from fundoscope.sentinel.cdse import s2_l2a, sh_config
from fundoscope.sentinel.composites import (
    COMPOSITE_BANDS,
    MonthlyComposites,
    build_graph,
    fetch_monthly_composites,
    openeo_connection,
    parse_netcdf,
)
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
    "COMPOSITE_BANDS",
    "DailyStats",
    "MonthlyComposites",
    "SentinelError",
    "build_graph",
    "date_chunks",
    "fetch_daily_stats",
    "fetch_monthly_composites",
    "openeo_connection",
    "parse_netcdf",
    "parse_response",
    "s2_l2a",
    "sh_config",
]
