"""Copernicus Data Space access.

Adapter to the Copernicus Data Space Ecosystem: authentication settings (`cdse`), evalscripts,
daily parcel statistics from the Statistical API (`statistical`), and monthly cloud-free
composites from openEO (`composites`), and the monthly free quota (`quota`, FR-25).
Weeks 2-3, 5.
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
from fundoscope.sentinel.quota import (
    BATCH_QUOTA_SHARE,
    FREE_CREDITS_PER_MONTH,
    FREE_PROCESSING_UNITS_PER_MONTH,
    CreditBudget,
    batch_cap,
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
    "BATCH_QUOTA_SHARE",
    "COMPOSITE_BANDS",
    "CreditBudget",
    "DailyStats",
    "FREE_CREDITS_PER_MONTH",
    "FREE_PROCESSING_UNITS_PER_MONTH",
    "MonthlyComposites",
    "SentinelError",
    "batch_cap",
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
