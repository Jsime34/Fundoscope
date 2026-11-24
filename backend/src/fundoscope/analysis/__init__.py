"""Parcel analysis.

Abandonment/crop-change/anomaly flags, neighbor ring comparison, ANA water check, report
assembly (FR-07, FR-27, FR-08, FR-09, FR-10).

Submodules: `flags` (flags on a parcel's campaign table).
"""

from fundoscope.analysis.flags import (
    ABANDONMENT,
    ANOMALY,
    CROP_CHANGE,
    LOW_DATA,
    PERENNIAL_ESTABLISHED,
    PERENNIAL_REMOVED,
    PERENNIAL_REPLACED,
    PERENNIALS,
    Flag,
    abandonment_flags,
    anomaly_flags,
    campaign_flags,
    crop_change_flags,
    low_data_flags,
)

__all__ = [
    "ABANDONMENT",
    "ANOMALY",
    "CROP_CHANGE",
    "LOW_DATA",
    "PERENNIALS",
    "PERENNIAL_ESTABLISHED",
    "PERENNIAL_REMOVED",
    "PERENNIAL_REPLACED",
    "Flag",
    "abandonment_flags",
    "anomaly_flags",
    "campaign_flags",
    "crop_change_flags",
    "low_data_flags",
]
