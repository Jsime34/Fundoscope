"""Peruvian agricultural campaigns: August 1 to July 31 (the MIDAGRI calendar).

A date belongs to the campaign that started on the most recent August 1. Labels are written
"start year / last two digits of the end year": 2023-08-01 .. 2024-07-31 is "2023/24".
"""

import re
from datetime import date, datetime

import pandas as pd

CAMPAIGN_START_MONTH = 8

DateLike = date | datetime | pd.Timestamp | str

_LABEL = re.compile(r"^(\d{4})/(\d{2})$")


def campaign_start_year(when: DateLike) -> int:
    """Year in which the campaign containing `when` started (2024-03-10 -> 2023)."""
    ts = pd.Timestamp(when)
    return ts.year if ts.month >= CAMPAIGN_START_MONTH else ts.year - 1


def campaign_label(start_year: int) -> str:
    """Label of the campaign starting on August 1 of `start_year` (2023 -> "2023/24")."""
    return f"{start_year}/{(start_year + 1) % 100:02d}"


def campaign_of(when: DateLike) -> str:
    """Campaign label for a date: 2023-08-01 -> "2023/24", 2024-07-31 -> "2023/24"."""
    return campaign_label(campaign_start_year(when))


def parse_campaign(label: str) -> int:
    """Start year of a campaign label ("2023/24" -> 2023).

    Raises:
        ValueError: if the label is not of the form "YYYY/YY" with consecutive years.
    """
    match = _LABEL.match(label)
    if not match:
        raise ValueError(f"Invalid campaign label {label!r}; expected e.g. '2023/24'")
    start = int(match.group(1))
    if int(match.group(2)) != (start + 1) % 100:
        raise ValueError(f"Invalid campaign label {label!r}; the years must be consecutive")
    return start


def campaign_bounds(label: str) -> tuple[date, date]:
    """First and last day of a campaign: "2023/24" -> (2023-08-01, 2024-07-31)."""
    start = parse_campaign(label)
    return date(start, CAMPAIGN_START_MONTH, 1), date(start + 1, CAMPAIGN_START_MONTH - 1, 31)


def campaign_labels(index: pd.DatetimeIndex) -> pd.Index:
    """Campaign label for every date of a DatetimeIndex (vectorized `campaign_of`)."""
    start_years = index.year - (index.month < CAMPAIGN_START_MONTH).astype(int)
    return pd.Index([campaign_label(y) for y in start_years], name="campaign")


def campaigns_between(start: DateLike, end: DateLike) -> list[str]:
    """All campaigns that overlap the period [start, end], oldest first."""
    first, last = campaign_start_year(start), campaign_start_year(end)
    if last < first:
        raise ValueError(f"The end {end} is before the start {start}")
    return [campaign_label(year) for year in range(first, last + 1)]
