from datetime import date, datetime

import pandas as pd
import pytest

from fundoscope.timeseries import (
    campaign_bounds,
    campaign_labels,
    campaign_of,
    campaigns_between,
    parse_campaign,
)


@pytest.mark.parametrize(
    ("when", "expected"),
    [
        ("2023-08-01", "2023/24"),  # first day of a campaign
        ("2024-07-31", "2023/24"),  # last day of the same campaign
        ("2024-08-01", "2024/25"),
        ("2024-01-15", "2023/24"),
        ("1999-09-01", "1999/00"),  # century rollover
        (date(2020, 12, 25), "2020/21"),
        (datetime(2021, 7, 31, 23, 59), "2020/21"),
        (pd.Timestamp("2018-08-03"), "2018/19"),
    ],
)
def test_campaign_of(when, expected):
    assert campaign_of(when) == expected


def test_campaign_bounds():
    assert campaign_bounds("2023/24") == (date(2023, 8, 1), date(2024, 7, 31))
    assert campaign_bounds("1999/00") == (date(1999, 8, 1), date(2000, 7, 31))


@pytest.mark.parametrize("label", ["2023-24", "2023/25", "23/24", "", "2023/2024"])
def test_parse_campaign_rejects_bad_labels(label):
    with pytest.raises(ValueError):
        parse_campaign(label)


def test_vectorized_labels_match_single_dates():
    days = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    labels = campaign_labels(days)
    assert list(labels) == [campaign_of(d) for d in days]
    assert labels.name == "campaign"


def test_campaigns_between_lists_every_overlapping_campaign():
    # From the start of our archive to today: 8 complete campaigns plus the current one.
    campaigns = campaigns_between("2018-08-01", "2026-09-30")
    assert campaigns[0] == "2018/19"
    assert campaigns[-1] == "2026/27"
    assert len(campaigns) == 9


def test_campaigns_between_rejects_reversed_period():
    with pytest.raises(ValueError):
        campaigns_between("2024-01-01", "2020-01-01")
