from datetime import date

import pytest

from fundoscope.valley.season import (
    Count,
    MonthKey,
    group_months,
    month_end,
    next_month,
    one_year_earlier,
    season,
)


def key(month: str, through: str | None = None) -> MonthKey:
    first = date.fromisoformat(f"{month}-01")
    return MonthKey(first, date.fromisoformat(through) if through else month_end(first))


def test_a_refresh_uses_the_images_up_to_two_days_ago():
    when = season(date(2026, 11, 24))

    assert (when.as_of, when.campaign, when.months, when.through) == (
        date(2026, 11, 22),
        "2026/27",
        4,
        date(2026, 11, 22),
    )
    assert when.past == ("2025/26", "2024/25")


def test_the_four_counts_of_a_season_in_progress():
    counts = season(date(2026, 11, 24)).counts

    assert [count.key for count in counts] == [
        "2026-27_to_2026-11-22",
        "2025-26_to_2025-11-22",  # the same date one year earlier
        "2025-26",
        "2024-25",
    ]
    assert [count.label for count in counts] == [
        "2026/27 to Nov 22",
        "2025/26 to Nov 22",
        "2025/26",
        "2024/25",
    ]
    assert [count.months for count in counts] == [4, 4, 12, 12]
    assert [count.in_progress for count in counts] == [True, True, False, False]


def test_a_month_with_less_than_a_week_of_images_waits():
    when = season(date(2026, 12, 4))  # images up to December 2

    assert (when.months, when.through) == (4, date(2026, 11, 30))
    assert [count.key for count in when.counts][:2] == [
        "2026-27_to_2026-11-30",
        "2025-26_to_2025-11-30",
    ]


def test_the_first_week_of_a_campaign_counts_only_the_past():
    when = season(date(2026, 8, 3))  # images up to August 1

    assert (when.campaign, when.months, when.through) == ("2026/27", 0, None)
    assert [count.key for count in when.counts] == ["2025-26", "2024-25"]


def test_a_campaign_just_over_is_counted_whole_once():
    when = season(date(2026, 8, 2))  # images up to July 31

    assert (when.campaign, when.months) == ("2025/26", 12)
    assert [count.key for count in when.counts] == ["2025-26", "2024-25", "2023-24"]


def test_the_months_of_a_count():
    count = Count("2026/27", 4, date(2026, 11, 22))

    assert count.month_keys() == [
        key("2026-08"),
        key("2026-09"),
        key("2026-10"),
        key("2026-11", "2026-11-22"),
    ]


def test_every_month_needed_appears_once_oldest_first():
    keys = season(date(2026, 11, 24)).month_keys()

    assert len(keys) == 24 + 4 + 1  # two whole campaigns, this one so far, one cut month
    assert keys == sorted(keys)
    assert key("2025-11", "2025-11-22") in keys and key("2025-11") in keys


def test_a_month_key_names_its_file():
    assert (key("2025-11").complete, key("2025-11").file_name) == (True, "2025-11.npz")
    cut = key("2025-11", "2025-11-22")
    assert (cut.complete, cut.file_name) == (False, "2025-11-22.npz")


def test_requests_pack_up_to_six_consecutive_months():
    months = [key(f"2024-{m:02d}") for m in range(1, 13)] + [key("2025-01")]

    groups = group_months(months)

    assert [len(group) for group in groups] == [6, 6, 1]


def test_a_month_cut_short_ends_its_request():
    groups = group_months([key("2025-10"), key("2025-11", "2025-11-22"), key("2025-12")])

    assert groups == [[key("2025-10"), key("2025-11", "2025-11-22")], [key("2025-12")]]


def test_a_gap_starts_a_new_request():
    groups = group_months([key("2025-01"), key("2025-03"), key("2025-04")])

    assert [[k.month.month for k in group] for group in groups] == [[1], [3, 4]]


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (date(2028, 2, 29), date(2027, 2, 28)),
        (date(2026, 11, 22), date(2025, 11, 22)),
        (date(2026, 2, 10), date(2025, 2, 10)),
    ],
)
def test_one_year_earlier(day, expected):
    assert one_year_earlier(day) == expected


def test_month_helpers():
    assert month_end(date(2028, 2, 3)) == date(2028, 2, 29)
    assert next_month(date(2026, 12, 15)) == date(2027, 1, 1)
