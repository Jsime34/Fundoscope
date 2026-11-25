"""What a valley refresh counts, given today's date (FR-14).

A refresh uses the images up to `as_of`, two days before today, because the newest images may
not be processed yet. The current campaign counts its months from August: each finished month,
and the month in progress once it has 7 days of images (with fewer, the count stops at the end
of the previous month). It makes four counts:

- the current campaign to date, with the season-to-date model of its months;
- the previous campaign at the same date: the same months, the last one cut at the same day
  one year earlier, with the same model, so the comparison is like for like;
- the previous campaign and the one before it, whole (12 months), with the full model.

Each count is a list of months, and each month of a tile is downloaded once and kept: a
`MonthKey` is a month and the last day of images in it (the month's last day once it is
finished). A refresh downloads only the keys it lacks, packing up to six consecutive months
into one openEO request, because a request costs about the same whatever its length.
"""

import calendar
from dataclasses import dataclass
from datetime import date, timedelta

from fundoscope.ml import MONTHS
from fundoscope.timeseries.campaigns import (
    campaign_bounds,
    campaign_label,
    campaign_of,
    parse_campaign,
)

IMAGE_LAG_DAYS = 2
MIN_DAYS_IN_MONTH = 7
MONTHS_PER_REQUEST = 6
PAST_CAMPAIGNS = 2


def month_end(day: date) -> date:
    """The last day of the month of `day`."""
    return day.replace(day=calendar.monthrange(day.year, day.month)[1])


def next_month(day: date) -> date:
    """The first day of the month after the month of `day`."""
    return month_end(day) + timedelta(days=1)


@dataclass(frozen=True, order=True)
class MonthKey:
    """One month of composites, with images up to `through`.

    Attributes:
        month: the first day of the month.
        through: the last day of images: the month's last day once it is finished.
    """

    month: date
    through: date

    @property
    def complete(self) -> bool:
        return self.through == month_end(self.month)

    @property
    def file_name(self) -> str:
        """ "2025-11.npz" for a finished month, "2025-11-22.npz" for one cut on the 22nd."""
        return f"{self.month:%Y-%m}.npz" if self.complete else f"{self.through:%Y-%m-%d}.npz"


@dataclass(frozen=True)
class Count:
    """One crop count: the first `months` months of a campaign, up to `through`.

    Attributes:
        campaign: "2026/27".
        months: 1 to 12, August first.
        through: the last day of images used.
    """

    campaign: str
    months: int
    through: date

    @property
    def in_progress(self) -> bool:
        """Counted before the campaign ended: the season-to-date model and its rule apply."""
        return self.through < campaign_bounds(self.campaign)[1]

    @property
    def key(self) -> str:
        """A name for files: "2025-26" when whole, "2026-27_to_2026-11-22" otherwise."""
        name = self.campaign.replace("/", "-")
        return f"{name}_to_{self.through:%Y-%m-%d}" if self.in_progress else name

    @property
    def label(self) -> str:
        """For people: "2025/26" when whole, "2026/27 to Nov 22" otherwise."""
        if not self.in_progress:
            return self.campaign
        return f"{self.campaign} to {self.through:%b} {self.through.day}"

    def month_keys(self) -> list[MonthKey]:
        """The months of the count, August first; the last one ends on `through`."""
        first, _ = campaign_bounds(self.campaign)
        months = [first]
        while len(months) < self.months:
            months.append(next_month(months[-1]))
        return [MonthKey(month, min(month_end(month), self.through)) for month in months]


@dataclass(frozen=True)
class Season:
    """What a refresh on a given day counts.

    Attributes:
        as_of: the newest day of images used.
        campaign: the current campaign.
        months: months of the current campaign counted so far (0 in its first week).
        through: the last day counted in the current campaign (None when `months` is 0).
    """

    as_of: date
    campaign: str
    months: int
    through: date | None

    @property
    def past(self) -> tuple[str, ...]:
        """The campaigns before the current one, most recent first."""
        start = parse_campaign(self.campaign)
        return tuple(campaign_label(start - n) for n in range(1, PAST_CAMPAIGNS + 1))

    @property
    def counts(self) -> tuple[Count, ...]:
        """The counts of the refresh: current, previous at the same date, then the past ones.

        The first two are missing in the first week of a campaign. A count that repeats an
        earlier one (the current campaign once it is over) appears once.
        """
        counts = []
        if self.through is not None:
            counts.append(Count(self.campaign, self.months, self.through))
            counts.append(Count(self.past[0], self.months, one_year_earlier(self.through)))
        counts += [
            Count(campaign, len(MONTHS), campaign_bounds(campaign)[1]) for campaign in self.past
        ]
        unique: dict[str, Count] = {}
        for count in counts:
            unique.setdefault(count.key, count)
        return tuple(unique.values())

    def month_keys(self) -> list[MonthKey]:
        """Every month that some count needs, oldest first."""
        return sorted({key for count in self.counts for key in count.month_keys()})


def season(today: date) -> Season:
    """What a refresh on `today` counts."""
    as_of = today - timedelta(days=IMAGE_LAG_DAYS)
    campaign = campaign_of(as_of)
    first, _ = campaign_bounds(campaign)
    finished = (as_of.year - first.year) * 12 + as_of.month - first.month
    if as_of.day >= MIN_DAYS_IN_MONTH:
        return Season(as_of, campaign, finished + 1, as_of)
    through = as_of.replace(day=1) - timedelta(days=1) if finished else None
    return Season(as_of, campaign, finished, through)


def one_year_earlier(day: date) -> date:
    """The same day one year earlier; February 29 becomes February 28."""
    return day.replace(year=day.year - 1, day=min(day.day, 28) if day.month == 2 else day.day)


def group_months(keys: list[MonthKey], size: int = MONTHS_PER_REQUEST) -> list[list[MonthKey]]:
    """Months to download, packed into requests of up to `size` consecutive months.

    A request covers a period, so a month cut short can only be the last of its request.
    """
    groups: list[list[MonthKey]] = []
    for key in sorted(keys):
        last = groups[-1][-1] if groups else None
        if (
            last is None
            or len(groups[-1]) >= size
            or not last.complete
            or key.month != next_month(last.month)
        ):
            groups.append([key])
        else:
            groups[-1].append(key)
    return groups
