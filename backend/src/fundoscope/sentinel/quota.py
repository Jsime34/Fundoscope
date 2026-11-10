"""The monthly free quota of Copernicus Data Space, and a credit budget for batch work (FR-25).

The free tier gives 10,000 openEO credits and 10,000 Sentinel Hub processing units per month,
reset on the 1st. Batch work (the training-data extraction, the valley refresh) may use at
most 80 % of the month's credits, so parcel reports and the demo always keep a reserve. The
quota is shared by everything that uses the account, so the caller passes what was already
used this month (shown on the Copernicus Data Space dashboard).
"""

FREE_CREDITS_PER_MONTH = 10_000
FREE_PROCESSING_UNITS_PER_MONTH = 10_000
BATCH_QUOTA_SHARE = 0.8
# FR-23: one campaign of a 3 ha parcel costs 10 credits or less (about 7 measured).
DEFAULT_REQUEST_CREDITS = 10.0


def batch_cap(used_this_month: float) -> float:
    """openEO credits a batch may still spend: 80 % of the month's quota minus what was used.

    Raises:
        ValueError: if `used_this_month` is negative.
    """
    if used_this_month < 0:
        raise ValueError(f"Credits used this month can't be negative (got {used_this_month}).")
    return max(0.0, BATCH_QUOTA_SHARE * FREE_CREDITS_PER_MONTH - used_this_month)


class CreditBudget:
    """openEO credits a batch may spend, counting requests that are still running.

    Before each request, `reserve` holds its expected cost: the most expensive request so far,
    and at least `DEFAULT_REQUEST_CREDITS`. When it finishes, `settle` replaces the hold with
    the credits actually spent. A request is only started if it fits under the cap even at
    the expected cost, so the batch never passes the cap by more than one request's surprise.
    """

    def __init__(self, cap: float, expected: float = DEFAULT_REQUEST_CREDITS) -> None:
        self.cap = cap
        self.expected = expected
        self.spent = 0.0
        self.held = 0.0

    def reserve(self) -> float | None:
        """Hold the expected cost of one more request; None if it would pass the cap."""
        if self.spent + self.held + self.expected > self.cap:
            return None
        self.held += self.expected
        return self.expected

    def settle(self, hold: float, credits: float) -> None:
        """Release a hold and record the credits the request actually used."""
        self.held -= hold
        self.spent += credits
        self.expected = max(self.expected, credits)
