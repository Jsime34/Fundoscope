"""Flags on a parcel's campaign table: abandonment, crop change, anomaly, low data.

Each rule compares a campaign with the parcel's own history, so a parcel is never judged
against an outside standard (FR-07, FR-27, FR-09). Every flag carries the values behind it,
so the report can show why it was raised.

Input: the campaign table of `pipeline.parcel_history`, one row per campaign (index "2023/24"),
with `complete`, `active_area` (0..1 or NaN), `crop` (model class or None), `crop_confidence`
(0..1 or NaN), `integrated_ndvi` (NDVI x days), and `n_obs` (clear dates).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fundoscope.ml import TREE_CROPS

ABANDONMENT = "possible abandonment"
CROP_CHANGE = "crop change"
PERENNIAL_REMOVED = "perennial removed"
PERENNIAL_ESTABLISHED = "perennial established"
PERENNIAL_REPLACED = "perennial replaced"  # one perennial removed, another established
ANOMALY = "anomalous campaign"
LOW_DATA = "low data"

# FR-07: below half of the parcel's median active area, with at least 2 earlier campaigns.
ABANDONMENT_SHARE = 0.5
MIN_EARLIER_CAMPAIGNS = 2
# FR-27: both predictions at least this sure.
MIN_CONFIDENCE = 0.6
PERENNIALS = frozenset(
    {"grape", "asparagus", "avocado", "blueberry", "citrus", "mango", TREE_CROPS}
)
# FR-09: more than 2.5 robust standard deviations, with at least 4 complete campaigns.
MAX_ROBUST_Z = 2.5
MIN_COMPLETE_CAMPAIGNS = 4
MAD_TO_SD = 1.4826  # MAD x 1.4826 estimates the standard deviation of normal data
# When the other campaigns are almost identical, their MAD is near 0 and any small difference
# would look extreme; the spread is never taken below 5 % of their median.
MIN_SPREAD_SHARE = 0.05
MIN_CLEAR_DATES = 15


@dataclass(frozen=True)
class Flag:
    """Something in one campaign that a buyer or lender should ask about.

    Attributes:
        campaign: e.g. "2023/24".
        kind: one of the constants of this module, e.g. `ABANDONMENT`.
        message: one plain-language sentence for the report.
        evidence: the values behind the flag, for the report's "why?" detail.
    """

    campaign: str
    kind: str
    message: str
    evidence: dict[str, float | int | str] = field(default_factory=dict)


def campaign_flags(campaigns: pd.DataFrame) -> list[Flag]:
    """Every flag of a parcel's campaign table, in campaign order."""
    flags = [
        *abandonment_flags(campaigns),
        *crop_change_flags(campaigns),
        *anomaly_flags(campaigns),
        *low_data_flags(campaigns),
    ]
    return sorted(flags, key=lambda flag: flag.campaign)


def abandonment_flags(campaigns: pd.DataFrame) -> list[Flag]:
    """FR-07: complete campaigns whose active area fell below half of the earlier median."""
    flags = []
    earlier: list[float] = []
    for label, row in campaigns.sort_index().iterrows():
        if not row["complete"] or np.isnan(row["active_area"]):
            continue
        area = float(row["active_area"])
        if len(earlier) >= MIN_EARLIER_CAMPAIGNS:
            median = float(np.median(earlier))
            threshold = ABANDONMENT_SHARE * median
            if area < threshold:
                flags.append(
                    Flag(
                        campaign=str(label),
                        kind=ABANDONMENT,
                        message=(
                            f"Possible abandonment: only {area:.0%} of the parcel was farmed, "
                            f"less than half of its usual {median:.0%}."
                        ),
                        evidence={
                            "active_area": round(area, 3),
                            "median_before": round(median, 3),
                            "threshold": round(threshold, 3),
                            "clear_dates": int(row["n_obs"]),
                        },
                    )
                )
        earlier.append(area)
    return flags


def crop_change_flags(campaigns: pd.DataFrame) -> list[Flag]:
    """FR-27: the predicted crop differs from the previous campaign's, both predictions sure."""
    flags = []
    rows = list(campaigns.sort_index().iterrows())
    for (before_label, before), (label, now) in zip(rows, rows[1:], strict=False):
        old, new = before["crop"], now["crop"]
        if not (isinstance(old, str) and isinstance(new, str)) or old == new:
            continue
        old_sure, new_sure = float(before["crop_confidence"]), float(now["crop_confidence"])
        if min(old_sure, new_sure) < MIN_CONFIDENCE:
            continue
        kind = _change_kind(old, new)
        flags.append(
            Flag(
                campaign=str(label),
                kind=kind,
                message=(
                    f"{kind.capitalize()}: {old} in {before_label} ({old_sure:.0%} sure), "
                    f"{new} in {label} ({new_sure:.0%} sure)."
                ),
                evidence={
                    "previous_campaign": str(before_label),
                    "previous_crop": old,
                    "previous_confidence": round(old_sure, 3),
                    "crop": new,
                    "confidence": round(new_sure, 3),
                },
            )
        )
    return flags


def anomaly_flags(campaigns: pd.DataFrame) -> list[Flag]:
    """FR-09: complete campaigns whose integrated NDVI is far from the other complete ones."""
    is_complete = campaigns["complete"].astype(bool)  # an empty table has object columns
    complete = campaigns.loc[is_complete, "integrated_ndvi"].dropna().sort_index()
    if len(complete) < MIN_COMPLETE_CAMPAIGNS:
        return []
    flags = []
    for label, value in complete.items():
        others = complete.drop(label).to_numpy()
        median = float(np.median(others))
        mad = float(np.median(np.abs(others - median)))
        spread = max(MAD_TO_SD * mad, MIN_SPREAD_SHARE * abs(median))
        z = (value - median) / spread
        if abs(z) <= MAX_ROBUST_Z:
            continue
        low, high = median - MAX_ROBUST_Z * spread, median + MAX_ROBUST_Z * spread
        direction = "below" if z < 0 else "above"
        flags.append(
            Flag(
                campaign=str(label),
                kind=ANOMALY,
                message=(
                    f"Unusual campaign: integrated NDVI {value:.0f}, well {direction} the "
                    f"parcel's usual {median:.0f} (normal range {low:.0f} to {high:.0f})."
                ),
                evidence={
                    "integrated_ndvi": round(float(value), 1),
                    "median_others": round(median, 1),
                    "normal_low": round(low, 1),
                    "normal_high": round(high, 1),
                    "robust_z": round(float(z), 2),
                },
            )
        )
    return flags


def low_data_flags(campaigns: pd.DataFrame) -> list[Flag]:
    """FR-09: campaigns with fewer than 15 clear dates; their values are less reliable."""
    return [
        Flag(
            campaign=str(label),
            kind=LOW_DATA,
            message=f"Low data: only {int(n)} clear dates, so this campaign's values are less "
            "reliable.",
            evidence={"clear_dates": int(n), "minimum": MIN_CLEAR_DATES},
        )
        for label, n in campaigns["n_obs"].sort_index().items()
        if n < MIN_CLEAR_DATES
    ]


def _change_kind(old: str, new: str) -> str:
    if old in PERENNIALS and new in PERENNIALS:
        return PERENNIAL_REPLACED
    if old in PERENNIALS:
        return PERENNIAL_REMOVED
    if new in PERENNIALS:
        return PERENNIAL_ESTABLISHED
    return CROP_CHANGE
