"""Compare every label with its NDVI curve before training (FR-06).

A Street View photo shows the crop on one day, but the model learns the dominant crop of a
whole campaign (August to July). This check lists the labeled campaigns whose NDVI curve
doesn't fit their crop class, so each one is reviewed by hand before training: corrected,
deleted, or kept with a note "NDVI ok: <reason>" in the label's `notes`.

The curve is the median NDVI of the parcel's training pixels for each month. The rules use
the active threshold of FR-23 (NDVI 0.4):

- fallow/bare: no month above 0.4, because a fallow campaign has no crop at all;
- every crop: at least one month above 0.4;
- trees (avocado, citrus, mango) and alfalfa: at most 2 months in a row near bare soil
  (NDVI below 0.25); grape: at most 4, because pruned vines stay bare for a while.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fundoscope.ingest import CROP_CLASSES, FALLOW
from fundoscope.ml.features import MONTHS
from fundoscope.timeseries import ACTIVE_NDVI_THRESHOLD

BARE_SOIL_NDVI = 0.25
MAX_BARE_MONTHS = {"avocado": 2, "citrus": 2, "mango": 2, "alfalfa": 2, "grape": 4}
REVIEWED_NOTE = "NDVI ok:"
NDVI_COLUMNS = tuple(f"ndvi_{month}" for month in MONTHS)

_REVIEWED = re.compile(re.escape(REVIEWED_NOTE) + r"\s*\S", re.IGNORECASE)


@dataclass(frozen=True)
class CurveMismatch:
    """A labeled campaign whose NDVI curve doesn't fit its crop.

    Attributes:
        crop: the label's crop as chosen in QGIS; `crop_class` is the model class it trains.
        problem: why the curve doesn't fit, for the user.
        ndvi: the parcel's median NDVI per month, August to July.
        reviewed: the label's notes say "NDVI ok: <reason>".
    """

    parcel_id: str
    campaign: str
    crop: str
    crop_class: str
    problem: str
    ndvi: tuple[float, ...]
    reviewed: bool


@dataclass(frozen=True)
class CurveCheck:
    """The result of the check: `checked` labeled campaigns and the ones that don't fit."""

    checked: int
    mismatches: list[CurveMismatch]

    @property
    def unreviewed(self) -> list[CurveMismatch]:
        return [mismatch for mismatch in self.mismatches if not mismatch.reviewed]


def campaign_curves(pixels: pd.DataFrame) -> pd.DataFrame:
    """Median NDVI per month (columns `aug` to `jul`) of each parcel and campaign (index)."""
    curves = pixels.groupby(["parcel_id", "campaign"])[list(NDVI_COLUMNS)].median()
    curves.columns = pd.Index(MONTHS)
    return curves


def check_curves(pixels: pd.DataFrame, labels: pd.DataFrame) -> CurveCheck:
    """Compare each labeled campaign in `pixels` with its label's crop.

    Args:
        pixels: training pixels with `parcel_id`, `campaign`, and `ndvi_aug` to `ndvi_jul`.
        labels: one row per parcel with `parcel_id`, `crop` (a value of `CROP_CLASSES`), and
            optionally `notes`. Every parcel in `pixels` must be here.

    Returns:
        The mismatches, sorted by parcel and campaign.
    """
    by_id = labels.set_index("parcel_id")
    curves = campaign_curves(pixels)
    mismatches = []
    for (parcel_id, campaign), curve in curves.iterrows():
        crop = by_id.at[parcel_id, "crop"]
        crop_class = CROP_CLASSES[crop]
        ndvi = curve.to_numpy(dtype=float)
        problem = curve_problem(crop_class, ndvi)
        if problem:
            note = by_id.at[parcel_id, "notes"] if "notes" in by_id else None
            mismatches.append(
                CurveMismatch(
                    parcel_id=parcel_id,
                    campaign=campaign,
                    crop=crop,
                    crop_class=crop_class,
                    problem=problem,
                    ndvi=tuple(round(float(value), 2) for value in ndvi),
                    reviewed=isinstance(note, str) and bool(_REVIEWED.search(note)),
                )
            )
    return CurveCheck(len(curves), mismatches)


def curve_problem(crop_class: str, ndvi: Sequence[float] | NDArray[np.float64]) -> str | None:
    """Why a campaign's monthly NDVI (12 values, August first) doesn't fit `crop_class`.

    Returns None when it fits.
    """
    ndvi = np.asarray(ndvi, dtype=float)
    peak = int(np.argmax(ndvi))
    if crop_class == FALLOW:
        if ndvi[peak] > ACTIVE_NDVI_THRESHOLD:
            return f"a crop grew: NDVI {ndvi[peak]:.2f} in {_month(peak)}"
        return None
    if ndvi[peak] <= ACTIVE_NDVI_THRESHOLD:
        return f"never green: NDVI peaked at {ndvi[peak]:.2f}"
    limit = MAX_BARE_MONTHS.get(crop_class)
    if limit is not None:
        start, length = _longest_run(ndvi < BARE_SOIL_NDVI)
        if length > limit:
            return (
                f"near bare soil for {length} months in a row "
                f"({_month(start)} to {_month(start + length - 1)})"
            )
    return None


def _longest_run(mask: NDArray[np.bool_]) -> tuple[int, int]:
    """Start and length of the longest run of True values (the first one on a tie)."""
    best_start = best_length = 0
    start = None
    for index, value in enumerate([*mask, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start > best_length:
                best_start, best_length = start, index - start
            start = None
    return best_start, best_length


def _month(index: int) -> str:
    return MONTHS[index].capitalize()
