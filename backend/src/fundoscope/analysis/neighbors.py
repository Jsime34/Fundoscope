"""How a parcel compares with the farmland around it (FR-08).

Neighbors are the farmland pixels whose centers lie between 50 m and 2 km from the parcel. For
each campaign, every pixel gets its mean monthly NDVI (the months so far, for a campaign in
progress). The parcel's value is the mean of its own pixels, and it is placed in a band of
the neighbors' values: below p10, p10-p25, p25-p50, p50-p75, p75-p90, or above p90. A
campaign below p10 is "underperforming": greener farmland surrounds it on every side, so the
water, the soil, or the management of the parcel may be the reason.

The comparison needs enough neighbors to be meaningful: at least 100 pixels (1 ha).
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from shapely.geometry.base import BaseGeometry

from fundoscope.analysis.flags import Flag

INNER_M = 50.0
OUTER_M = 2_000.0
PERCENTILES = (10, 25, 50, 75, 90)
BANDS = ("below p10", "p10-p25", "p25-p50", "p50-p75", "p75-p90", "above p90")
MIN_NEIGHBOR_PIXELS = 100
UNDERPERFORMING = "underperforming"


@dataclass(frozen=True)
class NeighborBand:
    """Where one campaign of the parcel stands among its neighbors.

    Attributes:
        campaign: e.g. "2025/26".
        months: months of the campaign compared (12 once it is over).
        parcel_ndvi: the mean NDVI of the parcel's pixels.
        parcel_pixels: the parcel's pixels with at least one clear month.
        neighbor_pixels: the neighbors with at least one clear month.
        percentiles: the neighbors' p10, p25, p50, p75, and p90, keyed by percentile.
        band: one of `BANDS`.
    """

    campaign: str
    months: int
    parcel_ndvi: float
    parcel_pixels: int
    neighbor_pixels: int
    percentiles: dict[int, float] = field(default_factory=dict)
    band: str = ""

    @property
    def underperforming(self) -> bool:
        return self.band == BANDS[0]


def neighbor_ring(parcel_utm: BaseGeometry) -> BaseGeometry:
    """The area between 50 m and 2 km from the parcel (UTM meters)."""
    return parcel_utm.buffer(OUTER_M).difference(parcel_utm.buffer(INNER_M))


def campaign_ndvi(monthly: ArrayLike) -> NDArray[np.float64]:
    """Each pixel's mean NDVI over the months of a (month, y, x) stack; NaN if never clear."""
    stack = np.asarray(monthly, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # "Mean of empty slice": never clear
        return np.nanmean(stack, axis=0)


def band_of(value: float, percentiles: dict[int, float]) -> str:
    """The band of a value: how many of the percentiles it reaches."""
    edges = [percentiles[p] for p in PERCENTILES]
    return BANDS[int(np.searchsorted(edges, value, side="right"))]


def compare(
    campaign: str,
    months: int,
    parcel: ArrayLike,
    neighbors: ArrayLike,
    min_neighbors: int = MIN_NEIGHBOR_PIXELS,
) -> NeighborBand | None:
    """Place the parcel's mean NDVI among its neighbors' NDVI.

    Args:
        parcel, neighbors: per-pixel campaign NDVI (NaN pixels are ignored).

    Returns:
        None when the parcel has no clear pixel or there are fewer than `min_neighbors`
        neighbors.
    """
    own = np.asarray(parcel, dtype=np.float64).ravel()
    around = np.asarray(neighbors, dtype=np.float64).ravel()
    own, around = own[~np.isnan(own)], around[~np.isnan(around)]
    if not len(own) or len(around) < min_neighbors:
        return None
    percentiles = dict(zip(PERCENTILES, np.percentile(around, PERCENTILES).tolist(), strict=True))
    value = float(own.mean())
    return NeighborBand(
        campaign, months, value, len(own), len(around), percentiles, band_of(value, percentiles)
    )


def underperforming_flags(bands: list[NeighborBand]) -> list[Flag]:
    """FR-08: a flag for every campaign below its neighbors' 10th percentile."""
    return [
        Flag(
            campaign=band.campaign,
            kind=UNDERPERFORMING,
            message=(
                "Less green than 90 % of the farmland around it: mean NDVI "
                f"{band.parcel_ndvi:.2f}, against {band.percentiles[50]:.2f} for the median "
                "neighbor."
            ),
            evidence={
                "parcel_ndvi": round(band.parcel_ndvi, 3),
                "neighbors_p10": round(band.percentiles[10], 3),
                "neighbors_median": round(band.percentiles[50], 3),
                "neighbor_pixels": band.neighbor_pixels,
                "months_compared": band.months,
            },
        )
        for band in bands
        if band.underperforming
    ]
