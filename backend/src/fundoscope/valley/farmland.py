"""The valley's farmland: the pixels the crop count classifies (FR-14, FR-08).

A pixel is farmland when its monthly NDVI passed 0.4 at least once in the last three campaigns
(a crop covered it at some point) and ESA WorldCover 2021 doesn't map it as built-up or
permanent water: gardens and parks in town and the vegetation along canals and reservoirs are
green, but they aren't farmland. Desert and roads never reach 0.4.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fundoscope.timeseries import ACTIVE_NDVI_THRESHOLD, active_mask

FARMLAND_NDVI = ACTIVE_NDVI_THRESHOLD
WORLDCOVER_BUILT_UP = 50
WORLDCOVER_WATER = 80
NOT_FARMLAND_COVER = (WORLDCOVER_BUILT_UP, WORLDCOVER_WATER)


def farmland_mask(
    peak_ndvi: ArrayLike, land_cover: ArrayLike, inside: ArrayLike
) -> NDArray[np.bool_]:
    """(y, x) mask of the farmland pixels.

    Args:
        peak_ndvi: each pixel's highest monthly NDVI over the three campaigns (NaN if never
            seen clear).
        land_cover: ESA WorldCover classes on the same grid.
        inside: True where the pixel lies inside the valley.

    Raises:
        ValueError: the three grids don't have the same shape.
    """
    peak, cover, area = np.asarray(peak_ndvi), np.asarray(land_cover), np.asarray(inside, bool)
    if not peak.shape == cover.shape == area.shape:
        raise ValueError(
            f"The NDVI {peak.shape}, land cover {cover.shape}, and valley {area.shape} grids "
            "must have the same shape."
        )
    return active_mask(peak, FARMLAND_NDVI) & ~np.isin(cover, NOT_FARMLAND_COVER) & area
