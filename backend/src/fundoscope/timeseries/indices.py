"""Spectral indices from Sentinel-2 surface reflectance.

NDVI = (B08 - B04) / (B08 + B04): green biomass (Rouse et al., 1974).
NDWI = (B8A - B11) / (B8A + B11): canopy water content (Gao, 1996), also called NDMI.

Both range from -1 to 1. Inputs are surface reflectances (0..1), as scalars or numpy arrays
of the same shape. Raw L2A digital numbers from processing baseline 04.00 onward carry a
-1000 offset, so they must be harmonized first (Sentinel Hub and openEO return harmonized
values). Where both bands are 0 (no data), the result is NaN.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def normalized_difference(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """(a - b) / (a + b), elementwise; NaN where a + b == 0."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    total = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total != 0, (a - b) / total, np.nan)


def ndvi(red: ArrayLike, nir: ArrayLike) -> NDArray[np.float64]:
    """NDVI from red (B04) and near-infrared (B08) reflectance."""
    return normalized_difference(nir, red)


def ndwi(nir: ArrayLike, swir: ArrayLike) -> NDArray[np.float64]:
    """NDWI (Gao) from narrow near-infrared (B8A) and short-wave infrared (B11) reflectance."""
    return normalized_difference(nir, swir)
