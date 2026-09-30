"""Share of a parcel that was actively farmed in a campaign (FR-05, "% active area").

Works on the campaign's **max-NDVI composite**: for every pixel inside the parcel, its highest
NDVI during the campaign (pixels outside the parcel, or never seen clear, are NaN). A pixel
is "active" if that peak exceeds 0.4: a growing crop reaches it at some point in the
campaign, while bare soil and fallow land stay well below. The threshold is a starting value
to calibrate on partner parcels.
"""

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

ACTIVE_NDVI_THRESHOLD = 0.4


def max_ndvi_composite(ndvi_stack: ArrayLike) -> NDArray[np.float64]:
    """Per-pixel maximum over time of a (time, y, x) NDVI stack, ignoring NaN.

    Pixels that are NaN at every time step stay NaN.
    """
    stack = np.asarray(ndvi_stack, dtype=np.float64)
    if stack.ndim != 3:
        raise ValueError(f"Expected a (time, y, x) stack, got an array with shape {stack.shape}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # "All-NaN slice": expected outside
        return np.nanmax(stack, axis=0)


def active_mask(max_ndvi: ArrayLike, threshold: float = ACTIVE_NDVI_THRESHOLD) -> NDArray[np.bool_]:
    """True where a pixel's peak NDVI exceeds `threshold` (NaN pixels are False)."""
    peak = np.asarray(max_ndvi, dtype=np.float64)
    return np.greater(peak, threshold, where=~np.isnan(peak), out=np.zeros(peak.shape, bool))


def active_fraction(max_ndvi: ArrayLike, threshold: float = ACTIVE_NDVI_THRESHOLD) -> float:
    """Share (0..1) of the observed pixels whose peak NDVI exceeds `threshold`.

    Raises:
        ValueError: if no pixel has a value (nothing to measure).
    """
    peak = np.asarray(max_ndvi, dtype=np.float64)
    observed = ~np.isnan(peak)
    if not observed.any():
        raise ValueError("No observed pixels: the composite is empty or entirely NaN.")
    return float(np.mean(peak[observed] > threshold))
