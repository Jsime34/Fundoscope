import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fundoscope.ml import BANDS, feature_columns

RED = 0.05


def nir_for(ndvi):
    """The near-infrared reflectance that gives `ndvi` with red at 0.05."""
    ndvi = np.asarray(ndvi, dtype=float)
    return RED * (1 + ndvi) / (1 - ndvi)


def stack_from_ndvi(ndvi, start: str = "2026-08-01", x0: float = 400_005.0) -> xr.DataArray:
    """A (month, band, y, x) stack whose NDVI is `ndvi` (month, y, x); NaN stays NaN."""
    ndvi = np.asarray(ndvi, dtype=float)
    months, rows, columns = ndvi.shape
    data = np.full((months, len(BANDS), rows, columns), 0.1)
    data[:, BANDS.index("B04")] = RED
    data[:, BANDS.index("B08")] = nir_for(ndvi)
    data[np.broadcast_to(np.isnan(ndvi)[:, None], data.shape)] = np.nan
    return xr.DataArray(
        data.astype(np.float32),
        dims=("month", "band", "y", "x"),
        coords={
            "month": pd.date_range(start, periods=months, freq="MS"),
            "band": list(BANDS),
            "y": 8_400_005.0 + 10 * np.arange(rows)[::-1],
            "x": x0 + 10 * np.arange(columns),
        },
    )


class PeakModel:
    """A crop model that looks only at the NDVI peak: above 0.7 is grape (90 %), 0.4 to 0.7 is
    unsure (other annual crops 40 %), and below 0.4 is fallow/bare (90 %)."""

    classes_ = np.array(["alfalfa", "fallow/bare", "grape", "other annual crops"])

    def __init__(self, months: int = 4):
        self.n_features_in_ = len(feature_columns(months))

    def predict_proba(self, features):
        peak = features[:, -6]  # ndvi_peak, the first phenology measure
        grape = np.tile([0.1 / 3, 0.1 / 3, 0.9, 0.1 / 3], (len(peak), 1))
        unsure = np.tile([0.2, 0.2, 0.2, 0.4], (len(peak), 1))
        bare = np.tile([0.1 / 3, 0.9, 0.1 / 3, 0.1 / 3], (len(peak), 1))
        return np.where((peak > 0.7)[:, None], grape, np.where((peak > 0.4)[:, None], unsure, bare))


@pytest.fixture
def peak_model() -> type[PeakModel]:
    return PeakModel


@pytest.fixture
def make_stack():
    """`make_stack(ndvi, start=...)` builds a stack from an NDVI array (month, y, x)."""
    return stack_from_ndvi
