from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest


def _crop_curve(
    peak_day: float, peak: float = 0.8, base: float = 0.15, width_days: float = 40.0
) -> Callable[[np.ndarray], np.ndarray]:
    """NDVI of one crop cycle: bare soil (`base`) rising to `peak` around `peak_day`."""

    def curve(days: np.ndarray) -> np.ndarray:
        return base + (peak - base) * np.exp(-0.5 * ((days - peak_day) / width_days) ** 2)

    return curve


def _make_observations(
    start: str = "2022-08-01",
    end: str = "2023-07-31",
    step_days: int = 5,
    ndvi: float | Callable[[np.ndarray], np.ndarray] = 0.5,
    ndwi: float = 0.1,
    valid_fraction: float = 1.0,
) -> pd.DataFrame:
    """Synthetic parcel observations every `step_days`, in the format `clean_series` expects.

    `ndvi` is a constant or a function of "days since `start`".
    """
    index = pd.date_range(start, end, freq=f"{step_days}D", name="date")
    days = np.arange(len(index), dtype=float) * step_days
    ndvi_values = ndvi(days) if callable(ndvi) else np.full(len(index), float(ndvi))
    return pd.DataFrame(
        {"ndvi": ndvi_values, "ndwi": ndwi, "valid_fraction": valid_fraction}, index=index
    )


def _from_values(values: list[float], start: str = "2022-08-01", step_days: int = 5):
    """Observations with the given NDVI values, one every `step_days`."""
    index = pd.date_range(start, periods=len(values), freq=f"{step_days}D", name="date")
    return pd.DataFrame({"ndvi": values, "ndwi": 0.1, "valid_fraction": 1.0}, index=index)


@pytest.fixture
def make_observations():
    return _make_observations


@pytest.fixture
def crop_curve():
    return _crop_curve


@pytest.fixture
def observations_from_values():
    return _from_values
