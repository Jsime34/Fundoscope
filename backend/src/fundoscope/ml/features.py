"""Features of one pixel and campaign for the crop model (FR-06).

Input: a campaign's 12 monthly composites (August to July) as reflectance 0..1, with dims
(month, band, y, x), UTM x/y pixel centers in meters, and NaN where a pixel had no clear view
or lies outside the area of interest. Output: one row per usable pixel with 150 features:

- 144 monthly values: the 10 bands plus NDVI and NDWI for each month (`B04_aug`, `ndvi_jul`);
- 6 phenology measures of the NDVI curve: peak, month of peak (0 = August), minimum,
  amplitude, sum, and months above 0.4.

A missing month is filled by linear interpolation between its neighbors (the nearest month at
either end of the campaign). A pixel missing more than 4 of the 12 months is left out, so
pixels outside the area (masked to NaN by the caller) drop out on their own. The same
features feed training, parcel predictions, and the valley count.

A campaign still in progress has only its first months. Given August to month n, the features
are the same for those months only: n x 12 monthly values and the 6 phenology measures of the
shorter curve (`feature_columns(n)`), and a pixel may miss a third of them. These feed the
season-to-date models of the valley count (FR-14). `season_to_date` cuts full-campaign
features down to the first n months, to train those models on the saved training pixels.
"""

import zlib

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from fundoscope.timeseries import ACTIVE_NDVI_THRESHOLD, ndvi, ndwi

BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
MONTHS = ("aug", "sep", "oct", "nov", "dec", "jan", "feb", "mar", "apr", "may", "jun", "jul")
MONTHLY_VARIABLES = (*BANDS, "ndvi", "ndwi")
PHENOLOGY = (
    "ndvi_peak",
    "ndvi_peak_month",
    "ndvi_min",
    "ndvi_amplitude",
    "ndvi_sum",
    "ndvi_months_active",
)


def feature_columns(months: int = len(MONTHS)) -> tuple[str, ...]:
    """The feature names for the first `months` months of a campaign, August first.

    Raises:
        ValueError: `months` isn't 1 to 12.
    """
    if not 1 <= months <= len(MONTHS):
        raise ValueError(f"A campaign has 1 to {len(MONTHS)} months, not {months}.")
    return (
        *(f"{variable}_{month}" for variable in MONTHLY_VARIABLES for month in MONTHS[:months]),
        *PHENOLOGY,
    )


FEATURE_COLUMNS = feature_columns()


def max_missing_months(months: int) -> int:
    """How many of a pixel's `months` months may be missing: a third (4 of 12)."""
    return months // 3


MAX_MISSING_MONTHS = max_missing_months(len(MONTHS))
# Training pixels come from the parcel shrunk by this much (m), so no pixel mixes in the
# road, the field edge, or the neighbor.
INTERIOR_BUFFER_M = 15.0
MAX_PIXELS_PER_SAMPLE = 50


class FeatureError(ValueError):
    """The composites can't be turned into features. The message is meant for the user."""


def pixel_features(stack: xr.DataArray) -> pd.DataFrame:
    """The features of every usable pixel of one campaign: 150 for a whole campaign.

    Args:
        stack: reflectance 0..1 with dims (month, band, y, x): the first n months of one
            campaign (all 12 once it is over), August first, and at least the bands in
            `BANDS`; x/y are UTM pixel centers (m).

    Returns:
        One row per pixel missing at most a third of the n months (4 of 12), indexed by
        (x, y) in UTM meters, with `feature_columns(n)` as float columns.

    Raises:
        FeatureError: wrong months or missing bands.
    """
    _check(stack)
    data = stack.sel(band=list(BANDS)).transpose("month", "band", "y", "x").to_numpy()
    n_months, n_bands, ny, nx = data.shape
    bands = data.reshape(n_months, n_bands, ny * nx).astype(np.float64)
    band = {name: bands[:, i] for i, name in enumerate(BANDS)}
    indices = np.stack([ndvi(band["B04"], band["B08"]), ndwi(band["B8A"], band["B11"])], axis=1)
    values = np.concatenate([bands, indices], axis=1)  # (month, variable, pixel)

    missing = np.isnan(values).any(axis=1)  # a month counts only if every variable is clear
    keep = missing.sum(axis=0) <= max_missing_months(n_months)
    kept = np.where(missing[:, None, keep], np.nan, values[:, :, keep])
    n_variables, n_kept = kept.shape[1:]
    filled = _fill_months(kept.reshape(n_months, n_variables * n_kept)).reshape(kept.shape)

    monthly = filled.transpose(1, 0, 2).reshape(n_variables * n_months, n_kept)  # var-major
    phenology = _phenology(filled[:, MONTHLY_VARIABLES.index("ndvi")])

    xx, yy = np.meshgrid(stack["x"].to_numpy(), stack["y"].to_numpy())
    index = pd.MultiIndex.from_arrays([xx.ravel()[keep], yy.ravel()[keep]], names=("x", "y"))
    return pd.DataFrame(
        np.vstack([monthly, phenology]).T, index=index, columns=list(feature_columns(n_months))
    )


def season_to_date(features: pd.DataFrame, months: int) -> pd.DataFrame:
    """Full-campaign features cut down to the first `months` months.

    The monthly values of later months are dropped and the phenology is measured again on the
    shorter NDVI curve, so the result matches `pixel_features` of the first `months` months.
    One difference: a month that was missing was filled using later months too, which a
    campaign in progress can't do. Clear months are the rule in the coastal desert, so this
    matters little.

    Args:
        features: rows with at least the 150 `FEATURE_COLUMNS` (other columns are dropped).
        months: 1 to 12.

    Returns:
        The same rows and index, with `feature_columns(months)` as float columns.
    """
    columns = feature_columns(months)
    monthly = features[list(columns[: -len(PHENOLOGY)])].astype(np.float64)
    curve = features[[f"ndvi_{month}" for month in MONTHS[:months]]].to_numpy(np.float64).T
    phenology = pd.DataFrame(_phenology(curve).T, index=features.index, columns=list(PHENOLOGY))
    return pd.concat([monthly, phenology], axis=1)


def training_sample(
    features: pd.DataFrame, key: str, max_pixels: int = MAX_PIXELS_PER_SAMPLE
) -> pd.DataFrame:
    """Up to `max_pixels` rows chosen at random, in their original order.

    The same `key` (for example "ARE-0002 2024/25") always picks the same pixels, so a rerun
    reproduces the training data.
    """
    if len(features) <= max_pixels:
        return features
    chosen = features.sample(n=max_pixels, random_state=zlib.crc32(key.encode()))
    return features.loc[features.index.isin(chosen.index)]


def _phenology(curve: NDArray[np.float64]) -> NDArray[np.float64]:
    """The 6 `PHENOLOGY` measures (rows) of NDVI curves with dims (month, pixel)."""
    peak, low = curve.max(axis=0), curve.min(axis=0)
    return np.stack(
        [
            peak,
            curve.argmax(axis=0),
            low,
            peak - low,
            curve.sum(axis=0),
            (curve > ACTIVE_NDVI_THRESHOLD).sum(axis=0),
        ]
    ).astype(np.float64)


def _check(stack: xr.DataArray) -> None:
    if set(stack.dims) != {"month", "band", "y", "x"}:
        raise FeatureError(f"Expected dims (month, band, y, x), got {stack.dims}.")
    missing = [name for name in BANDS if name not in stack["band"].to_numpy()]
    if missing:
        raise FeatureError(f"The composites are missing the bands {', '.join(missing)}.")
    months = pd.DatetimeIndex(stack["month"].to_numpy())
    if not len(months):
        raise FeatureError("The composites have no months.")
    if len(months) > len(MONTHS) or months[0].month != 8:
        raise FeatureError(
            "Expected the months of one campaign, August first (up to 12); got "
            f"{len(months)} months starting {months[0]:%Y-%m}."
        )
    expected = pd.date_range(months[0], periods=len(months), freq="MS")
    if not months.equals(expected):
        raise FeatureError("The months must be consecutive, August first.")


def _fill_months(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Linear interpolation along axis 0 (months); ends take the nearest value.

    Every column must have at least one value.
    """
    n = values.shape[0]
    months = np.arange(n)[:, None]
    valid = ~np.isnan(values)
    before = np.maximum.accumulate(np.where(valid, months, -1), axis=0)
    after = np.minimum.accumulate(np.where(valid, months, n)[::-1], axis=0)[::-1]
    low = np.where(before < 0, after, before)
    high = np.where(after >= n, before, after)
    span = high - low
    weight = np.divide(months - low, span, out=np.zeros(values.shape), where=span > 0)
    low_value = np.take_along_axis(values, low, axis=0)
    high_value = np.take_along_axis(values, high, axis=0)
    return low_value + weight * (high_value - low_value)
