"""What the database keeps about a parcel, as plain Python objects (FR-02).

The pipeline reads and writes these through a parcel store: `SqlParcelStore` (PostGIS) or
`MemoryParcelStore` (tests and runs without a database).
"""

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Protocol
from uuid import UUID

import pandas as pd
from shapely.geometry.base import BaseGeometry

# The daily statistics of a parcel, one row per acquisition date (see sentinel.statistical).
OBSERVATION_COLUMNS = ("ndvi", "ndwi", "ndvi_std", "valid_fraction", "pixels")


@dataclass(frozen=True)
class StoredParcel:
    """A parcel in the store.

    Attributes:
        id: the parcel's id.
        key: a hash of the outline: the same outline uploaded again is the same parcel.
        name: name from the uploaded file, or None.
        geometry: MultiPolygon in WGS84 lon/lat.
        utm_epsg: the UTM zone used for measurements.
        area_ha: area in hectares.
        repaired: the uploaded outline was invalid and its repair changed the area.
        observed_through: the last day whose satellite statistics were requested (None
            before the first analysis).
        created_at: when it was first stored.
    """

    id: UUID
    key: str
    name: str | None
    geometry: BaseGeometry
    utm_epsg: int
    area_ha: float
    repaired: bool
    observed_through: date | None = None
    created_at: datetime | None = None


@dataclass(frozen=True)
class StoredCampaign:
    """What the monthly composites gave for one campaign of a parcel.

    Attributes:
        active_area: 0..1, NaN if unknown.
        crop: the predicted crop class, or None.
        crop_confidence: 0..1, NaN without a crop.
        computed_through: the last day of images used (the campaign's last day once over).
        crop_model: the version of the crop model that predicted the crop (None without one).
    """

    active_area: float
    crop: str | None
    crop_confidence: float
    computed_through: date
    crop_model: str | None = None


@dataclass(frozen=True)
class StoredHistory:
    """Everything stored for a parcel's history.

    Attributes:
        observations: the daily statistics, `OBSERVATION_COLUMNS` with a DatetimeIndex.
        observed_through: the last day requested (None if nothing was).
        campaigns: the composites' results by campaign ("2024/25").
    """

    observations: pd.DataFrame
    observed_through: date | None = None
    campaigns: dict[str, StoredCampaign] = field(default_factory=dict)


def empty_observations() -> pd.DataFrame:
    """No daily statistics, with the right columns and index."""
    index = pd.DatetimeIndex([], name="date")
    return pd.DataFrame({name: pd.Series(dtype=float) for name in OBSERVATION_COLUMNS}, index=index)


def none_if_nan(value: float | None) -> float | None:
    """NaN becomes None (a database NULL)."""
    return None if value is None or math.isnan(value) else float(value)


class ParcelStore(Protocol):
    """Where parcels and their histories are kept."""

    def find(self, key: str) -> StoredParcel | None: ...

    def add(
        self,
        key: str,
        name: str | None,
        geometry: BaseGeometry,
        utm_epsg: int,
        area_ha: float,
        repaired: bool,
    ) -> StoredParcel: ...

    def get(self, parcel_id: UUID) -> StoredParcel | None: ...

    def parcels(self) -> list[StoredParcel]: ...

    def load(self, parcel_id: UUID) -> StoredHistory: ...

    def save(
        self,
        parcel_id: UUID,
        observations: pd.DataFrame,
        observed_through: date,
        campaigns: dict[str, StoredCampaign],
    ) -> None: ...
