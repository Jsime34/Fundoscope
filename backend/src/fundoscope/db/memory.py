"""A parcel store in memory: for tests and for runs without a database (FR-02)."""

from dataclasses import replace
from datetime import UTC, date, datetime
from uuid import UUID, uuid4

import pandas as pd
from shapely.geometry.base import BaseGeometry

from fundoscope.db.records import (
    OBSERVATION_COLUMNS,
    StoredCampaign,
    StoredHistory,
    StoredParcel,
    empty_observations,
)


class MemoryParcelStore:
    """Keeps everything in dictionaries; lost when the process ends."""

    def __init__(self) -> None:
        self._parcels: dict[UUID, StoredParcel] = {}
        self._observations: dict[UUID, pd.DataFrame] = {}
        self._campaigns: dict[UUID, dict[str, StoredCampaign]] = {}

    def find(self, key: str) -> StoredParcel | None:
        return next((p for p in self._parcels.values() if p.key == key), None)

    def add(
        self,
        key: str,
        name: str | None,
        geometry: BaseGeometry,
        utm_epsg: int,
        area_ha: float,
        repaired: bool,
    ) -> StoredParcel:
        parcel = StoredParcel(
            uuid4(), key, name, geometry, utm_epsg, area_ha, repaired, None, datetime.now(UTC)
        )
        self._parcels[parcel.id] = parcel
        return parcel

    def get(self, parcel_id: UUID) -> StoredParcel | None:
        return self._parcels.get(parcel_id)

    def parcels(self) -> list[StoredParcel]:
        """Newest first (the order they were added: the clock may not tell them apart)."""
        return list(reversed(self._parcels.values()))

    def load(self, parcel_id: UUID) -> StoredHistory:
        parcel = self._parcels[parcel_id]
        observations = self._observations.get(parcel_id, empty_observations())
        campaigns = dict(self._campaigns.get(parcel_id, {}))
        return StoredHistory(observations.copy(), parcel.observed_through, campaigns)

    def save(
        self,
        parcel_id: UUID,
        observations: pd.DataFrame,
        observed_through: date,
        campaigns: dict[str, StoredCampaign],
    ) -> None:
        new = observations[list(OBSERVATION_COLUMNS)]
        old = self._observations.get(parcel_id)
        if old is not None and len(old):
            new = pd.concat([old[~old.index.isin(new.index)], new])
        self._observations[parcel_id] = new.sort_index()
        self._campaigns.setdefault(parcel_id, {}).update(campaigns)
        parcel = self._parcels[parcel_id]
        self._parcels[parcel_id] = replace(parcel, observed_through=observed_through)
