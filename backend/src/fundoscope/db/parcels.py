"""The parcel store in PostGIS (FR-02).

`SqlParcelStore` works inside a SQLAlchemy session that the caller opens and commits.
`AutocommitParcelStore` runs each call in its own transaction instead, for background
jobs: nothing stays open while satellite data download between reading and saving a
history.
"""

from datetime import date
from uuid import UUID

import pandas as pd
from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry.base import BaseGeometry
from sqlalchemy import Engine, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from fundoscope.db.models import CampaignRow, ObservationRow, ParcelRow
from fundoscope.db.records import (
    OBSERVATION_COLUMNS,
    StoredCampaign,
    StoredHistory,
    StoredParcel,
    empty_observations,
    none_if_nan,
)
from fundoscope.db.session import session_scope

SRID = 4326


class SqlParcelStore:
    """Parcels and their histories in the `parcels`, `observations`, and `campaigns` tables."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def find(self, key: str) -> StoredParcel | None:
        row = self.session.scalars(select(ParcelRow).where(ParcelRow.key == key)).first()
        return _parcel(row) if row else None

    def add(
        self,
        key: str,
        name: str | None,
        geometry: BaseGeometry,
        utm_epsg: int,
        area_ha: float,
        repaired: bool,
    ) -> StoredParcel:
        row = ParcelRow(
            key=key,
            name=name,
            geom=from_shape(geometry, srid=SRID),
            utm_epsg=utm_epsg,
            area_ha=area_ha,
            repaired=repaired,
        )
        self.session.add(row)
        self.session.flush()
        self.session.refresh(row)
        return _parcel(row)

    def get(self, parcel_id: UUID) -> StoredParcel | None:
        row = self.session.get(ParcelRow, parcel_id)
        return _parcel(row) if row else None

    def parcels(self) -> list[StoredParcel]:
        """Newest first."""
        rows = self.session.scalars(select(ParcelRow).order_by(ParcelRow.created_at.desc()))
        return [_parcel(row) for row in rows]

    def load(self, parcel_id: UUID) -> StoredHistory:
        parcel = self.session.get_one(ParcelRow, parcel_id)
        rows = self.session.scalars(
            select(ObservationRow)
            .where(ObservationRow.parcel_id == parcel_id)
            .order_by(ObservationRow.date)
        ).all()
        observations = empty_observations()
        if rows:
            observations = pd.DataFrame(
                {name: [getattr(row, name) for row in rows] for name in OBSERVATION_COLUMNS},
                index=pd.DatetimeIndex([row.date for row in rows], name="date"),
            ).astype({"ndvi": float, "ndwi": float, "ndvi_std": float, "valid_fraction": float})
        campaigns = {
            row.campaign: StoredCampaign(
                active_area=_nan(row.active_area),
                crop=row.crop,
                crop_confidence=_nan(row.crop_confidence),
                computed_through=row.computed_through,
                crop_model=row.crop_model,
            )
            for row in self.session.scalars(
                select(CampaignRow).where(CampaignRow.parcel_id == parcel_id)
            )
        }
        return StoredHistory(observations, parcel.observed_through, campaigns)

    def save(
        self,
        parcel_id: UUID,
        observations: pd.DataFrame,
        observed_through: date,
        campaigns: dict[str, StoredCampaign],
    ) -> None:
        """Add or replace the given dates and campaigns, and move `observed_through`."""
        rows = [
            {
                "parcel_id": parcel_id,
                "date": day.date(),
                "ndvi": none_if_nan(row.ndvi),
                "ndwi": none_if_nan(row.ndwi),
                "ndvi_std": none_if_nan(row.ndvi_std),
                "valid_fraction": float(row.valid_fraction),
                "pixels": int(row.pixels),
            }
            for day, row in observations.iterrows()
        ]
        if rows:
            statement = insert(ObservationRow).values(rows)
            columns = {name: statement.excluded[name] for name in OBSERVATION_COLUMNS}
            self.session.execute(
                statement.on_conflict_do_update(index_elements=["parcel_id", "date"], set_=columns)
            )
        for label, result in campaigns.items():
            values = {
                "active_area": none_if_nan(result.active_area),
                "crop": result.crop,
                "crop_confidence": none_if_nan(result.crop_confidence),
                "computed_through": result.computed_through,
                "crop_model": result.crop_model,
            }
            statement = insert(CampaignRow).values(parcel_id=parcel_id, campaign=label, **values)
            self.session.execute(
                statement.on_conflict_do_update(
                    index_elements=["parcel_id", "campaign"], set_=values
                )
            )
        self.session.get_one(ParcelRow, parcel_id).observed_through = observed_through
        self.session.flush()


class AutocommitParcelStore:
    """The PostGIS store with each call committed in its own transaction."""

    def __init__(self, bind: Engine | None = None) -> None:
        self.bind = bind

    def find(self, key: str) -> StoredParcel | None:
        with session_scope(self.bind) as session:
            return SqlParcelStore(session).find(key)

    def add(
        self,
        key: str,
        name: str | None,
        geometry: BaseGeometry,
        utm_epsg: int,
        area_ha: float,
        repaired: bool,
    ) -> StoredParcel:
        with session_scope(self.bind) as session:
            return SqlParcelStore(session).add(key, name, geometry, utm_epsg, area_ha, repaired)

    def get(self, parcel_id: UUID) -> StoredParcel | None:
        with session_scope(self.bind) as session:
            return SqlParcelStore(session).get(parcel_id)

    def parcels(self) -> list[StoredParcel]:
        with session_scope(self.bind) as session:
            return SqlParcelStore(session).parcels()

    def load(self, parcel_id: UUID) -> StoredHistory:
        with session_scope(self.bind) as session:
            return SqlParcelStore(session).load(parcel_id)

    def save(
        self,
        parcel_id: UUID,
        observations: pd.DataFrame,
        observed_through: date,
        campaigns: dict[str, StoredCampaign],
    ) -> None:
        with session_scope(self.bind) as session:
            SqlParcelStore(session).save(parcel_id, observations, observed_through, campaigns)


def _parcel(row: ParcelRow) -> StoredParcel:
    return StoredParcel(
        id=row.id,
        key=row.key,
        name=row.name,
        geometry=to_shape(row.geom),
        utm_epsg=row.utm_epsg,
        area_ha=row.area_ha,
        repaired=row.repaired,
        observed_through=row.observed_through,
        created_at=row.created_at,
    )


def _nan(value: float | None) -> float:
    return float("nan") if value is None else value
