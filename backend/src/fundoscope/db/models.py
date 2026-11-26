"""The database tables (FR-02), as SQLAlchemy models. The migrations create them.

- `parcels`: one row per parcel outline, with its geometry (GiST index);
- `observations`: the daily satellite statistics of each parcel, one row per date;
- `campaigns`: what the monthly composites gave for each campaign of a parcel;
- `jobs`: the analyses and valley refreshes run in the background (FR-13).
"""

import uuid
from datetime import UTC, date, datetime

from geoalchemy2 import Geometry
from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now(UTC)


class ParcelRow(Base):
    __tablename__ = "parcels"
    __table_args__ = (Index("ix_parcels_geom", "geom", postgresql_using="gist"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key: Mapped[str] = mapped_column(String(64), unique=True)
    name: Mapped[str | None] = mapped_column(Text)
    geom = mapped_column(Geometry("MULTIPOLYGON", srid=4326, spatial_index=False), nullable=False)
    utm_epsg: Mapped[int] = mapped_column(Integer)
    area_ha: Mapped[float] = mapped_column(Float)
    repaired: Mapped[bool] = mapped_column(Boolean, default=False)
    observed_through: Mapped[date | None] = mapped_column(Date)
    # Set by Python: Postgres's now() is the same for every row of a transaction.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, server_default=func.now()
    )


class ObservationRow(Base):
    __tablename__ = "observations"

    parcel_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("parcels.id", ondelete="CASCADE"), primary_key=True
    )
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    ndvi: Mapped[float | None] = mapped_column(Float)
    ndwi: Mapped[float | None] = mapped_column(Float)
    ndvi_std: Mapped[float | None] = mapped_column(Float)
    valid_fraction: Mapped[float] = mapped_column(Float)
    pixels: Mapped[int] = mapped_column(Integer)


class CampaignRow(Base):
    __tablename__ = "campaigns"

    parcel_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("parcels.id", ondelete="CASCADE"), primary_key=True
    )
    campaign: Mapped[str] = mapped_column(String(7), primary_key=True)
    active_area: Mapped[float | None] = mapped_column(Float)
    crop: Mapped[str | None] = mapped_column(Text)
    crop_confidence: Mapped[float | None] = mapped_column(Float)
    computed_through: Mapped[date] = mapped_column(Date)
    crop_model: Mapped[str | None] = mapped_column(Text)


class JobRow(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    parcel_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("parcels.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(20))  # "analyze" | "valley_refresh"
    status: Mapped[str] = mapped_column(  # queued|running|done|failed
        String(10), default="queued", index=True
    )
    progress: Mapped[int] = mapped_column(Integer, default=0)  # 0-100
    step: Mapped[str | None] = mapped_column(Text)  # what it is doing now (migration 0002)
    error: Mapped[str | None] = mapped_column(Text)
    processing_units: Mapped[float] = mapped_column(Float, default=0.0)
    credits: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
