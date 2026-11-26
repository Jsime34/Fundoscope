"""Parcels, their daily statistics and campaign results, and jobs (FR-02).

Revision ID: 0001
Revises:
Create Date: 2026-11-26
"""

import sqlalchemy as sa
from alembic import op
from geoalchemy2 import Geometry
from sqlalchemy.dialects.postgresql import UUID

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    op.create_table(
        "parcels",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("key", sa.String(64), nullable=False, unique=True),
        sa.Column("name", sa.Text()),
        sa.Column("geom", Geometry("MULTIPOLYGON", srid=4326, spatial_index=False), nullable=False),
        sa.Column("utm_epsg", sa.Integer(), nullable=False),
        sa.Column("area_ha", sa.Float(), nullable=False),
        sa.Column("repaired", sa.Boolean(), nullable=False),
        sa.Column("observed_through", sa.Date()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_parcels_geom", "parcels", ["geom"], postgresql_using="gist")
    op.create_table(
        "observations",
        sa.Column(
            "parcel_id",
            UUID(as_uuid=True),
            sa.ForeignKey("parcels.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("date", sa.Date(), primary_key=True),
        sa.Column("ndvi", sa.Float()),
        sa.Column("ndwi", sa.Float()),
        sa.Column("ndvi_std", sa.Float()),
        sa.Column("valid_fraction", sa.Float(), nullable=False),
        sa.Column("pixels", sa.Integer(), nullable=False),
    )
    op.create_table(
        "campaigns",
        sa.Column(
            "parcel_id",
            UUID(as_uuid=True),
            sa.ForeignKey("parcels.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("campaign", sa.String(7), primary_key=True),
        sa.Column("active_area", sa.Float()),
        sa.Column("crop", sa.Text()),
        sa.Column("crop_confidence", sa.Float()),
        sa.Column("computed_through", sa.Date(), nullable=False),
        sa.Column("crop_model", sa.Text()),
    )
    op.create_table(
        "jobs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("parcel_id", UUID(as_uuid=True), sa.ForeignKey("parcels.id", ondelete="CASCADE")),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("status", sa.String(10), nullable=False),
        sa.Column("progress", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column("processing_units", sa.Float(), nullable=False),
        sa.Column("credits", sa.Float(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_jobs_parcel_id", "jobs", ["parcel_id"])


def downgrade() -> None:
    op.drop_table("jobs")
    op.drop_table("campaigns")
    op.drop_table("observations")
    op.drop_index("ix_parcels_geom", table_name="parcels")
    op.drop_table("parcels")
