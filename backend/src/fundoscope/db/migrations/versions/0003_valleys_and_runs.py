"""The valley crop counts: each valley's area of interest and its runs (FR-14).

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-28
"""

import sqlalchemy as sa
from alembic import op
from geoalchemy2 import Geometry
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "valleys",
        sa.Column("slug", sa.String(40), primary_key=True),
        sa.Column("geom", Geometry("MULTIPOLYGON", srid=4326, spatial_index=False), nullable=False),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_valleys_geom", "valleys", ["geom"], postgresql_using="gist")
    op.create_table(
        "valley_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "valley",
            sa.String(40),
            sa.ForeignKey("valleys.slug", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("run_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("newest_image", sa.Date()),
        sa.Column("area_km2", sa.Float(), nullable=False),
        sa.Column("farmland_ha", sa.Float(), nullable=False),
        sa.Column("credits", sa.Float(), nullable=False),
        sa.Column("downloads", sa.Integer(), nullable=False),
        sa.Column("seconds", sa.Float(), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("model_trained_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("corners", JSONB(), nullable=False),
        sa.Column("colors", JSONB(), nullable=False),
        sa.Column("counts", JSONB(), nullable=False),
    )
    op.create_index("ix_valley_runs_valley_run_at", "valley_runs", ["valley", "run_at"])


def downgrade() -> None:
    op.drop_table("valley_runs")
    op.drop_index("ix_valleys_geom", table_name="valleys")
    op.drop_table("valleys")
