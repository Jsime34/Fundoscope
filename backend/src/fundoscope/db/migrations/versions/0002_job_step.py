"""What a background job is doing now, in plain language (FR-13).

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-26
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("step", sa.Text()))
    op.create_index("ix_jobs_status", "jobs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_jobs_status", table_name="jobs")
    op.drop_column("jobs", "step")
