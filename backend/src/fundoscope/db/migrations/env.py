"""Alembic environment: runs the migrations against the URL set by `db.session.upgrade`."""

from alembic import context
from sqlalchemy import engine_from_config, pool

from fundoscope.db.models import Base

config = context.config
connectable = engine_from_config(
    config.get_section(config.config_ini_section, {}),
    prefix="sqlalchemy.",
    poolclass=pool.NullPool,
    connect_args={"connect_timeout": 5},
)
with connectable.connect() as connection:
    context.configure(connection=connection, target_metadata=Base.metadata)
    with context.begin_transaction():
        context.run_migrations()
