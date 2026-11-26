"""Connections to PostGIS, migrations, and a look at the tables (FR-02).

The database URL comes from the settings (`DATABASE_URL`, default: the Docker database on
port 5433). The migrations live in `migrations/` next to this file and run with
`fundoscope db upgrade`.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from fundoscope.config import get_settings

MIGRATIONS = Path(__file__).parent / "migrations"
CONNECT_TIMEOUT_S = 5  # fail fast when the database isn't running


def make_engine(url: str) -> Engine:
    return create_engine(url, connect_args={"connect_timeout": CONNECT_TIMEOUT_S})


@lru_cache
def engine() -> Engine:
    """The engine of the configured database (one per process)."""
    return make_engine(get_settings().database_url)


@contextmanager
def session_scope(bind: Engine | None = None) -> Iterator[Session]:
    """A session that commits if the block succeeds and rolls back if it raises."""
    factory = sessionmaker(bind=bind or engine(), expire_on_commit=False)
    with factory() as session, session.begin():
        yield session


def upgrade(url: str | None = None) -> None:
    """Bring the database's tables up to the latest migration."""
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS))
    config.set_main_option(
        "sqlalchemy.url", (url or get_settings().database_url).replace("%", "%%")
    )
    command.upgrade(config, "head")


def describe(bind: Engine | None = None) -> list[tuple[str, int, list[str]]]:
    """Each table of the app with its row count and its spatial (GiST) indexes."""
    query = text(
        "SELECT t.tablename, coalesce(string_agg(i.indexname, ',') FILTER "
        "(WHERE i.indexdef ILIKE '%USING gist%'), '') "
        "FROM pg_tables t LEFT JOIN pg_indexes i ON i.tablename = t.tablename "
        "WHERE t.schemaname = 'public' "
        "AND t.tablename NOT IN ('spatial_ref_sys', 'alembic_version') "
        "GROUP BY t.tablename ORDER BY t.tablename"
    )
    with (bind or engine()).connect() as conn:
        tables = conn.execute(query).all()
        return [
            (
                name,
                conn.execute(text(f'SELECT count(*) FROM "{name}"')).scalar_one(),
                [i for i in gist.split(",") if i],
            )
            for name, gist in tables
        ]
