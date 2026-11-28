"""Migrations, transactions, and the table overview, on the test database."""

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text

from fundoscope.db import SqlParcelStore, describe, engine, session_scope
from fundoscope.db.session import MIGRATIONS


def test_the_tables_and_their_spatial_index(db_engine):
    tables = {name: (rows, gist) for name, rows, gist in describe(db_engine)}

    assert set(tables) == {
        "campaigns",
        "jobs",
        "observations",
        "parcels",
        "valley_runs",
        "valleys",
    }
    assert tables["parcels"] == (0, ["ix_parcels_geom"])
    assert tables["valleys"] == (0, ["ix_valleys_geom"])
    spatial = {"parcels", "valleys"}
    assert all(gist == [] for name, (_, gist) in tables.items() if name not in spatial)


def test_a_failing_block_saves_nothing(db_engine):
    from shapely.geometry import MultiPolygon, box

    square = MultiPolygon([box(-75.731, -14.071, -75.729, -14.069)])
    with pytest.raises(RuntimeError), session_scope(db_engine) as session:
        SqlParcelStore(session).add("abc", None, square, 32718, 4.6, False)
        raise RuntimeError("the download failed")

    with session_scope(db_engine) as session:
        assert SqlParcelStore(session).parcels() == []


def test_the_migrations_go_down_and_up_again(database_url, db_engine):
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))

    command.downgrade(config, "base")
    with db_engine.connect() as conn:
        assert conn.execute(text("SELECT to_regclass('public.parcels')")).scalar() is None
    command.upgrade(config, "head")

    assert {name for name, _, _ in describe(db_engine)} == {
        "campaigns",
        "jobs",
        "observations",
        "parcels",
        "valley_runs",
        "valleys",
    }


def test_one_engine_per_process():
    assert engine() is engine()
