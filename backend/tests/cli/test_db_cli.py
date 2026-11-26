"""`fundoscope db` when the database can't be reached (no database needed)."""

import pytest
from sqlalchemy import text

from fundoscope.cli.db import run_db_tables, run_db_upgrade
from fundoscope.config import get_settings
from fundoscope.db import engine

NOWHERE = "postgresql+psycopg://user:pass@127.0.0.1:1/none"  # port 1: refused at once


@pytest.fixture
def unreachable(monkeypatch):
    monkeypatch.setattr(get_settings(), "database_url", NOWHERE)
    engine.cache_clear()
    yield
    engine().dispose()  # close its pooled connections before dropping it
    engine.cache_clear()


def test_upgrade_says_how_to_start_the_database(unreachable, capsys):
    assert run_db_upgrade() == 1
    out = capsys.readouterr().out
    assert "Can't reach the database" in out and "docker compose" in out


def test_tables_says_how_to_start_the_database(unreachable, capsys):
    assert run_db_tables() == 1
    assert "Can't reach the database" in capsys.readouterr().out


def test_an_empty_database_has_no_tables_yet(monkeypatch, capsys):
    import fundoscope.cli.db as db_cli

    monkeypatch.setattr(db_cli, "describe", lambda: [])

    assert run_db_tables() == 1
    assert "No tables yet" in capsys.readouterr().out
    assert text  # sqlalchemy is importable without a database
