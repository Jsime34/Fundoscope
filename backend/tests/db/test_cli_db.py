"""`fundoscope db upgrade` and `fundoscope db tables` on the test database."""

import pytest

from fundoscope.cli import main
from fundoscope.config import get_settings
from fundoscope.db import engine


@pytest.fixture
def test_database(monkeypatch, database_url):
    monkeypatch.setattr(get_settings(), "database_url", database_url)
    engine.cache_clear()
    yield
    engine().dispose()  # close its pooled connections before dropping it
    engine.cache_clear()


def test_upgrade_and_list_the_tables(test_database, capsys):
    assert main(["db", "upgrade"]) == 0
    assert main(["db", "tables"]) == 0

    out = capsys.readouterr().out
    assert "The database is up to date." in out
    assert "parcels" in out and "ix_parcels_geom" in out and "observations" in out
    assert "valley_runs" in out and "ix_valleys_geom" in out


def test_the_api_store_uses_one_transaction_per_request(test_database):
    from contextlib import closing

    from fundoscope.api.history import parcel_store

    with closing(parcel_store()) as requests:
        store = next(requests)
        assert store.parcels() == []
