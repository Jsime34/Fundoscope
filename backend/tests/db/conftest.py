"""A real PostGIS database for the `db` tests: `fundoscope_test`, created fresh each session.

It lives on the server of `DATABASE_URL` (the Docker database by default). When that server
isn't running, these tests are skipped, unless FUNDOSCOPE_REQUIRE_DB is set (as in CI).
"""

import os

import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError

from fundoscope.config import get_settings
from fundoscope.db import (
    AutocommitParcelStore,
    MemoryJobStore,
    MemoryParcelStore,
    MemoryValleyStore,
    SqlJobStore,
    SqlParcelStore,
    SqlValleyStore,
    make_engine,
    session_scope,
    upgrade,
)

TEST_DATABASE = "fundoscope_test"


@pytest.fixture(scope="session")
def database_url() -> str:
    server = make_url(get_settings().database_url)
    admin = make_engine(server.render_as_string(hide_password=False)).execution_options(
        isolation_level="AUTOCOMMIT"
    )
    try:
        with admin.connect() as conn:
            conn.execute(text(f"DROP DATABASE IF EXISTS {TEST_DATABASE} WITH (FORCE)"))
            conn.execute(text(f"CREATE DATABASE {TEST_DATABASE}"))
    except OperationalError as exc:
        if os.environ.get("FUNDOSCOPE_REQUIRE_DB"):
            raise
        pytest.skip(f"PostGIS isn't running ({exc.orig}); start the database to run these tests.")
    finally:
        admin.dispose()
    url = server.set(database=TEST_DATABASE).render_as_string(hide_password=False)
    upgrade(url)
    return url


@pytest.fixture
def db_engine(database_url):
    engine = make_engine(database_url)
    yield engine
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE parcels, jobs, valleys CASCADE"))
    engine.dispose()


@pytest.fixture
def sql_store(db_engine):
    with session_scope(db_engine) as session:
        yield SqlParcelStore(session)


@pytest.fixture(params=["memory", "postgis", "postgis-autocommit"])
def any_store(request):
    """The same tests for every parcel store."""
    if request.param == "memory":
        return MemoryParcelStore()
    if request.param == "postgis":
        return request.getfixturevalue("sql_store")
    return AutocommitParcelStore(request.getfixturevalue("db_engine"))


@pytest.fixture(params=["memory", "postgis"])
def any_jobs(request):
    """The same tests for both job stores: (job store, parcel store for the parcels)."""
    if request.param == "memory":
        return MemoryJobStore(), MemoryParcelStore()
    engine = request.getfixturevalue("db_engine")
    return SqlJobStore(engine), AutocommitParcelStore(engine)


@pytest.fixture(params=["memory", "postgis"])
def any_valleys(request):
    """The same tests for both valley stores."""
    if request.param == "memory":
        return MemoryValleyStore()
    return SqlValleyStore(request.getfixturevalue("db_engine"))
