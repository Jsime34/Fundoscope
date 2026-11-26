"""`fundoscope db upgrade` and `fundoscope db tables`: the PostGIS database (FR-02)."""

from sqlalchemy.exc import OperationalError

from fundoscope.db import describe, engine, upgrade

START_HINT = (
    "Start it with: wsl docker compose -f /mnt/c/Fundoscope/Fundoscope/docker-compose.yml up -d db"
)


def run_db_upgrade() -> int:
    """Create or update the tables. Returns the exit code."""
    try:
        upgrade()
    except OperationalError as exc:
        print(f"Can't reach the database: {exc.orig}\n{START_HINT}")
        return 1
    print("The database is up to date.")
    return 0


def run_db_tables() -> int:
    """Print each table with its rows and spatial indexes. Returns the exit code."""
    try:
        tables = describe()
    except OperationalError as exc:
        print(f"Can't reach the database: {exc.orig}\n{START_HINT}")
        return 1
    finally:
        engine().dispose()
    if not tables:
        print("No tables yet: run `fundoscope db upgrade`.")
        return 1
    print(f"{'Table':<14}{'Rows':>8}   Spatial (GiST) indexes")
    for name, rows, gist in tables:
        print(f"{name:<14}{rows:>8,}   {', '.join(gist) or '-'}")
    return 0
