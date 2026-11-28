"""Database layer: PostGIS tables, migrations, the parcel store (FR-02), jobs (FR-13), and the
valley counts (FR-14).

Submodules: `records` (what is stored, as plain objects, and the `ParcelStore` protocol),
`models` (the tables), `parcels` (`SqlParcelStore` and `AutocommitParcelStore`, the store
in PostGIS), `jobs` (background jobs: `Job`, the `JobStore` protocol, `SqlJobStore`),
`valleys` (the valleys' areas and counts: `StoredValleyRun`, the `ValleyStore` protocol,
`SqlValleyStore`), `memory` (`MemoryParcelStore`, `MemoryJobStore`, and `MemoryValleyStore`,
for tests and runs without a database), `session` (connections, migrations, and a look at
the tables).
"""

from fundoscope.db.jobs import (
    ANALYZE,
    DONE,
    FAILED,
    QUEUED,
    RUNNING,
    Job,
    JobStore,
    SqlJobStore,
)
from fundoscope.db.memory import MemoryJobStore, MemoryParcelStore, MemoryValleyStore
from fundoscope.db.parcels import AutocommitParcelStore, SqlParcelStore
from fundoscope.db.records import (
    OBSERVATION_COLUMNS,
    ParcelStore,
    StoredCampaign,
    StoredHistory,
    StoredParcel,
    empty_observations,
)
from fundoscope.db.session import describe, engine, make_engine, session_scope, upgrade
from fundoscope.db.valleys import SqlValleyStore, StoredValleyRun, ValleyStore

__all__ = [
    "ANALYZE",
    "DONE",
    "FAILED",
    "OBSERVATION_COLUMNS",
    "QUEUED",
    "RUNNING",
    "AutocommitParcelStore",
    "Job",
    "JobStore",
    "MemoryJobStore",
    "MemoryParcelStore",
    "MemoryValleyStore",
    "ParcelStore",
    "SqlJobStore",
    "SqlParcelStore",
    "SqlValleyStore",
    "StoredCampaign",
    "StoredHistory",
    "StoredParcel",
    "StoredValleyRun",
    "ValleyStore",
    "describe",
    "empty_observations",
    "engine",
    "make_engine",
    "session_scope",
    "upgrade",
]
