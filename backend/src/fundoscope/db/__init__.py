"""Database layer: PostGIS tables, migrations, the parcel store (FR-02), and jobs (FR-13).

Submodules: `records` (what is stored, as plain objects, and the `ParcelStore` protocol),
`models` (the tables), `parcels` (`SqlParcelStore` and `AutocommitParcelStore`, the store
in PostGIS), `jobs` (background jobs: `Job`, the `JobStore` protocol, `SqlJobStore`),
`memory` (`MemoryParcelStore` and `MemoryJobStore`, for tests and runs without a
database), `session` (connections, migrations, and a look at the tables).
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
from fundoscope.db.memory import MemoryJobStore, MemoryParcelStore
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
    "ParcelStore",
    "SqlJobStore",
    "SqlParcelStore",
    "StoredCampaign",
    "StoredHistory",
    "StoredParcel",
    "describe",
    "empty_observations",
    "engine",
    "make_engine",
    "session_scope",
    "upgrade",
]
