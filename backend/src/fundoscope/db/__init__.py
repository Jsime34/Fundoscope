"""Database layer: PostGIS tables, migrations, and the parcel store (FR-02).

Submodules: `records` (what is stored, as plain objects, and the `ParcelStore` protocol),
`models` (the tables), `parcels` (`SqlParcelStore`, the store in PostGIS), `memory`
(`MemoryParcelStore`, for tests and runs without a database), `session` (connections,
migrations, and a look at the tables).
"""

from fundoscope.db.memory import MemoryParcelStore
from fundoscope.db.parcels import SqlParcelStore
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
    "OBSERVATION_COLUMNS",
    "MemoryParcelStore",
    "ParcelStore",
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
