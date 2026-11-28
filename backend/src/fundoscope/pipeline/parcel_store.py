"""Store, update, and reopen parcels (FR-02).

A parcel is stored when it is uploaded, keyed by its outline: the same outline uploaded
again, in any file format, is the same parcel. Updating it downloads only the dates after
the last stored one (and the composites that aren't stored yet); reopening it rebuilds the
report from the store alone, with no satellite request. The API runs updates as background
jobs (`pipeline.jobs`, FR-13).
"""

import hashlib
from dataclasses import dataclass
from datetime import date
from uuid import UUID

import shapely

from fundoscope.db.records import ParcelStore, StoredParcel
from fundoscope.ingest import Parcel, parse_parcel, to_utm
from fundoscope.ml import CropModel
from fundoscope.pipeline.parcel_history import (
    CompositeFetcher,
    Fetcher,
    ParcelHistory,
    Progress,
    history_of,
)
from fundoscope.pipeline.valley_store import CountedValleys

KEY_PRECISION_DEG = 1e-7  # about 1 cm: rounding noise from file formats doesn't matter


@dataclass(frozen=True)
class StoredAnalysis:
    """A parcel's history and where it is stored.

    Attributes:
        parcel_id: the parcel's id in the store.
        history: the history (its costs are those of this run only).
    """

    parcel_id: UUID
    history: ParcelHistory


def parcel_key(parcel: Parcel) -> str:
    """A hash of the parcel's outline (WGS84, rounded to about 1 cm, normalized)."""
    shape = shapely.normalize(shapely.set_precision(parcel.geometry, KEY_PRECISION_DEG))
    return hashlib.sha256(shapely.to_wkb(shape)).hexdigest()


def store_parcel(data: bytes, filename: str, store: ParcelStore) -> StoredParcel:
    """Validate an uploaded parcel and store it, unless the same outline is stored already.

    Nothing is downloaded: `update_parcel` brings its history up to date.

    Raises:
        ingest.ParcelError: the file isn't a usable parcel (nothing is stored then).
    """
    parcel = parse_parcel(data, filename)
    key = parcel_key(parcel)
    return store.find(key) or store.add(
        key, parcel.name, parcel.geometry, parcel.utm_epsg, parcel.area_ha, parcel.repaired
    )


def update_parcel(
    parcel_id: UUID,
    store: ParcelStore,
    *,
    today: date,
    fetch: Fetcher,
    fetch_composites: CompositeFetcher | None = None,
    model: CropModel | None = None,
    model_version: str | None = None,
    valleys: CountedValleys | None = None,
    on_progress: Progress | None = None,
) -> StoredAnalysis:
    """Bring a stored parcel's history up to `today`, and save what is new.

    Only the dates after the stored ones are downloaded (see `history_of`, which also says
    when `on_progress` is called).

    Raises:
        LookupError: there is no such parcel.
        sentinel.SentinelError: the daily statistics could not be downloaded (nothing new
            is saved then).
    """
    stored_parcel = store.get(parcel_id)
    if stored_parcel is None:
        raise LookupError(f"There is no stored parcel {parcel_id}.")
    known = store.load(parcel_id)
    history = history_of(
        as_parcel(stored_parcel),
        end=today,
        fetch=fetch,
        fetch_composites=fetch_composites,
        model=model,
        model_version=model_version,
        valleys=valleys,
        stored=known,
        on_progress=on_progress,
    )
    observations = history.daily.observations
    if known.observed_through is not None:
        observations = observations[observations.index.date > known.observed_through]
    through = max(today, known.observed_through) if known.observed_through else today
    store.save(parcel_id, observations, through, history.campaign_results)
    return StoredAnalysis(parcel_id, history)


def reopen_parcel(
    parcel_id: UUID, store: ParcelStore, valleys: CountedValleys | None = None
) -> StoredAnalysis | None:
    """A stored parcel's history, rebuilt from the store with no satellite request.

    Returns:
        None if there is no such parcel, or it was never analyzed.
    """
    stored_parcel = store.get(parcel_id)
    if stored_parcel is None or stored_parcel.observed_through is None:
        return None
    history = history_of(
        as_parcel(stored_parcel),
        end=stored_parcel.observed_through,
        fetch=None,
        fetch_composites=None,
        valleys=valleys,
        stored=store.load(parcel_id),
    )
    return StoredAnalysis(parcel_id, history)


def as_parcel(stored: StoredParcel) -> Parcel:
    """The validated parcel of a stored one."""
    return Parcel(
        name=stored.name,
        geometry=stored.geometry,
        utm_epsg=stored.utm_epsg,
        geometry_utm=to_utm(stored.geometry, stored.utm_epsg),
        area_ha=stored.area_ha,
        repaired=stored.repaired,
    )
