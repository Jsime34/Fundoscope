"""The valley crop counts in PostGIS (FR-14): each valley's area of interest and its runs.

`fundoscope valley refresh` saves one run per count: the hectares of every crop class with
their 95 % ranges, for this campaign so far and the earlier ones, the dates, the cost, and
what the dashboard's map needs (the images' corners, colors, and files). The dashboard
(FR-15) shows a valley's latest run, and the neighbor comparison (FR-08) finds the valley that
holds a parcel through the GiST index on the areas. The rasters stay in files
(data/valley/<slug>/, see `pipeline.valley_store`).

Like the job store, every call of `SqlValleyStore` runs in its own short transaction.
"""

from dataclasses import dataclass, fields
from datetime import UTC, date, datetime
from typing import Any, Protocol

from geoalchemy2.shape import from_shape
from shapely.geometry import MultiPolygon, Point
from shapely.geometry.base import BaseGeometry
from sqlalchemy import Engine, func, select

from fundoscope.db.models import ValleyRow, ValleyRunRow
from fundoscope.db.session import session_scope

SRID = 4326


@dataclass(frozen=True)
class StoredValleyRun:
    """One count of a valley's crop hectares.

    Attributes:
        valley: the valley's slug, e.g. "majes" (its folder in data/valley).
        run_at: when the count was made.
        as_of: the last day of images the count could use.
        newest_image: the day of the newest image over the valley (None if unknown).
        area_km2: the area of interest counted, in km².
        farmland_ha: the farmland inside it, in hectares.
        credits: openEO credits the run spent (FR-25).
        downloads: openEO requests the run made.
        seconds: how long the run took.
        model_version: the crop model's version (from its model card).
        model_trained_at: when that model was trained.
        corners: the map images' corners in WGS84 lon/lat: north-west, north-east, south-east,
            south-west.
        colors: each class's color on the map ("#rrggbb").
        counts: one per count, the current campaign so far first, as JSON objects with `key`,
            `campaign`, `months`, `through` (ISO date), `in_progress`, `model` (file), `map`
            (class map file), `images` (map image files by class, and "farmland"), and
            `classes`: `name`, `pixels`, `hectares`, `adjusted`, `low`, `high` (the 95 % range
            of the adjusted hectares; None for classes made by rules). Files are relative to
            the valley's folder.
    """

    valley: str
    run_at: datetime
    as_of: date
    newest_image: date | None
    area_km2: float
    farmland_ha: float
    credits: float
    downloads: int
    seconds: float
    model_version: str
    model_trained_at: datetime
    corners: list[list[float]]
    colors: dict[str, str]
    counts: list[dict[str, Any]]


class ValleyStore(Protocol):
    """Where the valleys' counts are kept."""

    def save_run(self, run: StoredValleyRun, area: BaseGeometry) -> None:
        """Keep a run, and `area` (WGS84) as its valley's area of interest from now on."""
        ...

    def latest_run(self, slug: str) -> StoredValleyRun | None:
        """The valley's newest run (None if it was never counted)."""
        ...

    def latest_run_at(self, point: Point) -> StoredValleyRun | None:
        """The newest run of the valley whose area of interest holds `point` (WGS84)."""
        ...


def as_multipolygon(area: BaseGeometry) -> MultiPolygon:
    """The area as a MultiPolygon (the column's type)."""
    if isinstance(area, MultiPolygon):
        return area
    if area.geom_type == "Polygon":
        return MultiPolygon([area])
    raise ValueError(f"A valley's area of interest must be a polygon, not a {area.geom_type}.")


class SqlValleyStore:
    """Valleys and their runs in the `valleys` and `valley_runs` tables."""

    def __init__(self, bind: Engine | None = None) -> None:
        self.bind = bind

    def save_run(self, run: StoredValleyRun, area: BaseGeometry) -> None:
        geom = from_shape(as_multipolygon(area), srid=SRID)
        with session_scope(self.bind) as session:
            valley = session.get(ValleyRow, run.valley)
            if valley is None:
                session.add(ValleyRow(slug=run.valley, geom=geom))
            else:
                valley.geom, valley.updated_at = geom, datetime.now(UTC)
            session.flush()  # the valley before its run (foreign key)
            session.add(ValleyRunRow(**_columns(run)))

    def latest_run(self, slug: str) -> StoredValleyRun | None:
        with session_scope(self.bind) as session:
            row = session.scalars(
                select(ValleyRunRow)
                .where(ValleyRunRow.valley == slug)
                .order_by(ValleyRunRow.run_at.desc())
                .limit(1)
            ).first()
            return _run(row) if row else None

    def latest_run_at(self, point: Point) -> StoredValleyRun | None:
        with session_scope(self.bind) as session:
            row = session.scalars(
                select(ValleyRunRow)
                .join(ValleyRow, ValleyRow.slug == ValleyRunRow.valley)
                .where(func.ST_Contains(ValleyRow.geom, from_shape(point, srid=SRID)))
                .order_by(ValleyRunRow.run_at.desc())
                .limit(1)
            ).first()
            return _run(row) if row else None


_FIELDS = [field.name for field in fields(StoredValleyRun)]  # the same names as the columns


def _columns(run: StoredValleyRun) -> dict[str, Any]:
    return {name: getattr(run, name) for name in _FIELDS}


def _run(row: ValleyRunRow) -> StoredValleyRun:
    return StoredValleyRun(**{name: getattr(row, name) for name in _FIELDS})
