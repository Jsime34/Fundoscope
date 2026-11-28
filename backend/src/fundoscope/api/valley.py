"""The valley crop count for the dashboard (FR-15).

GET /api/valleys/{slug} returns the latest count saved by `fundoscope valley refresh <slug>`
(in PostGIS, `valley_runs`); GET /api/valleys/{slug}/maps/{count}/{image}.png returns the
images of its map (files in data/valley/<slug>/maps). Nothing here calls Copernicus: the
refresh does the work.
"""

from datetime import date, datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi import Path as PathParam
from fastapi.responses import FileResponse
from pydantic import BaseModel

from fundoscope.config import get_settings
from fundoscope.db import SqlValleyStore, StoredValleyRun, ValleyStore
from fundoscope.pipeline.valley_store import MAPS_DIR, CountedValleys

NAME = r"^[a-z0-9][a-z0-9_-]*$"  # slugs, count keys, and image names: no paths
IMAGE_MAX_AGE_S = 3600

router = APIRouter(prefix="/api/valleys", tags=["valley crop monitor"])


class ClassOut(BaseModel):
    name: str
    pixels: int
    hectares: float
    adjusted: float | None  # error-adjusted hectares (Olofsson et al., 2014)
    low: float | None  # 95 % range of the adjusted hectares
    high: float | None
    color: str
    image: str  # URL of the class's map image


class CountOut(BaseModel):
    key: str
    campaign: str
    months: int
    through: date
    in_progress: bool
    farmland_image: str  # URL
    classes: list[ClassOut]


class ValleyOut(BaseModel):
    slug: str
    updated_at: datetime
    as_of: date
    newest_image: date | None
    area_km2: float
    farmland_ha: float
    corners: list[list[float]]  # lon/lat: north-west, north-east, south-east, south-west
    counts: list[CountOut]  # this campaign so far, the previous one at the same date, ...


def valleys_folder() -> Path:
    """data/valley (tests point it elsewhere)."""
    return get_settings().data_dir / "valley"


def valley_store() -> ValleyStore:
    """The valleys' counts in PostGIS (tests override this with an in-memory store)."""
    return SqlValleyStore()


def counted_valleys(
    folder: Annotated[Path, Depends(valleys_folder)],
    store: Annotated[ValleyStore, Depends(valley_store)],
) -> CountedValleys:
    """The valleys' files and counts, for the neighbor comparison (FR-08)."""
    return CountedValleys(folder, store)


@router.get(
    "/{slug}",
    response_model=ValleyOut,
    responses={404: {"description": "The valley has no count yet"}},
)
def get_valley(
    slug: Annotated[str, PathParam(pattern=NAME)],
    store: Annotated[ValleyStore, Depends(valley_store)],
) -> ValleyOut:
    """The hectares of each crop class, this campaign and the previous ones, with the map."""
    run = store.latest_run(slug)
    if run is None:
        raise HTTPException(404, f"The valley {slug!r} has no count yet.")
    return valley_out(run)


@router.get(
    "/{slug}/maps/{count}/{image}.png",
    response_class=FileResponse,
    responses={404: {"description": "No such image"}},
)
def get_map_image(
    slug: Annotated[str, PathParam(pattern=NAME)],
    count: Annotated[str, PathParam(pattern=NAME)],
    image: Annotated[str, PathParam(pattern=NAME)],
    folder: Annotated[Path, Depends(valleys_folder)],
) -> FileResponse:
    """One map image: the farmland, or one class in its color (transparent PNG)."""
    path = folder / slug / MAPS_DIR / count / f"{image}.png"
    if not path.is_file():
        raise HTTPException(404, "No such image.")
    return FileResponse(
        path,
        media_type="image/png",
        headers={"Cache-Control": f"public, max-age={IMAGE_MAX_AGE_S}"},
    )


def valley_out(run: StoredValleyRun) -> ValleyOut:
    """The API shape of a saved run: image paths become URLs."""

    def url(relative: str) -> str:
        return f"/api/valleys/{run.valley}/{relative}"

    counts = [
        CountOut(
            key=count["key"],
            campaign=count["campaign"],
            months=count["months"],
            through=count["through"],
            in_progress=count["in_progress"],
            farmland_image=url(count["images"]["farmland"]),
            classes=[
                ClassOut(
                    **item,
                    color=run.colors.get(item["name"], "#52514e"),
                    image=url(count["images"][item["name"]]),
                )
                for item in count["classes"]
            ],
        )
        for count in run.counts
    ]
    return ValleyOut(
        slug=run.valley,
        updated_at=run.run_at,
        as_of=run.as_of,
        newest_image=run.newest_image,
        area_km2=run.area_km2,
        farmland_ha=run.farmland_ha,
        corners=run.corners,
        counts=counts,
    )
