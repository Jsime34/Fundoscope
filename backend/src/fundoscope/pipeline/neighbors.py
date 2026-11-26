"""The neighbor comparison of a parcel, from a counted valley's saved data (FR-08).

The parcel must lie in a valley that `fundoscope valley refresh` has counted: its centroid
inside the valley's area of interest. The farmland is the latest count's class map (green in
one of the last three campaigns, minus towns and water), and the monthly NDVI comes from the
composites the refresh keeps on disk, so the comparison makes no satellite request. It covers
the campaigns of the latest count: the current one so far and the two before it.
"""

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import shapely
import xarray as xr
from numpy.typing import NDArray
from shapely.geometry.base import BaseGeometry

from fundoscope.analysis.neighbors import (
    MIN_NEIGHBOR_PIXELS,
    OUTER_M,
    NeighborBand,
    campaign_ndvi,
    compare,
    neighbor_ring,
)
from fundoscope.ingest import Parcel, ParcelError, from_utm, to_utm
from fundoscope.pipeline.valley_store import (
    COMPOSITES_DIR,
    LATEST_FILE,
    ValleyError,
    load_month,
    load_valley,
)
from fundoscope.timeseries import ndvi
from fundoscope.valley import PIXEL_M, Tile, Valley, snap_bounds
from fundoscope.valley.season import Count, MonthKey

OUTSIDE = "The neighbor comparison is only available inside the pilot area (the Pampa de Majes)."
TOO_FEW = "There are too few farmland pixels around the parcel to compare it with."


@dataclass(frozen=True)
class NeighborReport:
    """The neighbor comparison of a parcel.

    Attributes:
        valley: the slug of the valley that holds the parcel; None outside every valley.
        bands: one per campaign compared, oldest first.
        ring: the 50 m-2 km ring in WGS84 longitude and latitude (None outside).
        note: why there are no bands, if there are none.
    """

    valley: str | None
    bands: list[NeighborBand] = field(default_factory=list)
    ring: BaseGeometry | None = None
    note: str = ""


def neighbor_report(
    parcel: Parcel, valleys: Path, min_neighbors: int = MIN_NEIGHBOR_PIXELS
) -> NeighborReport:
    """Compare each campaign of the parcel with the farmland 50 m to 2 km around it.

    Args:
        parcel: a validated parcel.
        valleys: the folder of the counted valleys (data/valley).
    """
    found = find_valley(parcel, valleys)
    if found is None:
        return NeighborReport(None, note=OUTSIDE)
    valley_, folder, run = found
    parcel_utm = to_utm(parcel.geometry, valley_.epsg)
    ring = neighbor_ring(parcel_utm)
    window = _window(parcel_utm.buffer(OUTER_M).bounds, valley_.bounds)
    xx, yy = np.meshgrid(window.x, window.y)
    own = shapely.contains_xy(parcel_utm, xx, yy)
    around = shapely.contains_xy(ring, xx, yy) & _farmland(folder / run["counts"][0]["map"], window)
    bands = []
    for count in campaign_counts(run):
        values = campaign_ndvi(window_ndvi(folder, valley_, window, count.month_keys()))
        band = compare(count.campaign, count.months, values[own], values[around], min_neighbors)
        if band is not None:
            bands.append(band)
    return NeighborReport(
        valley_.slug, bands, from_utm(ring, valley_.epsg), "" if bands else TOO_FEW
    )


def find_valley(parcel: Parcel, valleys: Path) -> tuple[Valley, Path, dict[str, Any]] | None:
    """The counted valley whose area of interest holds the parcel's centroid, if any."""
    if not valleys.is_dir():
        return None
    for folder in sorted(path for path in valleys.iterdir() if (path / LATEST_FILE).is_file()):
        try:
            valley_ = load_valley(folder, folder.name)
        except (ValleyError, ParcelError):
            continue  # a valley being set up: no comparison from it
        if valley_.area_utm.contains(to_utm(parcel.geometry.centroid, valley_.epsg)):
            return valley_, folder, json.loads((folder / LATEST_FILE).read_text(encoding="utf-8"))
    return None


def campaign_counts(run: dict[str, Any]) -> list[Count]:
    """The campaigns to compare: the latest count's first one (the current campaign so far)
    and the whole campaigns before it, oldest first."""
    counts = [
        Count(item["campaign"], item["months"], date.fromisoformat(item["through"]))
        for item in run["counts"]
    ]
    current, rest = counts[0], counts[1:]
    earlier = [c for c in rest if not c.in_progress and c.campaign != current.campaign]
    return sorted([current, *earlier], key=lambda count: count.campaign)


def window_ndvi(folder: Path, valley_: Valley, window: Tile, keys: list[MonthKey]) -> NDArray:
    """Monthly NDVI (month, y, x) on the window's grid, read from the tiles it overlaps."""
    out = np.full((len(keys), len(window.y), len(window.x)), np.nan)
    west, south, east, north = window.bounds
    for tile in valley_.tiles:
        t_west, t_south, t_east, t_north = tile.bounds
        o_west, o_east = max(west, t_west), min(east, t_east)
        o_south, o_north = max(south, t_south), min(north, t_north)
        if o_west >= o_east or o_south >= o_north:
            continue
        tile_rows = slice(_px(t_north - o_north), _px(t_north - o_south))
        tile_cols = slice(_px(o_west - t_west), _px(o_east - t_west))
        rows = slice(_px(north - o_north), _px(north - o_south))
        cols = slice(_px(o_west - west), _px(o_east - west))
        for i, key in enumerate(keys):
            path = folder / COMPOSITES_DIR / tile.id / key.file_name
            red, nir = load_month(path, ("B04", "B08")).to_numpy()
            out[i, rows, cols] = ndvi(red[tile_rows, tile_cols], nir[tile_rows, tile_cols])
    return out


def _px(meters: float) -> int:
    return round(meters / PIXEL_M)


def _window(
    bounds: tuple[float, float, float, float], grid: tuple[float, float, float, float]
) -> Tile:
    """The part of the valley grid within `bounds`, on its pixel grid."""
    west, south, east, north = snap_bounds(bounds)
    return Tile(
        "window",
        (max(west, grid[0]), max(south, grid[1]), min(east, grid[2]), min(north, grid[3])),
    )


def _farmland(map_path: Path, window: Tile) -> NDArray[np.bool_]:
    with xr.open_dataset(map_path, engine="h5netcdf") as saved:
        codes = saved["crop_class"].reindex(x=window.x, y=window.y, fill_value=0).to_numpy()
    return codes != 0
