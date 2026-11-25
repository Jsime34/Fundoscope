"""The valley's area of interest, cut into tiles small enough for one openEO request (FR-14).

The valley is a polygon in UTM meters. Its bounding box is snapped outward to the 10 m pixel
grid and cut into a grid of equal tiles, at most 9 km on a side (about 70 km²): six months of
the 10 bands of one tile take about four minutes and 7 openEO credits. Tile ids name the row
(from the north) and the column (from the west): "r0c0" is the north-west corner.

Once months are downloaded for a grid, the grid must stay put: a valley can be given the grid
bounds of an earlier run, and its polygon may then change as long as it stays inside them.

Pixel centers lie half a pixel inside the grid lines, so each pixel belongs to exactly one
tile, and it belongs to the valley when its center is inside the polygon.
"""

import math
from dataclasses import dataclass

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

PIXEL_M = 10
PIXEL_HA = PIXEL_M * PIXEL_M / 10_000
MAX_TILE_M = 9_000


@dataclass(frozen=True)
class Tile:
    """One tile of the valley grid.

    Attributes:
        id: "r<row>c<column>", row 0 in the north and column 0 in the west.
        bounds: west, south, east, north in UTM meters, on the 10 m grid.
    """

    id: str
    bounds: tuple[float, float, float, float]

    @property
    def geometry(self) -> BaseGeometry:
        return box(*self.bounds)

    @property
    def x(self) -> NDArray[np.float64]:
        """Pixel-center x coordinates, west to east."""
        west, _, east, _ = self.bounds
        return west + PIXEL_M / 2 + PIXEL_M * np.arange(round((east - west) / PIXEL_M))

    @property
    def y(self) -> NDArray[np.float64]:
        """Pixel-center y coordinates, north to south (the order of image rows)."""
        _, south, _, north = self.bounds
        return north - PIXEL_M / 2 - PIXEL_M * np.arange(round((north - south) / PIXEL_M))


@dataclass(frozen=True)
class Valley:
    """A valley to count: its polygon and its tiles, in UTM.

    Attributes:
        slug: short name used in paths and URLs ("majes").
        area_utm: the area of interest, a polygon in UTM meters.
        epsg: the UTM zone's EPSG code.
        bounds: west, south, east, north of the tile grid, on the 10 m grid.
        max_tile_m: the longest side a tile may have.
        tiles: the tiles covering `bounds`, row by row from the north-west.
    """

    slug: str
    area_utm: BaseGeometry
    epsg: int
    bounds: tuple[float, float, float, float]
    max_tile_m: float
    tiles: tuple[Tile, ...]

    @property
    def area_km2(self) -> float:
        return self.area_utm.area / 1e6

    def inside(self, tile: Tile) -> NDArray[np.bool_]:
        """(y, x) mask of a tile: True where the pixel center lies inside the valley."""
        xx, yy = np.meshgrid(tile.x, tile.y)
        return shapely.contains_xy(self.area_utm, xx, yy)


def valley(
    slug: str,
    area_utm: BaseGeometry,
    epsg: int,
    max_tile_m: float = MAX_TILE_M,
    bounds: tuple[float, float, float, float] | None = None,
) -> Valley:
    """The valley `slug` with its tile grid.

    Args:
        bounds: the grid of an earlier run; by default, the polygon's bounds snapped to the
            pixel grid.

    Raises:
        ValueError: the area is empty, or it goes beyond `bounds`.
    """
    if area_utm.is_empty or area_utm.area == 0:
        raise ValueError(f"The area of interest of {slug!r} is empty.")
    west, south, east, north = snap_bounds(area_utm.bounds) if bounds is None else bounds
    grid = (west, south, east, north)
    if not box(*grid).covers(area_utm):
        raise ValueError(f"The area of interest of {slug!r} goes beyond its tile grid {grid}.")
    return Valley(slug, area_utm, epsg, grid, max_tile_m, tile_grid(grid, max_tile_m))


def snap_bounds(bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Bounds grown outward to multiples of the pixel size."""
    west, south, east, north = bounds
    return (
        math.floor(west / PIXEL_M) * PIXEL_M,
        math.floor(south / PIXEL_M) * PIXEL_M,
        math.ceil(east / PIXEL_M) * PIXEL_M,
        math.ceil(north / PIXEL_M) * PIXEL_M,
    )


def tile_grid(
    bounds: tuple[float, float, float, float], max_tile_m: float = MAX_TILE_M
) -> tuple[Tile, ...]:
    """Equal tiles at most `max_tile_m` on a side covering snapped `bounds`."""
    west, south, east, north = bounds
    xs = _edges(west, east, max_tile_m)
    ys = _edges(north, south, max_tile_m)  # from the north, like image rows
    return tuple(
        Tile(f"r{row}c{col}", (xs[col], ys[row + 1], xs[col + 1], ys[row]))
        for row in range(len(ys) - 1)
        for col in range(len(xs) - 1)
    )


def _edges(start: float, stop: float, max_size: float) -> list[float]:
    """Grid lines from `start` to `stop` (either direction) on the pixel grid."""
    pixels = round(abs(stop - start) / PIXEL_M)
    count = math.ceil(pixels * PIXEL_M / max_size)
    step = 1 if stop > start else -1
    return [start + step * PIXEL_M * round(pixels * i / count) for i in range(count + 1)]
