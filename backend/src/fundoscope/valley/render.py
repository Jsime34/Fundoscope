"""Images of a valley's class map for the dashboard (FR-15).

The dashboard shows the farmland in a light wash and one class at a time in its color, over
the satellite basemap. On a map every class touches every other, and no set of more than
three colors stays apart for color-blind readers, so the classes take turns instead of
sharing one image. Each class has a fixed color (it follows the class, never its rank), from
the categorical palette of the web app in slot order. "Planted, not yet identifiable" is
striped, so color isn't its only cue.

Each image is a transparent PNG on the valley's UTM pixel grid, with 2 or 3 palette colors,
so it compresses to a few hundred kilobytes. `corners` gives the longitude and latitude of
its four corners, where the map places it.
"""

import io
import re

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from shapely.geometry import MultiPoint

from fundoscope.ingest import from_utm
from fundoscope.valley.classify import NO_CLEAR_IMAGES, NOT_FARMLAND, PLANTED

# Categorical slots 1-8 of the web app's palette (validated for color-blind readers).
SLOT_COLORS = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
CLASS_SLOTS = {  # fixed: alfalfa blue, other annual orange, avocado aqua, ...
    "alfalfa": 0,
    "other annual crops": 1,
    "avocado": 2,
    "fallow/bare": 3,
    "grape": 4,
    PLANTED: 5,
}
NO_DATA_COLOR = "#52514e"
FARMLAND_COLOR = (255, 255, 255, 96)  # a light wash over the satellite image
CLASS_ALPHA = 235
STRIPE_PX = 3  # stripes of 3 pixels (30 m), at 45 degrees
FARMLAND = "farmland"


def class_color(name: str, classes: tuple[str, ...] = ()) -> str:
    """The color of a class; classes the palette doesn't name take the free slots in order."""
    if name in CLASS_SLOTS:
        return SLOT_COLORS[CLASS_SLOTS[name]]
    if name == NO_CLEAR_IMAGES:
        return NO_DATA_COLOR
    others = [c for c in classes if c not in CLASS_SLOTS and c != NO_CLEAR_IMAGES]
    free = SLOT_COLORS[len(CLASS_SLOTS) :]
    index = others.index(name) if name in others else len(free)
    return free[index] if index < len(free) else NO_DATA_COLOR


def image_name(name: str) -> str:
    """A file name for a class: "other annual crops" -> "other-annual-crops"."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def class_png(codes: NDArray[np.uint8], code: int, color: str, *, striped: bool = False) -> bytes:
    """The pixels of one class (`code`) in `color`; everything else transparent."""
    rgb = _rgb(color)
    mine = codes == code
    indices = mine.astype(np.uint8)
    palette = [(0, 0, 0, 0), (*rgb, CLASS_ALPHA)]
    if striped:
        rows, cols = np.indices(codes.shape)
        light = ((rows + cols) // STRIPE_PX) % 2 == 1
        indices[mine & light] = 2
        palette.append((*(round(c + (255 - c) * 0.6) for c in rgb), CLASS_ALPHA))
    return _png(indices, palette)


def farmland_png(codes: NDArray[np.uint8]) -> bytes:
    """Every farmland pixel in a light wash; the rest transparent."""
    return _png((codes != NOT_FARMLAND).astype(np.uint8), [(0, 0, 0, 0), FARMLAND_COLOR])


def corners(bounds: tuple[float, float, float, float], epsg: int) -> list[list[float]]:
    """Longitude and latitude of the image corners: north-west, north-east, south-east,
    south-west (the order of MapLibre image sources)."""
    west, south, east, north = bounds
    points = MultiPoint([(west, north), (east, north), (east, south), (west, south)])
    return [[round(p.x, 7), round(p.y, 7)] for p in from_utm(points, epsg).geoms]


def _rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _png(indices: NDArray[np.uint8], palette: list[tuple[int, ...]]) -> bytes:
    image = Image.fromarray(indices)  # 8-bit; the palette makes it a "P" image
    image.putpalette([value for color in palette for value in color[:3]])
    buffer = io.BytesIO()
    image.save(
        buffer, format="PNG", optimize=True, transparency=bytes(color[3] for color in palette)
    )
    return buffer.getvalue()
