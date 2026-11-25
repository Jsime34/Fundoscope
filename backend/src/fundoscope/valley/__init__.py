"""Valley crop monitor (FR-14).

Pure: takes arrays and shapes in and returns arrays and numbers; no database or network.

Submodules: `aoi` (the area of interest and its tiles), `season` (what a refresh on a given
day counts, and which months it downloads), `farmland` (the pixels to count),
`classify` (the class of every farmland pixel, with the "planted, not yet identifiable" rule
of a campaign in progress), `areas` (hectares per class with 95 % ranges, Olofsson et al.
2014).
"""

from fundoscope.valley.aoi import MAX_TILE_M, PIXEL_HA, PIXEL_M, Tile, Valley, snap_bounds, valley
from fundoscope.valley.areas import ClassArea, class_areas, olofsson
from fundoscope.valley.classify import (
    MIN_CONFIDENCE,
    NO_CLEAR_IMAGES,
    NOT_FARMLAND,
    PLANTED,
    ClassMap,
    classify_tile,
    map_classes,
    rising_from_bare,
)
from fundoscope.valley.farmland import FARMLAND_NDVI, NOT_FARMLAND_COVER, farmland_mask

__all__ = [
    "FARMLAND_NDVI",
    "MAX_TILE_M",
    "MIN_CONFIDENCE",
    "NOT_FARMLAND",
    "NOT_FARMLAND_COVER",
    "NO_CLEAR_IMAGES",
    "PIXEL_HA",
    "PIXEL_M",
    "PLANTED",
    "ClassArea",
    "ClassMap",
    "Tile",
    "Valley",
    "class_areas",
    "classify_tile",
    "farmland_mask",
    "map_classes",
    "olofsson",
    "rising_from_bare",
    "snap_bounds",
    "valley",
]
