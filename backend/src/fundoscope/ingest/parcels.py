"""Turn uploaded shapes into one validated parcel with its UTM zone and area (FR-01).

Steps: drop altitude -> repair invalid geometry -> merge all polygons (a fundo can have
several separate lots) -> check it lies in Peru -> choose the UTM zone from the centroid ->
compute hectares in UTM -> enforce the size limits.
"""

from dataclasses import dataclass

import shapely
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from fundoscope.ingest.readers import ParcelError, ParcelFeature, read_features
from fundoscope.ingest.utm import to_utm, utm_epsg

MIN_AREA_HA = 0.5
MAX_AREA_HA = 2_000.0
M2_PER_HA = 10_000.0

# Peru's extent (lon/lat) with a small margin: -81.33..-68.65 E, -18.35..-0.04 N.
PERU_BOUNDS = (-81.5, -18.5, -68.5, 0.1)

# A repair that changes the area by more than this is reported to the user.
REPAIR_AREA_TOLERANCE = 0.01


@dataclass(frozen=True)
class Parcel:
    """A validated parcel.

    Attributes:
        name: name from the file (first named shape), or None.
        geometry: MultiPolygon in WGS84 lon/lat (what we store and send to Copernicus).
        utm_epsg: EPSG code of the UTM zone used for measurements (32717 or 32718 in Peru).
        geometry_utm: the same MultiPolygon in UTM meters.
        area_ha: area in hectares, measured in UTM.
        repaired: True if the uploaded outline was invalid (e.g. self-intersecting) and the
            repair changed its area by more than 1 %. The report should mention it.
    """

    name: str | None
    geometry: MultiPolygon
    utm_epsg: int
    geometry_utm: MultiPolygon
    area_ha: float
    repaired: bool


def parse_parcel(data: bytes, filename: str) -> Parcel:
    """Read an uploaded KML/KMZ/GeoJSON file and validate it as one parcel.

    Raises:
        ParcelError: with a message the user can act on.
    """
    features = read_features(data, filename)
    name = next((f.name for f in features if f.name), None)
    return validate_parcel(features, name=name)


def validate_parcel(features: list[ParcelFeature], name: str | None = None) -> Parcel:
    """Validate polygon features (WGS84) and measure them as a single parcel."""
    raw = [shapely.force_2d(f.geometry) for f in features]
    if not raw:
        raise ParcelError("No polygon found.")

    repaired_parts = [shapely.make_valid(g) if not g.is_valid else g for g in raw]
    merged = _as_multipolygon(shapely.union_all(repaired_parts))
    if merged.is_empty:
        raise ParcelError("The parcel outline has no area.")

    _check_in_peru(merged)

    lon, lat = merged.centroid.x, merged.centroid.y
    epsg = utm_epsg(lon, lat)
    merged_utm = _as_multipolygon(to_utm(merged, epsg))
    area_ha = merged_utm.area / M2_PER_HA

    if area_ha < MIN_AREA_HA:
        raise ParcelError(
            f"The parcel is {area_ha:.2f} ha; the minimum is {MIN_AREA_HA} ha "
            "(smaller fields contain too few 10 m satellite pixels)."
        )
    if area_ha > MAX_AREA_HA:
        raise ParcelError(
            f"The parcel is {area_ha:,.0f} ha; the maximum is {MAX_AREA_HA:,.0f} ha. "
            "Upload individual fundos rather than a whole valley."
        )

    return Parcel(
        name=name,
        geometry=merged,
        utm_epsg=epsg,
        geometry_utm=merged_utm,
        area_ha=area_ha,
        repaired=_repair_changed_area(raw, merged_utm, epsg),
    )


def _as_multipolygon(geom: BaseGeometry) -> MultiPolygon:
    """Keep only the polygonal parts (make_valid can also produce lines/points)."""
    if isinstance(geom, MultiPolygon):
        return geom
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    polygons = [p for p in shapely.get_parts(geom) if isinstance(p, Polygon | MultiPolygon)]
    parts = [q for p in polygons for q in shapely.get_parts(p)]
    return MultiPolygon(parts)


def _check_in_peru(geom: MultiPolygon) -> None:
    min_lon, min_lat, max_lon, max_lat = geom.bounds
    west, south, east, north = PERU_BOUNDS
    if min_lon < west or max_lon > east or min_lat < south or max_lat > north:
        raise ParcelError(
            "The parcel is outside Peru. Check that the file uses longitude/latitude (WGS84) "
            "and that the coordinates are not swapped."
        )


def _repair_changed_area(raw: list[BaseGeometry], merged_utm: MultiPolygon, epsg: int) -> bool:
    if all(g.is_valid for g in raw):
        return False
    # GEOS measures an invalid outline with the shoelace formula, i.e. "as drawn". A bow-tie
    # (self-crossing) outline cancels itself out there, so the repaired area differs a lot.
    drawn_m2 = sum(to_utm(g, epsg).area for g in raw)
    return abs(drawn_m2 - merged_utm.area) > REPAIR_AREA_TOLERANCE * merged_utm.area
