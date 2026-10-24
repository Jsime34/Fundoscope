"""UTM zone selection and reprojection between WGS84 (lon/lat degrees) and UTM (meters).

Why: lon/lat degrees are not a unit of length, so areas and buffers must be computed in a
projected CRS. UTM splits the globe into 60 zones, each 6 degrees of longitude wide; inside
its zone the distortion is small (< 0.1 %). Coastal Peru falls in zone 17S (Piura,
La Libertad) or 18S (Lima to Arequipa); the far south (Moquegua, Tacna) is 19S.
"""

from functools import lru_cache

import shapely
from pyproj import Transformer
from shapely.geometry.base import BaseGeometry

WGS84_EPSG = 4326


def zone_label(epsg: int) -> str:
    """Human-readable name of a WGS84/UTM zone: 32718 -> "UTM 18S"; other codes -> "EPSG:...".

    A zone says nothing about the region: 18S covers Ica but also Lima and Arequipa.
    """
    if 32601 <= epsg <= 32660:
        return f"UTM {epsg - 32600}N"
    if 32701 <= epsg <= 32760:
        return f"UTM {epsg - 32700}S"
    return f"EPSG:{epsg}"


def utm_epsg(lon: float, lat: float) -> int:
    """Return the EPSG code of the WGS84/UTM zone that contains a point.

    Args:
        lon: longitude in degrees, -180..180.
        lat: latitude in degrees; UTM is defined from -80 to 84.

    Returns:
        326xx for the northern hemisphere, 327xx for the southern one (xx = zone 01..60).
        Example: Ica (-75.7, -14.1) -> 32718.

    The Norway/Svalbard exceptions to the standard zones are not handled (not needed for Peru).
    """
    if not -180.0 <= lon <= 180.0:
        raise ValueError(f"Longitude {lon} is outside -180..180")
    if not -80.0 <= lat <= 84.0:
        raise ValueError(f"Latitude {lat} is outside the UTM range -80..84")
    zone = min(int((lon + 180.0) // 6.0) + 1, 60)  # lon = 180 belongs to zone 60
    return (32600 if lat >= 0 else 32700) + zone


@lru_cache(maxsize=16)
def _transformer(src_epsg: int, dst_epsg: int) -> Transformer:
    return Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)


def to_utm(geom: BaseGeometry, epsg: int) -> BaseGeometry:
    """Reproject a WGS84 (lon/lat) geometry into the UTM zone `epsg`. Output units: meters."""
    return shapely.transform(geom, _transformer(WGS84_EPSG, epsg).transform, interleaved=False)


def from_utm(geom: BaseGeometry, epsg: int) -> BaseGeometry:
    """Reproject a geometry from the UTM zone `epsg` (meters) back to WGS84 (lon/lat)."""
    return shapely.transform(geom, _transformer(epsg, WGS84_EPSG).transform, interleaved=False)
