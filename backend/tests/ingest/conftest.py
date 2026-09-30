from collections.abc import Callable

import pytest
from shapely.geometry import Point, Polygon, box

from fundoscope.ingest import from_utm, to_utm, utm_epsg


@pytest.fixture
def square_wgs84() -> Callable[..., Polygon]:
    """Build a square of `side_m` meters centered on (lon, lat), returned in WGS84."""

    def make(lon: float, lat: float, side_m: float) -> Polygon:
        epsg = utm_epsg(lon, lat)
        center = to_utm(Point(lon, lat), epsg)
        half = side_m / 2
        square = box(center.x - half, center.y - half, center.x + half, center.y + half)
        return from_utm(square, epsg)

    return make


def _kml_coords(ring) -> str:
    """KML coordinate text ("lon,lat,0 lon,lat,0 ...") for a polygon ring."""
    return " ".join(f"{x},{y},0" for x, y in ring.coords)


def _kml_placemark(name: str, *polygons: Polygon) -> str:
    bodies = []
    for poly in polygons:
        holes = "".join(
            "<innerBoundaryIs><LinearRing><coordinates>"
            f"{_kml_coords(hole)}</coordinates></LinearRing></innerBoundaryIs>"
            for hole in poly.interiors
        )
        bodies.append(
            "<Polygon><outerBoundaryIs><LinearRing><coordinates>"
            f"{_kml_coords(poly.exterior)}</coordinates></LinearRing></outerBoundaryIs>"
            f"{holes}</Polygon>"
        )
    geometry = (
        bodies[0] if len(bodies) == 1 else f"<MultiGeometry>{''.join(bodies)}</MultiGeometry>"
    )
    return f"<Placemark><name>{name}</name>{geometry}</Placemark>"


def _kml_document(*placemarks: str, namespace: bool = True) -> bytes:
    xmlns = ' xmlns="http://www.opengis.net/kml/2.2"' if namespace else ""
    return (
        f'<?xml version="1.0" encoding="UTF-8"?><kml{xmlns}><Document>'
        f"{''.join(placemarks)}</Document></kml>"
    ).encode()


@pytest.fixture
def kml_placemark() -> Callable[..., str]:
    """kml_placemark(name, *polygons) -> <Placemark> XML; several polygons -> MultiGeometry."""
    return _kml_placemark


@pytest.fixture
def kml_document() -> Callable[..., bytes]:
    """kml_document(*placemarks, namespace=True) -> full KML file as bytes."""
    return _kml_document
