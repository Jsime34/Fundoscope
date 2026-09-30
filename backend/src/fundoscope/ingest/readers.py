"""Read parcel boundaries from uploaded KML, KMZ, or GeoJSON bytes.

Output geometries are in WGS84 lon/lat. Only polygons are kept: a pin (Point) or a path
(LineString) is not a parcel. Nothing here touches the disk or network; callers pass bytes.

KML is parsed with `defusedxml` because uploads are untrusted: plain XML parsers can be
attacked with "billion laughs" entity expansion. KMZ is a zip; its uncompressed size is
capped to stop zip bombs.
"""

import io
import json
import zipfile
from dataclasses import dataclass
from xml.etree.ElementTree import Element

from defusedxml import ElementTree as SafeET
from defusedxml.common import DefusedXmlException
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.geometry.base import BaseGeometry

MAX_KMZ_UNCOMPRESSED_BYTES = 20 * 1024 * 1024
SUPPORTED_EXTENSIONS = (".kml", ".kmz", ".geojson", ".json")


class ParcelError(ValueError):
    """An uploaded parcel can't be used. The message is meant for the end user."""


@dataclass(frozen=True)
class ParcelFeature:
    """One named shape from an uploaded file (WGS84 lon/lat)."""

    name: str | None
    geometry: Polygon | MultiPolygon


def read_features(data: bytes, filename: str) -> list[ParcelFeature]:
    """Parse an uploaded file into polygon features, choosing the format by file extension.

    Raises:
        ParcelError: unsupported extension, unreadable file, or no polygon in it.
    """
    name = filename.lower()
    if name.endswith(".kmz"):
        features = _read_kml(_unzip_kml(data))
    elif name.endswith(".kml"):
        features = _read_kml(data)
    elif name.endswith((".geojson", ".json")):
        features = _read_geojson(data)
    else:
        raise ParcelError(
            f"Unsupported file type '{filename}'. Upload a {', '.join(SUPPORTED_EXTENSIONS)} file."
        )
    if not features:
        raise ParcelError(
            "No polygon found. Draw the parcel as a polygon (not a pin or a path) and export again."
        )
    return features


# --- GeoJSON ---------------------------------------------------------------------------


def _read_geojson(data: bytes) -> list[ParcelFeature]:
    try:
        doc = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ParcelError(f"The GeoJSON file is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise ParcelError("The GeoJSON file must contain a JSON object.")
    _reject_non_wgs84_crs(doc)

    if doc.get("type") == "FeatureCollection":
        items = [(_feature_name(f), f.get("geometry")) for f in doc.get("features") or []]
    elif doc.get("type") == "Feature":
        items = [(_feature_name(doc), doc.get("geometry"))]
    else:
        items = [(None, doc)]

    features = []
    for feature_name, geometry in items:
        if not geometry:
            continue
        try:
            geom = shape(geometry)
        except Exception as exc:
            raise ParcelError(f"Invalid GeoJSON geometry: {exc}") from exc
        features.extend(ParcelFeature(feature_name, poly) for poly in _polygons_of(geom))
    return features


def _feature_name(feature: dict) -> str | None:
    props = feature.get("properties") or {}
    name = props.get("name") or props.get("Name")
    return str(name) if name else None


def _reject_non_wgs84_crs(doc: dict) -> None:
    # RFC 7946 GeoJSON is always WGS84; older files may declare another CRS we'd misread.
    crs_name = ((doc.get("crs") or {}).get("properties") or {}).get("name", "")
    if crs_name and not any(code in crs_name for code in ("CRS84", "4326")):
        raise ParcelError(
            f"The GeoJSON uses the coordinate system '{crs_name}'. Export it in WGS84 "
            "(EPSG:4326, longitude/latitude) and upload again."
        )


def _polygons_of(geom: BaseGeometry) -> list[Polygon | MultiPolygon]:
    if isinstance(geom, Polygon | MultiPolygon):
        return [geom]
    if hasattr(geom, "geoms"):  # GeometryCollection: keep only its polygonal members
        return [part for g in geom.geoms for part in _polygons_of(g)]
    return []


# --- KML / KMZ -------------------------------------------------------------------------


def _unzip_kml(data: bytes) -> bytes:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ParcelError("The KMZ file is not a valid zip archive.") from exc
    with archive:
        kml_entries = [i for i in archive.infolist() if i.filename.lower().endswith(".kml")]
        if not kml_entries:
            raise ParcelError("The KMZ file does not contain a .kml document.")
        # Google Earth names the main document doc.kml; otherwise take the first one.
        entry = next((i for i in kml_entries if i.filename.lower() == "doc.kml"), kml_entries[0])
        if entry.file_size > MAX_KMZ_UNCOMPRESSED_BYTES:
            raise ParcelError("The KMZ file is too large once uncompressed.")
        return archive.read(entry)


def _read_kml(data: bytes) -> list[ParcelFeature]:
    try:
        root = SafeET.fromstring(data)
    except DefusedXmlException as exc:
        raise ParcelError("The KML file contains unsafe XML and was rejected.") from exc
    except SafeET.ParseError as exc:
        raise ParcelError(f"The KML file is not valid XML: {exc}") from exc

    features = []
    placemarks = [el for el in root.iter() if _local(el.tag) == "Placemark"]
    for placemark in placemarks or [root]:
        name_el = next((el for el in placemark if _local(el.tag) == "name"), None)
        name = name_el.text.strip() if name_el is not None and name_el.text else None
        for polygon_el in (el for el in placemark.iter() if _local(el.tag) == "Polygon"):
            features.append(ParcelFeature(name, _kml_polygon(polygon_el)))
    return features


def _kml_polygon(polygon_el: Element) -> Polygon:
    shell, holes = None, []
    for boundary in polygon_el:
        kind = _local(boundary.tag)
        if kind not in ("outerBoundaryIs", "innerBoundaryIs"):
            continue
        coords_el = next((el for el in boundary.iter() if _local(el.tag) == "coordinates"), None)
        ring = _kml_coordinates(coords_el.text if coords_el is not None else "")
        if kind == "outerBoundaryIs":
            shell = ring
        else:
            holes.append(ring)
    if not shell or len(shell) < 3:
        raise ParcelError("A KML polygon has no outer boundary with at least 3 points.")
    return Polygon(shell, [h for h in holes if len(h) >= 3])


def _kml_coordinates(text: str | None) -> list[tuple[float, float]]:
    # KML coordinates are "lon,lat[,alt]" tuples separated by whitespace; altitude is dropped.
    points = []
    for token in (text or "").split():
        parts = token.split(",")
        try:
            points.append((float(parts[0]), float(parts[1])))
        except (IndexError, ValueError) as exc:
            raise ParcelError(f"Invalid KML coordinate '{token}'.") from exc
    return points


def _local(tag: object) -> str:
    """Tag name without its namespace: '{http://www.opengis.net/kml/2.2}Polygon' -> 'Polygon'."""
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""
