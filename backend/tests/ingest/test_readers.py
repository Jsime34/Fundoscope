import io
import json
import zipfile

import pytest
import shapely
from shapely.geometry import mapping

from fundoscope.ingest import ParcelError, read_features, readers

ICA = (-75.73, -14.07)


# --- KML ---------------------------------------------------------------------------------


def test_kml_polygon_with_name(square_wgs84, kml_placemark, kml_document):
    square = square_wgs84(*ICA, 200)
    features = read_features(kml_document(kml_placemark("Lote 1", square)), "lote.kml")
    assert len(features) == 1
    assert features[0].name == "Lote 1"
    assert shapely.equals_exact(features[0].geometry, square, tolerance=1e-9)


def test_kml_without_namespace(square_wgs84, kml_placemark, kml_document):
    data = kml_document(kml_placemark("A", square_wgs84(*ICA, 200)), namespace=False)
    assert len(read_features(data, "a.kml")) == 1


def test_kml_hole_is_kept(square_wgs84, kml_placemark, kml_document):
    outer = square_wgs84(*ICA, 400)
    inner = square_wgs84(*ICA, 100)
    with_hole = shapely.Polygon(outer.exterior.coords, [inner.exterior.coords])
    [feature] = read_features(kml_document(kml_placemark("Pozo", with_hole)), "x.kml")
    assert len(feature.geometry.interiors) == 1


def test_kml_multigeometry_gives_one_feature_per_polygon(square_wgs84, kml_placemark, kml_document):
    lot_a = square_wgs84(-75.73, -14.07, 200)
    lot_b = square_wgs84(-75.72, -14.07, 200)
    features = read_features(kml_document(kml_placemark("Fundo", lot_a, lot_b)), "f.kml")
    assert [f.name for f in features] == ["Fundo", "Fundo"]


def test_kml_altitude_is_dropped(square_wgs84, kml_placemark, kml_document):
    [feature] = read_features(kml_document(kml_placemark("A", square_wgs84(*ICA, 200))), "a.kml")
    assert not feature.geometry.has_z


def test_kml_with_only_a_pin_is_rejected():
    pin = (
        b'<kml xmlns="http://www.opengis.net/kml/2.2"><Placemark><name>Casa</name>'
        b"<Point><coordinates>-75.73,-14.07,0</coordinates></Point></Placemark></kml>"
    )
    with pytest.raises(ParcelError, match="No polygon"):
        read_features(pin, "pin.kml")


def test_malformed_kml_is_rejected():
    with pytest.raises(ParcelError, match="not valid XML"):
        read_features(b"<kml><Placemark>", "broken.kml")


def test_bad_kml_coordinate_is_rejected(kml_document):
    bad = kml_document(
        "<Placemark><Polygon><outerBoundaryIs><LinearRing><coordinates>"
        "-75.7;-14.0 1,2 3,4</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>"
    )
    with pytest.raises(ParcelError, match="Invalid KML coordinate"):
        read_features(bad, "bad.kml")


def test_xml_entity_bomb_is_rejected():
    bomb = (
        b'<?xml version="1.0"?><!DOCTYPE kml [<!ENTITY a "aaaaaaaaaa">'
        b'<!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">]><kml>&b;</kml>'
    )
    with pytest.raises(ParcelError, match="unsafe XML"):
        read_features(bomb, "bomb.kml")


# --- KMZ ---------------------------------------------------------------------------------


def _zip(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def test_kmz_reads_doc_kml(square_wgs84, kml_placemark, kml_document):
    kml = kml_document(kml_placemark("Lote KMZ", square_wgs84(*ICA, 200)))
    kmz = _zip({"files/icon.png": b"\x89PNG", "doc.kml": kml})
    [feature] = read_features(kmz, "LOTE.KMZ")
    assert feature.name == "Lote KMZ"


def test_kmz_without_kml_is_rejected():
    with pytest.raises(ParcelError, match="does not contain a .kml"):
        read_features(_zip({"readme.txt": b"hi"}), "x.kmz")


def test_kmz_that_is_not_a_zip_is_rejected():
    with pytest.raises(ParcelError, match="not a valid zip"):
        read_features(b"definitely not a zip", "x.kmz")


# --- GeoJSON -----------------------------------------------------------------------------


def test_geojson_feature_collection(square_wgs84):
    doc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "A"},
                "geometry": mapping(square_wgs84(*ICA, 200)),
            },
            {
                "type": "Feature",
                "properties": {},
                "geometry": mapping(square_wgs84(-75.72, -14.07, 200)),
            },
        ],
    }
    features = read_features(json.dumps(doc).encode(), "lots.geojson")
    assert [f.name for f in features] == ["A", None]


def test_geojson_bare_geometry(square_wgs84):
    data = json.dumps(mapping(square_wgs84(*ICA, 200))).encode()
    assert len(read_features(data, "lot.json")) == 1


def test_geojson_in_another_crs_is_rejected(square_wgs84):
    doc = {
        "type": "Feature",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32718"}},
        "geometry": mapping(square_wgs84(*ICA, 200)),
    }
    with pytest.raises(ParcelError, match="EPSG:4326"):
        read_features(json.dumps(doc).encode(), "utm.geojson")


def test_geojson_with_only_points_is_rejected():
    point = json.dumps({"type": "Point", "coordinates": [-75.73, -14.07]}).encode()
    with pytest.raises(ParcelError, match="No polygon"):
        read_features(point, "p.geojson")


def test_invalid_json_is_rejected():
    with pytest.raises(ParcelError, match="not valid JSON"):
        read_features(b"{not json", "x.geojson")


# --- Other -------------------------------------------------------------------------------


def test_unsupported_extension_is_rejected():
    with pytest.raises(ParcelError, match="Unsupported file type"):
        read_features(b"...", "lote.shp")


def test_geojson_single_feature(square_wgs84):
    doc = {
        "type": "Feature",
        "properties": {"Name": "Lote 7"},
        "geometry": mapping(square_wgs84(*ICA, 200)),
    }
    [feature] = read_features(json.dumps(doc).encode(), "lote7.geojson")
    assert feature.name == "Lote 7"


def test_geojson_that_is_not_an_object_is_rejected():
    with pytest.raises(ParcelError, match="JSON object"):
        read_features(b"[1, 2, 3]", "x.geojson")


def test_kml_polygon_with_fewer_than_three_points_is_rejected(kml_document):
    two_points = kml_document(
        "<Placemark><Polygon><outerBoundaryIs><LinearRing><coordinates>"
        "-75.73,-14.07 -75.72,-14.07</coordinates>"
        "</LinearRing></outerBoundaryIs></Polygon></Placemark>"
    )
    with pytest.raises(ParcelError, match="at least 3 points"):
        read_features(two_points, "line.kml")


def test_kmz_zip_bomb_is_rejected(monkeypatch, square_wgs84, kml_placemark, kml_document):
    monkeypatch.setattr(readers, "MAX_KMZ_UNCOMPRESSED_BYTES", 100)  # instead of building 20 MB
    kmz = _zip({"doc.kml": kml_document(kml_placemark("A", square_wgs84(*ICA, 200)))})
    with pytest.raises(ParcelError, match="too large"):
        read_features(kmz, "big.kmz")
