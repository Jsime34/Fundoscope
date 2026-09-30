import json

import pytest
import shapely
from shapely.geometry import MultiPolygon, Polygon, mapping

from fundoscope.ingest import (
    ParcelError,
    ParcelFeature,
    from_utm,
    parse_parcel,
    to_utm,
    utm_epsg,
    validate_parcel,
)

ICA = (-75.73, -14.07)
PIURA = (-80.63, -5.19)
SIDE_10_HA = 316.2278  # meters: 316.23 m x 316.23 m = 10 ha


def _features(*polygons: Polygon) -> list[ParcelFeature]:
    return [ParcelFeature(None, p) for p in polygons]


def test_ten_hectares_in_ica(square_wgs84):
    parcel = validate_parcel(_features(square_wgs84(*ICA, SIDE_10_HA)))
    assert parcel.area_ha == pytest.approx(10.0, rel=1e-4)
    assert parcel.utm_epsg == 32718
    assert isinstance(parcel.geometry, MultiPolygon)
    assert parcel.geometry_utm.area == pytest.approx(100_000, rel=1e-4)
    assert parcel.repaired is False


def test_piura_is_measured_in_zone_17(square_wgs84):
    assert validate_parcel(_features(square_wgs84(*PIURA, SIDE_10_HA))).utm_epsg == 32717


def test_separate_lots_are_merged_into_one_parcel(square_wgs84):
    lot_a = square_wgs84(-75.73, -14.07, 200)  # 4 ha each, about 1 km apart
    lot_b = square_wgs84(-75.72, -14.07, 200)
    parcel = validate_parcel(_features(lot_a, lot_b))
    assert len(parcel.geometry.geoms) == 2
    assert parcel.area_ha == pytest.approx(8.0, rel=1e-3)


def test_overlapping_lots_are_not_counted_twice(square_wgs84):
    square = square_wgs84(*ICA, 200)
    parcel = validate_parcel(_features(square, square))
    assert parcel.area_ha == pytest.approx(4.0, rel=1e-3)


def test_self_crossing_outline_is_repaired_and_flagged():
    epsg = utm_epsg(*ICA)
    center = to_utm(shapely.Point(*ICA), epsg)
    x, y = center.x, center.y
    # A "bow tie": the outline crosses itself in the middle.
    bow_tie_utm = Polygon([(x, y), (x + 300, y + 300), (x + 300, y), (x, y + 300)])
    bow_tie = from_utm(bow_tie_utm, epsg)
    assert not bow_tie.is_valid

    parcel = validate_parcel(_features(bow_tie))
    assert parcel.geometry.is_valid
    assert parcel.repaired is True
    assert parcel.area_ha == pytest.approx(4.5, rel=1e-3)  # two triangles of 300 x 150 m


def test_too_small_is_rejected(square_wgs84):
    with pytest.raises(ParcelError, match="minimum is 0.5 ha"):
        validate_parcel(_features(square_wgs84(*ICA, 50)))  # 0.25 ha


def test_too_large_is_rejected(square_wgs84):
    with pytest.raises(ParcelError, match="maximum is 2,000 ha"):
        validate_parcel(_features(square_wgs84(*ICA, 5_000)))  # 2,500 ha


def test_outside_peru_is_rejected(square_wgs84):
    with pytest.raises(ParcelError, match="outside Peru"):
        validate_parcel(_features(square_wgs84(-3.70, 40.42, SIDE_10_HA)))  # Madrid


def test_swapped_lat_lon_is_rejected(square_wgs84):
    square = square_wgs84(*ICA, SIDE_10_HA)
    swapped = Polygon([(lat, lon) for lon, lat in square.exterior.coords])
    with pytest.raises(ParcelError, match="coordinates are not swapped"):
        validate_parcel(_features(swapped))


def test_altitude_is_dropped(square_wgs84):
    square = square_wgs84(*ICA, SIDE_10_HA)
    with_z = Polygon([(x, y, 120.0) for x, y in square.exterior.coords])
    parcel = validate_parcel(_features(with_z))
    assert not parcel.geometry.has_z


def test_parse_parcel_reads_a_file_and_keeps_the_first_name(square_wgs84):
    doc = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {}, "geometry": mapping(square_wgs84(*ICA, 200))},
            {
                "type": "Feature",
                "properties": {"name": "Fundo San José"},
                "geometry": mapping(square_wgs84(-75.72, -14.07, 200)),
            },
        ],
    }
    parcel = parse_parcel(json.dumps(doc).encode(), "fundo.geojson")
    assert parcel.name == "Fundo San José"
    assert parcel.area_ha == pytest.approx(8.0, rel=1e-3)


def test_empty_feature_list_is_rejected():
    with pytest.raises(ParcelError, match="No polygon"):
        validate_parcel([])


def test_zero_width_spike_is_removed_without_flagging_a_repair():
    epsg = utm_epsg(*ICA)
    center = to_utm(shapely.Point(*ICA), epsg)
    x, y = center.x, center.y
    # A 300 m square whose top edge has a 100 m spike going out and back (zero width).
    spiky_utm = Polygon(
        [
            (x, y),
            (x + 300, y),
            (x + 300, y + 300),
            (x + 150, y + 300),
            (x + 150, y + 400),
            (x + 150, y + 300),
            (x, y + 300),
        ]
    )
    spiky = from_utm(spiky_utm, epsg)
    assert not spiky.is_valid

    parcel = validate_parcel(_features(spiky))
    assert parcel.geometry.is_valid
    assert parcel.area_ha == pytest.approx(9.0, rel=1e-3)
    assert parcel.repaired is False  # the area didn't change, only a zero-width line was dropped
