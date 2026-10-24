import pytest
import shapely
from shapely.geometry import Point, box

from fundoscope.ingest import from_utm, to_utm, utm_epsg, zone_label


@pytest.mark.parametrize(
    ("lon", "lat", "expected"),
    [
        (-80.63, -5.19, 32717),  # Piura
        (-79.03, -8.11, 32717),  # Trujillo, La Libertad
        (-75.73, -14.07, 32718),  # Ica
        (-78.0, -10.0, 32718),  # exactly on the 17/18 boundary -> the eastern zone
        (-3.70, 40.42, 32630),  # Madrid, northern hemisphere
        (180.0, 10.0, 32660),  # the antimeridian belongs to zone 60
        (-180.0, -10.0, 32701),
    ],
)
def test_utm_epsg_picks_the_standard_zone(lon, lat, expected):
    assert utm_epsg(lon, lat) == expected


@pytest.mark.parametrize(("lon", "lat"), [(-181, 0), (181, 0), (0, -81), (0, 85)])
def test_utm_epsg_rejects_out_of_range_coordinates(lon, lat):
    with pytest.raises(ValueError):
        utm_epsg(lon, lat)


def test_central_meridian_maps_to_false_easting():
    # Zone 18's central meridian is 75 deg W: easting there is exactly 500 km, and in the
    # southern hemisphere the equator sits at the 10,000 km false northing.
    point = to_utm(Point(-75.0, 0.0), 32718)
    assert point.x == pytest.approx(500_000, abs=0.01)
    assert point.y == pytest.approx(10_000_000, abs=0.01)


def test_round_trip_keeps_coordinates():
    square = box(-75.74, -14.08, -75.73, -14.07)
    back = from_utm(to_utm(square, 32718), 32718)
    assert shapely.equals_exact(back, square, tolerance=1e-9)


def test_one_hectare_measured_in_utm():
    hectare_utm = box(420_000, 8_440_000, 420_100, 8_440_100)  # 100 m x 100 m in zone 18S
    lonlat = from_utm(hectare_utm, 32718)
    assert to_utm(lonlat, 32718).area == pytest.approx(10_000, rel=1e-6)


def test_zone_labels():
    assert zone_label(32718) == "UTM 18S"
    assert zone_label(32717) == "UTM 17S"
    assert zone_label(32630) == "UTM 30N"
    assert zone_label(4326) == "EPSG:4326"
