import numpy as np
import pytest
from shapely.geometry import Polygon, box

from fundoscope.valley import PIXEL_HA, Tile, snap_bounds, valley
from fundoscope.valley.aoi import tile_grid

WEST, SOUTH = 400_000.0, 8_400_000.0


def test_bounds_grow_outward_to_the_pixel_grid():
    assert snap_bounds((400_003.2, 8_400_009.9, 400_101.0, 8_400_150.0)) == (
        400_000,
        8_400_000,
        400_110,
        8_400_150,
    )


def test_tiles_are_equal_and_cover_the_bounds_without_overlap():
    tiles = tile_grid((WEST, SOUTH, WEST + 25_970, SOUTH + 23_610), max_tile_m=9_000)

    assert [tile.id for tile in tiles] == [f"r{r}c{c}" for r in range(3) for c in range(3)]
    assert tiles[0].bounds == (WEST, SOUTH + 15_740, WEST + 8_660, SOUTH + 23_610)  # north-west
    assert tiles[-1].bounds == (WEST + 17_310, SOUTH, WEST + 25_970, SOUTH + 7_870)
    assert sum(tile.geometry.area for tile in tiles) == pytest.approx(25_970 * 23_610)
    for tile in tiles:
        assert all(value % 10 == 0 for value in tile.bounds)
        west, south, east, north = tile.bounds
        assert east - west <= 9_000 and north - south <= 9_000


def test_a_small_area_is_one_tile():
    assert tile_grid((WEST, SOUTH, WEST + 500, SOUTH + 300)) == (
        Tile("r0c0", (WEST, SOUTH, WEST + 500, SOUTH + 300)),
    )


def test_pixel_centers_run_west_to_east_and_north_to_south():
    tile = Tile("r0c0", (WEST, SOUTH, WEST + 30, SOUTH + 20))

    assert list(tile.x) == [WEST + 5, WEST + 15, WEST + 25]
    assert list(tile.y) == [SOUTH + 15, SOUTH + 5]


def test_a_valley_knows_its_area_tiles_and_pixels_inside():
    triangle = Polygon([(WEST, SOUTH), (WEST + 40, SOUTH), (WEST, SOUTH + 40)])

    majes = valley("majes", triangle, 32718, max_tile_m=20)

    assert majes.area_km2 == pytest.approx(0.0008)
    assert majes.bounds == (WEST, SOUTH, WEST + 40, SOUTH + 40)
    assert len(majes.tiles) == 4
    inside = majes.inside(majes.tiles[2])  # the south-west tile, fully inside
    assert inside.all()
    corner = majes.inside(majes.tiles[1])  # the north-east tile: only below the diagonal
    assert not corner.any()
    assert PIXEL_HA == 0.01


def test_pixels_are_inside_when_their_center_is():
    majes = valley("majes", box(WEST, SOUTH, WEST + 36, SOUTH + 10), 32718)  # 4th center at +35

    assert majes.inside(majes.tiles[0]).tolist() == [[True, True, True, True]]
    assert np.isclose(majes.bounds[2], WEST + 40)


def test_an_empty_area_is_refused():
    with pytest.raises(ValueError, match="empty"):
        valley("majes", Polygon(), 32718)


def test_a_valley_can_keep_the_grid_of_an_earlier_run():
    grid = (WEST, SOUTH, WEST + 40, SOUTH + 40)
    smaller = box(WEST, SOUTH, WEST + 20, SOUTH + 20)

    kept = valley("majes", smaller, 32718, max_tile_m=20, bounds=grid)

    assert kept.bounds == grid and len(kept.tiles) == 4


def test_the_area_must_stay_inside_the_grid():
    with pytest.raises(ValueError, match="goes beyond its tile grid"):
        valley(
            "majes",
            box(WEST, SOUTH, WEST + 50, SOUTH + 20),
            32718,
            bounds=(WEST, SOUTH, WEST + 40, SOUTH + 40),
        )
