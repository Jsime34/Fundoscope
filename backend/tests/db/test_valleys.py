"""What every valley store must do (in memory and in PostGIS)."""

from datetime import UTC, date, datetime

import pytest
from shapely.geometry import LineString, MultiPolygon, Point, box

AREA = box(-75.74, -14.08, -75.72, -14.06)  # public farmland near Ica
ELSEWHERE = box(-75.62, -14.08, -75.58, -14.06)
INSIDE, OUTSIDE = Point(-75.73, -14.07), Point(-75.60, -14.07)


def test_a_saved_run_comes_back_whole(any_valleys, make_valley_run):
    run = make_valley_run()

    any_valleys.save_run(run, AREA)

    assert any_valleys.latest_run("demo") == run
    assert any_valleys.latest_run("other") is None


def test_a_valley_never_counted(any_valleys):
    assert any_valleys.latest_run("demo") is None
    assert any_valleys.latest_run_at(INSIDE) is None


def test_the_latest_run_is_the_newest_count(any_valleys, make_valley_run):
    newer = make_valley_run(run_at=datetime(2026, 11, 24, 14, 30, tzinfo=UTC))
    older = make_valley_run(
        run_at=datetime(2026, 11, 17, 14, 30, tzinfo=UTC), as_of=date(2026, 11, 15)
    )

    any_valleys.save_run(newer, AREA)
    any_valleys.save_run(older, AREA)  # saved last, but counted first

    assert any_valleys.latest_run("demo") == newer


def test_a_point_finds_the_valley_that_holds_it(any_valleys, make_valley_run):
    any_valleys.save_run(make_valley_run(), AREA)
    any_valleys.save_run(make_valley_run("other"), MultiPolygon([ELSEWHERE]))

    assert any_valleys.latest_run_at(INSIDE).valley == "demo"
    assert any_valleys.latest_run_at(OUTSIDE).valley == "other"
    assert any_valleys.latest_run_at(Point(-72.27, -16.37)) is None


def test_a_redrawn_area_replaces_the_old_one(any_valleys, make_valley_run):
    any_valleys.save_run(make_valley_run(), AREA)
    any_valleys.save_run(
        make_valley_run(run_at=datetime(2026, 12, 1, 14, 30, tzinfo=UTC)), ELSEWHERE
    )

    assert any_valleys.latest_run_at(INSIDE) is None
    assert any_valleys.latest_run_at(OUTSIDE).run_at == datetime(2026, 12, 1, 14, 30, tzinfo=UTC)


def test_an_area_of_interest_must_be_a_polygon(any_valleys, make_valley_run):
    with pytest.raises(ValueError, match="must be a polygon, not a LineString"):
        any_valleys.save_run(make_valley_run(), LineString([(-75.74, -14.08), (-75.72, -14.06)]))

    assert any_valleys.latest_run("demo") is None
