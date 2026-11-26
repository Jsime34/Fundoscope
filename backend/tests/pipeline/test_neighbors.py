"""The neighbor comparison read from a counted valley's files (hand-built near Ica)."""

import json

import pytest

from fundoscope.ingest import parse_parcel
from fundoscope.pipeline.neighbors import OUTSIDE, TOO_FEW, campaign_counts, neighbor_report


def test_each_campaign_gets_its_band_among_the_neighbors(neighbor_valley):
    valleys, parcel_file = neighbor_valley

    report = neighbor_report(parse_parcel(parcel_file, "vecino.geojson"), valleys)

    assert report.valley == "demo" and report.note == ""
    assert [(band.campaign, band.months) for band in report.bands] == [
        ("2024/25", 12),
        ("2025/26", 12),
        ("2026/27", 4),  # the current campaign, so far
    ]
    below, above, current = report.bands
    assert below.band == "below p10" and below.underperforming
    assert above.band == "above p90"
    assert current.band in ("p25-p50", "p50-p75")
    assert below.parcel_ndvi == pytest.approx(0.15, abs=1e-3)
    assert below.parcel_pixels == 64  # 80 m x 80 m
    # Pixels 50 m or more from the parcel, without the two columns that aren't farmland.
    assert 1_000 < below.neighbor_pixels < 1_400
    assert report.ring.geom_type == "Polygon"
    assert -75.71 < report.ring.centroid.x < -75.69  # WGS84, for the map


def test_a_parcel_outside_every_valley(neighbor_valley, parcel_geojson):
    valleys, _ = neighbor_valley

    report = neighbor_report(parse_parcel(parcel_geojson, "lote.geojson"), valleys)

    assert (report.valley, report.bands, report.ring, report.note) == (None, [], None, OUTSIDE)


def test_no_valleys_folder(tmp_path, parcel_geojson):
    report = neighbor_report(parse_parcel(parcel_geojson, "lote.geojson"), tmp_path / "none")

    assert report.note == OUTSIDE


def test_a_valley_being_set_up_is_skipped(neighbor_valley):
    valleys, parcel_file = neighbor_valley
    broken = valleys / "broken"
    broken.mkdir()
    (broken / "latest.json").write_text("{}", encoding="utf-8")  # no area of interest yet

    report = neighbor_report(parse_parcel(parcel_file, "vecino.geojson"), valleys)

    assert report.valley == "demo"


def test_too_few_neighbors(neighbor_valley):
    valleys, parcel_file = neighbor_valley

    report = neighbor_report(parse_parcel(parcel_file, "v.geojson"), valleys, min_neighbors=5_000)

    assert (report.valley, report.bands, report.note) == ("demo", [], TOO_FEW)
    assert report.ring is not None


def test_the_campaigns_compared():
    run = json.loads(
        '{"counts": ['
        '{"campaign": "2026/27", "months": 4, "through": "2026-11-22"},'
        '{"campaign": "2025/26", "months": 4, "through": "2025-11-22"},'
        '{"campaign": "2025/26", "months": 12, "through": "2026-07-31"},'
        '{"campaign": "2024/25", "months": 12, "through": "2025-07-31"}]}'
    )

    counts = campaign_counts(run)

    assert [(c.campaign, c.months) for c in counts] == [
        ("2024/25", 12),
        ("2025/26", 12),
        ("2026/27", 4),
    ]


def test_tiles_away_from_the_window_are_not_read(neighbor_valley):
    from dataclasses import replace
    from datetime import date

    import numpy as np

    from fundoscope.pipeline.neighbors import window_ndvi
    from fundoscope.pipeline.valley_store import load_valley
    from fundoscope.valley import Tile
    from fundoscope.valley.season import Count

    valleys, _ = neighbor_valley
    demo = load_valley(valleys / "demo", "demo")
    west, south, _, north = demo.tiles[0].bounds
    far = Tile("r9c9", (west + 10_000, south, west + 10_400, north))  # no files: never opened
    window = Tile("window", (west, north - 30, west + 20, north))  # 3 rows x 2 columns
    keys = Count("2024/25", 12, date(2025, 7, 31)).month_keys()[:2]

    values = window_ndvi(valleys / "demo", replace(demo, tiles=(far, demo.tiles[0])), window, keys)

    assert values.shape == (2, 3, 2)
    assert np.allclose(values[:, :, 1], 0.2 + 0.015, atol=1e-3)  # column 1 of the gradient
