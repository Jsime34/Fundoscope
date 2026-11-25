"""`fundoscope valley refresh`: the small synthetic valley near Ica, no network."""

import json
from datetime import date

from shapely.geometry import box, mapping

import fundoscope.cli.valley
from fundoscope.cli import main
from fundoscope.cli.valley import format_plan, run_valley_refresh
from fundoscope.ingest import from_utm
from fundoscope.pipeline.valley_refresh import plan_refresh
from fundoscope.valley import PLANTED, valley

TODAY = date(2026, 11, 24)


def write_aoi(folder, small_valley):
    """The test valley's area of interest, 1 m inside its 80 m x 40 m grid, in WGS84."""
    area = small_valley.area_utm.buffer(-1, join_style="mitre")
    feature = {"type": "Feature", "properties": {}, "geometry": mapping(from_utm(area, 32718))}
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "aoi.geojson").write_text(json.dumps(feature), encoding="utf-8")
    return folder


def run(folder, models, fakes, used=None, cap=1_000.0, **options):
    return run_valley_refresh(
        "demo", used, cap, folder=folder, models=models, today=TODAY, **fakes, **options
    )


def test_a_dry_run_shows_the_plan_and_downloads_nothing(
    tmp_path, valley_models, valley_fakes, capsys, small_valley
):
    code = run(
        write_aoi(tmp_path, small_valley), valley_models, valley_fakes, cap=None, dry_run=True
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "in 1 tiles. Images up to 2026-11-22." in out
    assert "Counts: 2026/27 to Nov 22; 2025/26 to Nov 22; 2025/26; 2024/25." in out
    assert "To download: 6 requests and the land cover (about 49 openEO credits" in out
    assert valley_fakes["fetch"].calls == []


def test_downloads_need_a_credit_limit(tmp_path, valley_models, valley_fakes, capsys, small_valley):
    code = run(write_aoi(tmp_path, small_valley), valley_models, valley_fakes, cap=None)

    assert code == 1
    assert "Pass --used-credits N" in capsys.readouterr().out
    assert valley_fakes["fetch"].calls == []


def test_a_refresh_prints_the_hectares_and_the_comparison(
    tmp_path, valley_models, valley_fakes, capsys, small_valley
):
    folder = write_aoi(tmp_path, small_valley)
    code = run(folder, valley_models, valley_fakes, used=7_000.0, workers=1)

    out = capsys.readouterr().out
    assert code == 0
    assert "This run may spend up to 1,000 credits." in out
    assert "  [1/7] land cover: 7 credits" in out
    assert "Downloaded 7 of 7 (49.0 openEO credits)." in out
    assert "Classifying the tiles...\n  r0c0 (1/1)" in out
    assert "Newest image: 2026-11-21." in out
    assert "2026/27 to Nov 22 (4 months, model season/rf_v1_to_nov.joblib):" in out
    assert "2025/26 (12 months, model rf_v1.joblib):" in out
    assert "Adjusted (95 % range)" in out
    assert f"  {PLANTED}" in out
    assert "Compared with 2025/26 to Nov 22 (same date, same model):" in out
    assert "(-14%)" in out  # 6 vineyard pixels this season against 7 (one is cloudy)
    assert "Saved in" in out and "latest.json" in out


def test_the_used_credits_limit_the_run(
    tmp_path, valley_models, valley_fakes, capsys, small_valley
):
    run(write_aoi(tmp_path, small_valley), valley_models, valley_fakes, used=7_990.0, cap=None)

    out = capsys.readouterr().out
    assert "This run may spend up to 10 credits." in out
    assert "Stopped at the credit cap." in out
    assert "6 downloads are still missing, so nothing was counted." in out


def test_a_second_refresh_needs_no_credits(
    tmp_path, valley_models, valley_fakes, capsys, small_valley
):
    folder = write_aoi(tmp_path, small_valley)
    run(folder, valley_models, valley_fakes)
    capsys.readouterr()

    code = run(folder, valley_models, valley_fakes, cap=None)

    out = capsys.readouterr().out
    assert code == 0
    assert "Nothing to download: every month is on disk." in out
    assert "Downloaded" not in out and "Farmland:" in out


def test_an_unknown_newest_image(tmp_path, valley_models, valley_fakes, capsys, small_valley):
    valley_fakes["newest_image"].fail = True

    run(write_aoi(tmp_path, small_valley), valley_models, valley_fakes)

    assert "Newest image: unknown (the catalog didn't answer)." in capsys.readouterr().out


def test_failures_are_listed(tmp_path, valley_models, valley_fakes, capsys, small_valley):
    valley_fakes["fetch"].fail = True

    code = run(write_aoi(tmp_path, small_valley), valley_models, valley_fakes, workers=1)

    out = capsys.readouterr().out
    assert code == 1
    assert "FAILED: simulated openEO failure" in out
    assert "  Failed: r0c0 2024-08 to 2025-01: simulated openEO failure" in out
    assert "Stopped after 3 failures in a row" in out


def test_a_valley_without_an_area_of_interest(tmp_path, valley_models, capsys):
    assert run(tmp_path, valley_models, {}) == 1
    assert "draw it and save it as" in capsys.readouterr().out


def test_a_refresh_without_models(tmp_path, valley_fakes, capsys, small_valley):
    assert run(write_aoi(tmp_path, small_valley), tmp_path / "none", valley_fakes) == 1
    assert "fundoscope model train" in capsys.readouterr().out


def test_the_command_line(tmp_path, monkeypatch, capsys, small_valley):
    write_aoi(tmp_path / "demo", small_valley)
    monkeypatch.setattr(fundoscope.cli.valley, "valley_dir", lambda slug: tmp_path / slug)

    assert main(["valley", "refresh", "demo", "--dry-run"]) == 0
    assert "Valley demo:" in capsys.readouterr().out


def test_a_long_first_run_is_estimated_in_hours(tmp_path):
    big = valley("big", box(0, 8_000_000, 30_000, 8_030_000), 32718)

    text = format_plan(plan_refresh(big, tmp_path, TODAY))

    assert "To download: 96 requests and the land cover" in text
    assert text.endswith(" h).")
