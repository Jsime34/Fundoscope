from dataclasses import replace
from datetime import date
from pathlib import Path

import fundoscope.cli
import fundoscope.cli.history as history_cli
from fundoscope.analysis import Flag
from fundoscope.cli import main
from fundoscope.cli.history import (
    default_output_dir,
    format_flags,
    format_neighbors,
    run_history,
)
from fundoscope.pipeline import parcel_history


def test_history_prints_the_table_and_saves_all_files(tmp_path, parcel_geojson, fake_fetch, capsys):
    parcel_file = tmp_path / "lote.geojson"
    parcel_file.write_bytes(parcel_geojson)
    out = tmp_path / "out"

    code = run_history(parcel_file, date(2022, 8, 1), date(2024, 7, 31), out, fetch=fake_fetch)

    printed = capsys.readouterr().out
    assert code == 0
    assert "Lote demo · 9.00 ha · UTM 18S" in printed
    assert "2.50 processing units" in printed
    assert "2022/23" in printed and "2023/24" in printed
    for name in ("campaigns.csv", "observations.csv", "series.csv"):
        assert (out / name).exists()
    assert (out / "ndvi_history.png").stat().st_size > 10_000  # a real image was drawn


def test_default_output_is_the_private_folder():
    folder = default_output_dir("B4 48")
    assert folder.parts[-4:] == ("data", "private", "history", "B4-48")


def test_an_invalid_file_is_reported_and_nothing_is_downloaded(tmp_path, fake_fetch, capsys):
    bad = tmp_path / "broken.kml"
    bad.write_bytes(b"<kml><Placemark>")
    assert run_history(bad, date(2022, 8, 1), None, tmp_path, fetch=fake_fetch) == 1
    assert "Parcel rejected" in capsys.readouterr().out
    assert fake_fetch.calls == []


def test_a_missing_file_is_reported(tmp_path, capsys):
    assert run_history(tmp_path / "nope.kml", date(2022, 8, 1), None, None) == 1
    assert "Can't read" in capsys.readouterr().out


def test_main_passes_parsed_arguments(monkeypatch):
    received = {}

    def fake_run(file, start, end, out, active_area):
        received.update(file=file, start=start, end=end, out=out, active_area=active_area)
        return 0

    monkeypatch.setattr(fundoscope.cli, "run_history", fake_run)
    argv = ["history", "B4 48.kml", "--start", "2020-08-01", "--end", "2021-07-31"]
    assert main([*argv, "--active-area"]) == 0
    assert received == {
        "file": Path("B4 48.kml"),
        "start": date(2020, 8, 1),
        "end": date(2021, 7, 31),
        "out": None,
        "active_area": True,
    }


def test_history_with_active_area_shows_the_column_the_crop_and_the_credits(
    tmp_path, parcel_geojson, fake_fetch, fake_composites, fake_crop_model, capsys
):
    parcel_file = tmp_path / "lote.geojson"
    parcel_file.write_bytes(parcel_geojson)
    code = run_history(
        parcel_file,
        date(2022, 8, 1),
        date(2024, 7, 31),
        tmp_path / "out",
        active_area=True,
        fetch=fake_fetch,
        fetch_composites=fake_composites,
        model=fake_crop_model,
    )
    printed = capsys.readouterr().out
    assert code == 0
    assert "+ 14 openEO credits" in printed
    assert "active" in printed and "77%" in printed  # 23 of 30 pixel columns are green
    header = next(line for line in printed.splitlines() if "peak NDVI" in line)
    assert "crop" in header and "confidence" in header
    assert printed.count("grape") == 2 and "75%" in printed


def test_without_a_trained_model_the_crop_is_left_out(
    monkeypatch, tmp_path, parcel_geojson, fake_fetch, fake_composites, capsys
):
    monkeypatch.setattr(history_cli, "load_crop_model", lambda folder: None)
    parcel_file = tmp_path / "lote.geojson"
    parcel_file.write_bytes(parcel_geojson)

    run_history(
        parcel_file,
        date(2022, 8, 1),
        date(2024, 7, 31),
        tmp_path / "out",
        active_area=True,
        fetch=fake_fetch,
        fetch_composites=fake_composites,
    )

    printed = capsys.readouterr().out
    assert "No crop model yet (run `fundoscope model train`)" in printed
    header = next(line for line in printed.splitlines() if "peak NDVI" in line)
    assert "active" in header and "crop" not in header


def test_history_without_active_area_hides_the_column(tmp_path, parcel_geojson, fake_fetch, capsys):
    parcel_file = tmp_path / "lote.geojson"
    parcel_file.write_bytes(parcel_geojson)
    run_history(
        parcel_file, date(2022, 8, 1), date(2024, 7, 31), tmp_path / "out", fetch=fake_fetch
    )
    table_header = [line for line in capsys.readouterr().out.splitlines() if "peak NDVI" in line][0]
    assert "active" not in table_header


def test_a_campaign_in_progress_shows_dashes_for_its_crop(
    tmp_path, parcel_geojson, fake_fetch, fake_composites, fake_crop_model, capsys
):
    parcel_file = tmp_path / "lote.geojson"
    parcel_file.write_bytes(parcel_geojson)

    run_history(
        parcel_file,
        date(2022, 8, 1),
        date(2023, 12, 31),  # 2023/24 has only August to December so far
        tmp_path / "out",
        active_area=True,
        fetch=fake_fetch,
        fetch_composites=fake_composites,
        model=fake_crop_model,
    )

    row = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("2023/24"))
    assert row.split()[-2:] == ["—", "—"]
    assert "None" not in row and "NaN" not in row


def test_flags_are_printed_after_the_table(tmp_path, parcel_geojson, fake_fetch, capsys):
    parcel_file = tmp_path / "lote.geojson"
    parcel_file.write_bytes(parcel_geojson)

    run_history(
        parcel_file, date(2022, 8, 1), date(2024, 7, 31), tmp_path / "out", fetch=fake_fetch
    )

    assert "Flags: none." in capsys.readouterr().out


def test_each_flag_is_one_line_with_its_campaign(parcel_geojson, fake_fetch):
    history = parcel_history(
        parcel_geojson, "lote.geojson", date(2022, 8, 1), date(2024, 7, 31), fetch=fake_fetch
    )
    flags = [
        Flag("2022/23", "low data", "Low data: only 9 clear dates."),
        Flag("2023/24", "crop change", "Crop change: onion, then alfalfa."),
    ]

    assert format_flags(replace(history, flags=flags)) == (
        "Flags:\n"
        "  2022/23  Low data: only 9 clear dates.\n"
        "  2023/24  Crop change: onion, then alfalfa."
    )


def test_the_neighbor_bands_are_printed_for_a_parcel_in_a_counted_valley(
    tmp_path, neighbor_valley, fake_fetch, capsys
):
    valleys, parcel_file = neighbor_valley
    path = tmp_path / "vecino.geojson"
    path.write_bytes(parcel_file)

    run_history(
        path,
        date(2022, 8, 1),
        date(2024, 7, 31),
        tmp_path / "out",
        fetch=fake_fetch,
        valleys=valleys,
    )

    printed = capsys.readouterr().out
    assert "Neighbors (farmland 50 m to 2 km around the parcel, valley demo):" in printed
    assert "below p10" in printed and "UNDERPERFORMING" in printed
    assert "2026/27 (4 mo)" in printed


def test_outside_the_pilot_area_the_neighbors_get_a_note(
    tmp_path, parcel_geojson, fake_fetch, capsys
):
    path = tmp_path / "lote.geojson"
    path.write_bytes(parcel_geojson)

    run_history(
        path,
        date(2022, 8, 1),
        date(2024, 7, 31),
        tmp_path / "out",
        fetch=fake_fetch,
        valleys=tmp_path,
    )

    assert "Neighbors: The neighbor comparison is only available" in capsys.readouterr().out


def test_without_a_comparison_the_neighbors_say_so(parcel_geojson, fake_fetch):
    history = parcel_history(
        parcel_geojson, "lote.geojson", date(2022, 8, 1), date(2024, 7, 31), fetch=fake_fetch
    )

    assert format_neighbors(history) == "Neighbors: not compared."
