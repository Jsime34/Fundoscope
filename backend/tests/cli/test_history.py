from datetime import date
from pathlib import Path

import fundoscope.cli
from fundoscope.cli import main
from fundoscope.cli.history import default_output_dir, run_history


def test_history_prints_the_table_and_saves_all_files(tmp_path, parcel_geojson, fake_fetch, capsys):
    parcel_file = tmp_path / "lote.geojson"
    parcel_file.write_bytes(parcel_geojson)
    out = tmp_path / "out"

    code = run_history(parcel_file, date(2022, 8, 1), date(2024, 7, 31), out, fetch=fake_fetch)

    printed = capsys.readouterr().out
    assert code == 0
    assert "Lote demo · 9.00 ha · UTM 18S (Ica)" in printed
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

    def fake_run(file, start, end, out):
        received.update(file=file, start=start, end=end, out=out)
        return 0

    monkeypatch.setattr(fundoscope.cli, "run_history", fake_run)
    assert main(["history", "B4 48.kml", "--start", "2020-08-01", "--end", "2021-07-31"]) == 0
    assert received == {
        "file": Path("B4 48.kml"),
        "start": date(2020, 8, 1),
        "end": date(2021, 7, 31),
        "out": None,
    }
