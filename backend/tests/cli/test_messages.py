"""The error and edge-case messages a user can see from the CLI (no network needed)."""

from datetime import date

import pytest

from fundoscope.cli.checks import check_cdse, check_db
from fundoscope.cli.history import run_history
from fundoscope.cli.plots import history_chart
from fundoscope.config import get_settings
from fundoscope.pipeline import parcel_history
from fundoscope.sentinel import SentinelError


@pytest.fixture
def parcel_file(tmp_path, parcel_geojson):
    path = tmp_path / "lote.geojson"
    path.write_bytes(parcel_geojson)
    return path


@pytest.fixture
def fresh_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_satellite_service_problem_is_reported(parcel_file, tmp_path, capsys):
    def failing_fetch(*_args):
        raise SentinelError("status 'FAILED'")

    assert run_history(parcel_file, date(2022, 8, 1), None, tmp_path, fetch=failing_fetch) == 1
    assert "Satellite data problem: status 'FAILED'" in capsys.readouterr().out


def test_a_period_without_clear_images_is_explained(parcel_file, tmp_path, fake_fetch, capsys):
    def cloudy_fetch(*args):
        stats = fake_fetch(*args)
        stats.observations["valid_fraction"] = 0.0
        return stats

    out = tmp_path / "out"
    assert (
        run_history(parcel_file, date(2022, 8, 1), date(2022, 12, 31), out, fetch=cloudy_fetch) == 0
    )
    assert "No clear satellite image" in capsys.readouterr().out
    assert (out / "observations.csv").exists() and not (out / "ndvi_history.png").exists()


def test_chart_refuses_an_empty_series(parcel_geojson, fake_fetch, tmp_path):
    def cloudy_fetch(*args):
        stats = fake_fetch(*args)
        stats.observations["valid_fraction"] = 0.0
        return stats

    history = parcel_history(
        parcel_geojson, "lote.geojson", date(2022, 8, 1), date(2022, 12, 31), fetch=cloudy_fetch
    )
    with pytest.raises(ValueError, match="nothing to plot"):
        history_chart(history, tmp_path / "chart.png")


def test_check_cdse_without_credentials(monkeypatch, fresh_settings, capsys):
    monkeypatch.setenv("CDSE_CLIENT_ID", "")
    monkeypatch.setenv("CDSE_CLIENT_SECRET", "")
    assert check_cdse() == 1
    assert "set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET" in capsys.readouterr().out


def test_check_db_reports_an_unreachable_database(monkeypatch, fresh_settings, capsys):
    # Port 1 on this machine: nothing listens there, so the connection is refused at once.
    url = "postgresql+psycopg://user:pass@127.0.0.1:1/none?connect_timeout=3"
    monkeypatch.setenv("DATABASE_URL", url)
    assert check_db() == 1
    assert "Database check FAILED" in capsys.readouterr().out
