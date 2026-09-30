"""Smoke tests: the package imports, settings load, the API answers, and the CLI parses."""

import pytest
from fastapi.testclient import TestClient

from fundoscope.api.main import app
from fundoscope.cli import main
from fundoscope.config import Settings


def test_health_endpoint():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_settings_read_environment(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@db:5432/x")
    monkeypatch.setenv("QUOTA_PU_MONTHLY_CAP", "123")
    settings = Settings()
    assert settings.database_url == "postgresql+psycopg://u:p@db:5432/x"
    assert settings.quota_pu_monthly_cap == 123


def test_cdse_secret_is_not_printed(monkeypatch):
    monkeypatch.setenv("CDSE_CLIENT_SECRET", "super-secret")
    assert "super-secret" not in repr(Settings())


def test_cli_requires_a_command():
    with pytest.raises(SystemExit):
        main([])
