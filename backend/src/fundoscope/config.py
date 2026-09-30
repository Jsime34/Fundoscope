"""Application settings, read from environment variables and the repo-root `.env` file."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/fundoscope/config.py -> parents: fundoscope, src, backend, <repo root>
REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql+psycopg://fundoscope:fundoscope@localhost:5433/fundoscope"

    # Copernicus Data Space Ecosystem (CDSE) endpoints, not the commercial Sentinel Hub ones.
    cdse_client_id: str = ""
    cdse_client_secret: SecretStr = SecretStr("")
    cdse_base_url: str = "https://sh.dataspace.copernicus.eu"
    cdse_token_url: str = (
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    )

    quota_pu_monthly_cap: int = 8000
    data_dir: Path = REPO_ROOT / "data"


@lru_cache
def get_settings() -> Settings:
    return Settings()
