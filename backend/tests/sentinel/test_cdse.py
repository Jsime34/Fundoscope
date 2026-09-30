from fundoscope.config import get_settings
from fundoscope.sentinel import s2_l2a, sh_config


def test_config_points_at_copernicus_data_space_with_our_client(monkeypatch):
    monkeypatch.setenv("CDSE_CLIENT_ID", "test-id")
    monkeypatch.setenv("CDSE_CLIENT_SECRET", "test-secret")
    get_settings.cache_clear()
    try:
        config = sh_config()
    finally:
        get_settings.cache_clear()  # don't leak the fake credentials into other tests
    assert config.sh_client_id == "test-id"
    assert config.sh_client_secret == "test-secret"
    assert config.sh_base_url == "https://sh.dataspace.copernicus.eu"
    assert config.sh_token_url.startswith("https://identity.dataspace.copernicus.eu/")


def test_sentinel2_collection_is_served_by_cdse():
    assert s2_l2a().service_url == "https://sh.dataspace.copernicus.eu"
