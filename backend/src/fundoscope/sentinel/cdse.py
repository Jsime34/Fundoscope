"""Connection to the Copernicus Data Space Ecosystem (CDSE) Sentinel Hub APIs."""

from functools import lru_cache

from sentinelhub import DataCollection, SHConfig

from fundoscope.config import get_settings


def sh_config() -> SHConfig:
    """Return an SHConfig that points at the CDSE endpoints with our OAuth client."""
    settings = get_settings()
    config = SHConfig()
    config.sh_client_id = settings.cdse_client_id
    config.sh_client_secret = settings.cdse_client_secret.get_secret_value()
    config.sh_base_url = settings.cdse_base_url
    config.sh_token_url = settings.cdse_token_url
    return config


@lru_cache
def s2_l2a() -> DataCollection:
    """Sentinel-2 Level-2A (surface reflectance) as served by CDSE."""
    return DataCollection.SENTINEL2_L2A.define_from(
        "s2l2a", service_url=get_settings().cdse_base_url
    )
