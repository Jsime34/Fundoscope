"""Setup checks: `fundoscope check-db` and `fundoscope check-cdse`."""

from datetime import date


def check_db() -> int:
    """Connect to PostGIS and print its version."""
    from sqlalchemy import create_engine, text

    from fundoscope.config import get_settings

    engine = create_engine(get_settings().database_url)
    query = text("SELECT current_setting('server_version'), postgis_lib_version()")
    try:
        with engine.connect() as conn:
            pg_version, postgis_version = conn.execute(query).one()
    except Exception as exc:
        print(f"Database check FAILED: {exc}")
        return 1
    print(f"Database OK: PostgreSQL {pg_version}, PostGIS {postgis_version}")
    return 0


def check_cdse() -> int:
    """Get a CDSE token, then fetch one month of daily stats for a public box near Ica (~1 PU)."""
    from sentinelhub import CRS, BBox, SentinelHubSession

    from fundoscope.sentinel import fetch_daily_stats, sh_config

    config = sh_config()
    if not config.sh_client_id or not config.sh_client_secret:
        print("CDSE check FAILED: set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET in .env")
        return 1
    try:
        SentinelHubSession(config=config)
    except Exception as exc:
        print(f"CDSE token FAILED (check the client ID/secret): {exc}")
        return 1
    print("CDSE token OK")

    # ~1 km x 1 km box near Ica, in UTM 18S so the 10 m resolution is in meters.
    box = BBox((-75.74, -14.08, -75.73, -14.07), crs=CRS.WGS84).transform(CRS(32718))
    try:
        result = fetch_daily_stats(
            box.geometry, 32718, date(2024, 1, 1), date(2024, 1, 31), config=config
        )
    except Exception as exc:
        print(f"Statistical API request FAILED: {exc}")
        return 1

    print(f"Statistical API OK ({result.processing_units:.2f} processing units). Jan 2024:")
    for day, row in result.observations.iterrows():
        values = (
            "fully cloudy"
            if row["valid_fraction"] == 0
            else (f"NDVI={row['ndvi']:.3f}  NDWI={row['ndwi']:.3f}")
        )
        print(f"  {day:%Y-%m-%d}  clear={row['valid_fraction']:.0%}  {values}")
    return 0
