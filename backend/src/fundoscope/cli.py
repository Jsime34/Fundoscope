"""Command-line entry point: `uv run fundoscope <command>`."""

import argparse
import math

# Tiny NDVI evalscript for the connectivity check; cloud/shadow pixels are masked via SCL.
_CHECK_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04", "B08", "SCL", "dataMask"] }],
    output: [
      { id: "ndvi", bands: 1, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}
const BAD_SCL = [0, 1, 3, 8, 9, 10];
function evaluatePixel(s) {
  const clear = s.dataMask === 1 && BAD_SCL.indexOf(s.SCL) === -1;
  return { ndvi: [(s.B08 - s.B04) / (s.B08 + s.B04)], dataMask: [clear ? 1 : 0] };
}
"""


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
    """Get a CDSE token, then run one tiny Statistical API request (a few PU)."""
    from sentinelhub import CRS, BBox, SentinelHubSession, SentinelHubStatistical

    from fundoscope.sentinel.cdse import s2_l2a, sh_config

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

    # ~1 km x 1 km box near Ica; sent in UTM 18S so the 10 m resolution is in meters.
    bbox = BBox((-75.74, -14.08, -75.73, -14.07), crs=CRS.WGS84).transform(CRS(32718))
    request = SentinelHubStatistical(
        aggregation=SentinelHubStatistical.aggregation(
            evalscript=_CHECK_EVALSCRIPT,
            time_interval=("2024-01-01", "2024-01-31"),
            aggregation_interval="P1D",
            resolution=(10, 10),
        ),
        input_data=[SentinelHubStatistical.input_data(s2_l2a())],
        bbox=bbox,
        config=config,
    )
    try:
        response = request.get_data()[0]
    except Exception as exc:
        print(f"Statistical API request FAILED: {exc}")
        return 1

    print("Statistical API OK. Mean NDVI per clear date, Jan 2024:")
    for day in response.get("data", []):
        stats = day["outputs"]["ndvi"]["bands"]["B0"]["stats"]
        mean = float(stats.get("mean", "nan"))
        if not math.isnan(mean):
            clear = 1 - stats["noDataCount"] / stats["sampleCount"]
            print(f"  {day['interval']['from'][:10]}  NDVI={mean:.3f}  clear={clear:.0%}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fundoscope",
        description="Satellite crop monitoring and parcel due diligence for coastal Peru.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-db", help="connect to PostGIS and print its version")
    sub.add_parser("check-cdse", help="test Copernicus Data Space credentials and API access")
    args = parser.parse_args(argv)

    if args.command == "check-db":
        return check_db()
    if args.command == "check-cdse":
        return check_cdse()
    return 1
