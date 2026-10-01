"""Command-line interface: `uv run fundoscope <command>`.

Submodules: `checks` (setup checks), `history` (a parcel's history), `plots` (PNG charts).
"""

import argparse
from datetime import date
from pathlib import Path

from fundoscope.cli.checks import check_cdse, check_db
from fundoscope.cli.history import run_history
from fundoscope.sentinel import ARCHIVE_START


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fundoscope",
        description="Satellite crop monitoring and parcel due diligence for coastal Peru.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-db", help="connect to PostGIS and print its version")
    sub.add_parser("check-cdse", help="test Copernicus Data Space credentials and API access")

    history = sub.add_parser(
        "history", help="campaign-by-campaign history of a parcel (KML, KMZ, or GeoJSON file)"
    )
    history.add_argument("file", type=Path, help="parcel boundary file")
    history.add_argument(
        "--start",
        type=date.fromisoformat,
        default=ARCHIVE_START,
        help=f"first day, YYYY-MM-DD (default {ARCHIVE_START})",
    )
    history.add_argument(
        "--end", type=date.fromisoformat, default=None, help="last day (default today)"
    )
    history.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output folder (default data/private/history/<parcel name>)",
    )

    args = parser.parse_args(argv)
    if args.command == "check-db":
        return check_db()
    if args.command == "check-cdse":
        return check_cdse()
    if args.command == "history":
        return run_history(args.file, args.start, args.end, args.out)
    return 1
