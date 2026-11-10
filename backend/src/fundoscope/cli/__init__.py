"""Command-line interface: `uv run fundoscope <command>`.

Submodules: `checks` (setup checks), `history` (a parcel's history), `plots` (PNG charts),
`labels` (checks the labeled parcels and downloads their training pixels).
"""

import argparse
from datetime import date
from pathlib import Path

from fundoscope.cli.checks import check_cdse, check_db
from fundoscope.cli.history import run_history
from fundoscope.cli.labels import run_labels_check, run_labels_extract
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
        "--active-area",
        action="store_true",
        help="also compute %% active area per campaign (about 7 openEO credits per campaign)",
    )
    history.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output folder (default data/private/history/<parcel name>)",
    )

    labels = sub.add_parser("labels", help="work with the labeled training parcels")
    labels_sub = labels.add_subparsers(dest="labels_command", required=True)
    labels_check = labels_sub.add_parser(
        "check", help="check the labels against FR-06 and count the parcels per crop class"
    )
    labels_check.add_argument(
        "file", type=Path, nargs="?", default=None, help="default data/labels/labels.gpkg"
    )
    labels_extract = labels_sub.add_parser(
        "extract", help="download the training pixels of every label (uses openEO credits)"
    )
    labels_extract.add_argument(
        "file", type=Path, nargs="?", default=None, help="default data/labels/labels.gpkg"
    )
    labels_extract.add_argument(
        "--used-credits",
        type=float,
        default=None,
        help="openEO credits already used this month (Copernicus dashboard); "
        "the run stops before 80 %% of the month's quota",
    )
    labels_extract.add_argument(
        "--cap", type=float, default=None, help="spend at most this many credits in this run"
    )
    labels_extract.add_argument(
        "--dry-run", action="store_true", help="only count what is left to download"
    )

    args = parser.parse_args(argv)
    if args.command == "check-db":
        return check_db()
    if args.command == "check-cdse":
        return check_cdse()
    if args.command == "labels" and args.labels_command == "extract":
        return run_labels_extract(args.file, args.used_credits, args.cap, dry_run=args.dry_run)
    if args.command == "labels":
        return run_labels_check(args.file)
    if args.command == "history":
        return run_history(args.file, args.start, args.end, args.out, active_area=args.active_area)
    return 1
