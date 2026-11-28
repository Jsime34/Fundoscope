"""`fundoscope valley refresh <slug>`: count the crop hectares of a valley (FR-14).

The valley's area of interest is data/valley/<slug>/aoi.geojson. The first run downloads three
campaigns of composites (about two hours); later runs download only what is new. Each count is
saved in PostGIS, so the database must be running.
"""

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy.exc import OperationalError

from fundoscope.cli.model import default_models_dir
from fundoscope.db import SqlValleyStore, StoredValleyRun, ValleyStore
from fundoscope.ingest import ParcelError
from fundoscope.pipeline.valley_refresh import (
    WORKERS,
    CountResult,
    Download,
    Plan,
    Refresh,
    plan_refresh,
    refresh_valley,
)
from fundoscope.pipeline.valley_store import (
    MAPS_DIR,
    ValleyError,
    load_valley,
    valley_dir,
)
from fundoscope.sentinel import CreditBudget, batch_cap
from fundoscope.valley import Tile

TYPICAL_REQUEST_CREDITS = 7.0  # measured: one 100 km² box, 1 or 3 months
REQUEST_SECONDS = 45  # measured overhead of one request...
MONTH_SECONDS = 40  # ...plus this per month of a ~70 km² tile
LAND_COVER_SECONDS = 60
NAME_WIDTH = 31
NO_DATABASE = (
    "The database isn't running, and the count is saved there: start it "
    "(docker compose up -d db) and run the command again."
)


def run_valley_refresh(
    slug: str,
    used_credits: float | None,
    cap: float | None,
    *,
    dry_run: bool = False,
    folder: Path | None = None,
    models: Path | None = None,
    store: ValleyStore | None = None,
    today: date | None = None,
    **options: Any,
) -> int:
    """Plan, download within the quota, count, print the hectares, and save them in `store`
    (PostGIS by default). Returns the exit code.

    The run may spend at most 80 % of the month's free openEO credits minus `used_credits`,
    and at most `cap` credits; at least one of the two is required when something is left to
    download. A dry run only prints the plan. Otherwise the store is checked first, so a
    database that is down stops the run before anything is downloaded. `options` go to
    `refresh_valley` (tests replace the openEO requests).
    """
    folder = folder or valley_dir(slug)
    try:
        valley_ = load_valley(folder, slug)
    except (ValleyError, ParcelError) as exc:
        print(str(exc))
        return 1
    plan = plan_refresh(valley_, folder, today or date.today())
    print(format_plan(plan))
    if dry_run:
        return 0
    store = store or SqlValleyStore()
    try:
        print(format_last_run(store.latest_run(slug)))
    except OperationalError:
        print(NO_DATABASE)
        return 1

    limit = 0.0
    if plan.downloads:
        limits = [cap, None if used_credits is None else batch_cap(used_credits)]
        if limits == [None, None]:
            print(
                "Pass --used-credits N, the openEO credits already used this month (Billing "
                "tab of https://marketplace-portal.dataspace.copernicus.eu), or --cap N."
            )
            return 1
        limit = min(value for value in limits if value is not None)
        print(f"This run may spend up to {limit:,.0f} credits.")
    try:
        refresh = refresh_valley(
            plan,
            folder,
            models or default_models_dir(),
            CreditBudget(limit),
            store=store,
            on_download=_download_progress(plan.downloads),
            on_tile=_tile_progress(len(valley_.tiles)),
            **options,
        )
    except ValleyError as exc:
        print(str(exc))
        return 1
    except OperationalError:
        print(
            "The count is done, but the database stopped before it was saved. Start it and "
            "run the command again: the images are on disk, so it costs no credits."
        )
        return 1
    print(format_refresh(refresh, folder))
    return 0 if refresh.run else 1


def format_last_run(run: StoredValleyRun | None) -> str:
    """The valley's previous count, if any."""
    if run is None:
        return "No count saved yet: this is the first."
    return (
        f"Last count: images up to {run.as_of:%Y-%m-%d}, made "
        f"{run.run_at.astimezone():%Y-%m-%d %H:%M}."
    )


def format_plan(plan: Plan) -> str:
    """What the refresh will count and download."""
    when, valley_ = plan.season, plan.valley
    lines = [
        f"Valley {valley_.slug}: {valley_.area_km2 * 100:,.0f} ha in {len(valley_.tiles)} tiles. "
        f"Images up to {when.as_of:%Y-%m-%d}.",
        "Counts: " + "; ".join(count.label for count in when.counts) + ".",
    ]
    if not plan.downloads:
        lines.append("Nothing to download: every month is on disk.")
        return "\n".join(lines)
    seconds = sum(REQUEST_SECONDS + MONTH_SECONDS * len(r.months) for r in plan.requests)
    seconds = seconds / WORKERS + LAND_COVER_SECONDS * plan.needs_land_cover
    what = f"{len(plan.requests)} requests" + (
        " and the land cover" if plan.needs_land_cover else ""
    )
    lines.append(
        f"To download: {what} (about {plan.downloads * TYPICAL_REQUEST_CREDITS:,.0f} openEO "
        f"credits and {_duration(seconds)})."
    )
    return "\n".join(lines)


def format_refresh(refresh: Refresh, folder: Path) -> str:
    """The downloads, then the hectares of every count."""
    lines = []
    if refresh.downloads:
        lines.append(
            f"\nDownloaded {len(refresh.downloads) - len(refresh.failed)} of "
            f"{refresh.plan.downloads} ({refresh.credits:,.1f} openEO credits)."
        )
    for item in refresh.failed:
        lines.append(f"  Failed: {item.name}: {item.error}")
    if refresh.missing:
        if refresh.stopped_by_failures:
            lines.append(
                "Stopped after 3 failures in a row: Copernicus may be down, or the credentials "
                "may be wrong (`fundoscope check-cdse`)."
            )
        elif refresh.stopped_at_cap:
            lines.append(
                "Stopped at the credit cap. Run the command again to continue, with the new "
                "--used-credits from the dashboard."
            )
        lines.append(
            f"{refresh.missing} downloads are still missing, so nothing was counted. Run the "
            "command again to continue."
        )
        return "\n".join(lines)

    newest = refresh.newest_image
    lines.append(
        f"\nFarmland: {refresh.farmland_ha:,.0f} ha. Newest image: "
        + (f"{newest:%Y-%m-%d}." if newest else "unknown (the catalog didn't answer).")
    )
    for result in refresh.counts:
        lines += ["", *format_count(result)]
    if len(refresh.counts) > 1 and refresh.counts[0].count.in_progress:
        lines += ["", *format_comparison(refresh.counts[0], refresh.counts[1])]
    lines += [
        "",
        f"Saved in the database (valley_runs); class maps and images in {folder / MAPS_DIR} "
        f"({refresh.seconds / 60:.0f} min).",
    ]
    return "\n".join(lines)


def format_count(result: CountResult) -> list[str]:
    """One count: mapped hectares, and the adjusted hectares with their 95 % range."""
    count = result.count
    lines = [
        f"{count.label} ({count.months} months, model {result.model}):",
        f"  {'Class':<{NAME_WIDTH}}{'Hectares':>10}   Adjusted (95 % range)",
    ]
    for area in result.areas:
        line = f"  {area.name:<{NAME_WIDTH}}{area.hectares:>10,.0f}"
        if area.adjusted is not None:
            line += f"   {area.adjusted:,.0f} ({area.low:,.0f}-{area.high:,.0f})"
        lines.append(line)
    return lines


def format_comparison(current: CountResult, previous: CountResult) -> list[str]:
    """The current campaign against the previous one at the same date."""
    before = {area.name: area.hectares for area in previous.areas}
    now, then = current.count.campaign, previous.count.campaign
    lines = [
        f"Compared with {previous.count.label} (same date, same model):",
        f"  {'Class':<{NAME_WIDTH}}{now:>10}{then:>10}   Change",
    ]
    for area in current.areas:
        old = before.get(area.name, 0.0)
        change = area.hectares - old
        percent = f" ({change / old:+.0%})" if old else ""
        lines.append(
            f"  {area.name:<{NAME_WIDTH}}{area.hectares:>10,.0f}{old:>10,.0f}   "
            f"{change:+,.0f}{percent}"
        )
    return lines


def _duration(seconds: float) -> str:
    minutes = round(seconds / 60)
    return f"{minutes} min" if minutes < 90 else f"{minutes / 60:.1f} h"


def _download_progress(total: int) -> Callable[[Download], None]:
    done = 0

    def show(item: Download) -> None:
        nonlocal done
        done += 1
        outcome = f"FAILED: {item.error}" if item.error else f"{item.credits:g} credits"
        print(f"  [{done:>{len(str(total))}}/{total}] {item.name}: {outcome}, {item.seconds:.0f} s")

    return show


def _tile_progress(total: int) -> Callable[[Tile], None]:
    done = 0

    def show(tile: Tile) -> None:
        nonlocal done
        done += 1
        if done == 1:
            print("Classifying the tiles...")
        print(f"  {tile.id} ({done}/{total})")

    return show
