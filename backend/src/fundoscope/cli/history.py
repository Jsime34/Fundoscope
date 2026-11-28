"""`fundoscope history <file>`: a parcel's campaign-by-campaign history from Sentinel-2."""

import re
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy.exc import OperationalError

from fundoscope.cli.plots import history_chart
from fundoscope.config import get_settings
from fundoscope.db import SqlValleyStore, engine
from fundoscope.ingest import ParcelError, zone_label
from fundoscope.ml import CropModel
from fundoscope.pipeline import CountedValleys, ParcelHistory, load_crop_model, parcel_history
from fundoscope.pipeline.parcel_history import CompositeFetcher, Fetcher
from fundoscope.sentinel import SentinelError, fetch_monthly_composites

TABLE_HEADERS = {
    "peak_ndvi": "peak NDVI",
    "mean_ndvi": "mean NDVI",
    "mean_ndwi": "mean NDWI",
    "integrated_ndvi": "integrated",
    "n_obs": "dates",
    "coverage": "coverage",
    "complete": "complete",
    "active_area": "active",
    "crop": "crop",
    "crop_confidence": "confidence",
}


def run_history(
    file: Path,
    start: date,
    end: date | None,
    out: Path | None,
    *,
    active_area: bool = False,
    fetch: Fetcher | None = None,
    fetch_composites: CompositeFetcher | None = None,
    model: CropModel | None = None,
    valleys: CountedValleys | None = None,
) -> int:
    """Download, compute, print, and save a parcel's history. Returns the exit code.

    With `active_area`, also downloads monthly composites (about 7 openEO credits and
    30 seconds per campaign, two campaigns at a time) to compute % active area and, with the
    model saved by `fundoscope model train` (or `model`), the crop of each campaign. A parcel
    inside a counted valley is compared with its neighbors (`valleys`; by default the
    valleys in data/valley with their counts in PostGIS, skipped if the database is down).
    """
    try:
        data = file.read_bytes()
    except OSError as exc:
        print(f"Can't read {file}: {exc}")
        return 1

    options: dict = {"valleys": valleys or default_valleys()}
    if options["valleys"] is None:
        print("The database isn't running, so the parcel isn't compared with its neighbors.")
    if fetch:
        options["fetch"] = fetch
    if active_area:
        options["fetch_composites"] = fetch_composites or fetch_monthly_composites
        options["model"] = model or load_crop_model(get_settings().models_dir)
        if options["model"] is None:
            print("No crop model yet (run `fundoscope model train`), so no crop per campaign.")
    extra = " and monthly composites" if active_area else ""
    print(f"Downloading Sentinel-2 statistics{extra} for {file.name} ...")
    try:
        history = parcel_history(data, file.name, start, end, **options)
    except ParcelError as exc:
        print(f"Parcel rejected: {exc}")
        return 1
    except SentinelError as exc:
        print(f"Satellite data problem: {exc}")
        return 1

    print(summary(history))
    if history.campaigns.empty:
        print("\nNo clear satellite image in this period, so there is nothing to summarize.")
    else:
        print("\n" + format_table(history.campaigns, with_active_area=active_area))
        print("\n" + format_flags(history))
        print("\n" + format_neighbors(history))

    out = out or default_output_dir(history.parcel.name or file.stem)
    save_outputs(history, out)
    print(f"\nSaved the table, the series, and the chart to {out}")
    return 0


def default_valleys() -> CountedValleys | None:
    """The valleys in data/valley with their counts in PostGIS; None if the database is down."""
    try:
        with engine().connect():
            pass
    except OperationalError:
        return None
    return CountedValleys(get_settings().data_dir / "valley", SqlValleyStore())


def summary(history: ParcelHistory) -> str:
    parcel, daily, clean = history.parcel, history.daily, history.clean
    region = zone_label(parcel.utm_epsg)
    obs = daily.observations
    lines = [f"Parcel: {parcel.name or '(unnamed)'} · {parcel.area_ha:.2f} ha · {region}"]
    if not obs.empty:
        lines.append(
            f"Satellite dates: {len(obs)} ({obs.index[0]:%Y-%m-%d} to {obs.index[-1]:%Y-%m-%d}); "
            f"removed {clean.dropped_cloudy} cloudy and {clean.dropped_dips} cloud dips"
        )
    cost = f"Cost: {daily.processing_units:.2f} processing units in {daily.requests} requests"
    if history.credits:
        cost += f" + {history.credits:.0f} openEO credits"
    lines.append(cost)
    if parcel.repaired:
        lines.append("Note: the outline was invalid and was repaired; check the drawing.")
    if daily.failed_intervals:
        lines.append(f"Note: the service could not process {daily.failed_intervals} dates.")
    if history.active_area_failures:
        lines.append(
            f"Note: openEO failed for {history.active_area_failures} campaigns; "
            "their active area is left empty."
        )
    return "\n".join(lines)


def format_neighbors(history: ParcelHistory) -> str:
    """FR-08: each campaign's band among the farmland 50 m to 2 km around the parcel."""
    report = history.neighbors
    if report is None:
        return "Neighbors: not compared."
    if not report.bands:
        return f"Neighbors: {report.note}"
    lines = [f"Neighbors (farmland 50 m to 2 km around the parcel, valley {report.valley}):"]
    for band in report.bands:
        label = band.campaign if band.months == 12 else f"{band.campaign} ({band.months} mo)"
        mark = "  UNDERPERFORMING" if band.underperforming else ""
        lines.append(
            f"  {label:<16} mean NDVI {band.parcel_ndvi:.2f}  {band.band:<9}  (neighbors: "
            f"p10 {band.percentiles[10]:.2f}, median {band.percentiles[50]:.2f}, "
            f"p90 {band.percentiles[90]:.2f}; {band.neighbor_pixels:,} pixels){mark}"
        )
    return "\n".join(lines)


def format_flags(history: ParcelHistory) -> str:
    if not history.flags:
        return "Flags: none."
    return "\n".join(["Flags:", *(f"  {flag.campaign}  {flag.message}" for flag in history.flags)])


def format_table(campaigns: pd.DataFrame, with_active_area: bool = True) -> str:
    table = campaigns.rename(columns=TABLE_HEADERS)
    if not with_active_area:
        table = table.drop(columns="active", errors="ignore")
    if not with_active_area or table["crop"].isna().all():
        table = table.drop(columns=["crop", "confidence"])
    else:
        table["crop"] = table["crop"].fillna("—")  # a campaign in progress has no crop yet
    return table.to_string(
        na_rep="—",  # pandas skips the formatters for empty values
        formatters={
            "peak NDVI": "{:.2f}".format,
            "mean NDVI": "{:.2f}".format,
            "mean NDWI": "{:.2f}".format,
            "integrated": "{:.0f}".format,
            "coverage": "{:.0%}".format,
            "complete": lambda done: "yes" if done else "partial",
            "active": "{:.0%}".format,
            "confidence": "{:.0%}".format,
        },
    )


def default_output_dir(parcel_name: str) -> Path:
    """data/private/history/<parcel>: git-ignored, because parcel data is private."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", parcel_name).strip("-") or "parcel"
    return get_settings().data_dir / "private" / "history" / slug


def save_outputs(history: ParcelHistory, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    history.campaigns.to_csv(out / "campaigns.csv")
    history.daily.observations.to_csv(out / "observations.csv")
    history.clean.grid.to_csv(out / "series.csv")
    if not history.clean.grid.empty:
        history_chart(history, out / "ndvi_history.png")
