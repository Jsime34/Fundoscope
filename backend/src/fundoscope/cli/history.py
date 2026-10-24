"""`fundoscope history <file>`: a parcel's campaign-by-campaign history from Sentinel-2."""

import re
from datetime import date
from pathlib import Path

import pandas as pd

from fundoscope.cli.plots import history_chart
from fundoscope.config import get_settings
from fundoscope.ingest import ParcelError, zone_label
from fundoscope.pipeline import ParcelHistory, parcel_history
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
) -> int:
    """Download, compute, print, and save a parcel's history. Returns the exit code.

    With `active_area`, also downloads monthly composites (about 7 openEO credits and
    30 seconds per campaign, two campaigns at a time) to compute % active area.
    """
    try:
        data = file.read_bytes()
    except OSError as exc:
        print(f"Can't read {file}: {exc}")
        return 1

    options = {}
    if fetch:
        options["fetch"] = fetch
    if active_area:
        options["fetch_composites"] = fetch_composites or fetch_monthly_composites
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

    out = out or default_output_dir(history.parcel.name or file.stem)
    save_outputs(history, out)
    print(f"\nSaved the table, the series, and the chart to {out}")
    return 0


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


def format_table(campaigns: pd.DataFrame, with_active_area: bool = True) -> str:
    table = campaigns.rename(columns=TABLE_HEADERS)
    if not with_active_area:
        table = table.drop(columns="active", errors="ignore")
    return table.to_string(
        formatters={
            "peak NDVI": "{:.2f}".format,
            "mean NDVI": "{:.2f}".format,
            "mean NDWI": "{:.2f}".format,
            "integrated": "{:.0f}".format,
            "coverage": "{:.0%}".format,
            "complete": lambda done: "yes" if done else "partial",
            "active": lambda share: "—" if pd.isna(share) else f"{share:.0%}",
        }
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
