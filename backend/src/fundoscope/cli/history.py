"""`fundoscope history <file>`: a parcel's campaign-by-campaign history from Sentinel-2."""

import re
from datetime import date
from pathlib import Path

import pandas as pd

from fundoscope.cli.plots import UTM_REGION, history_chart
from fundoscope.config import get_settings
from fundoscope.ingest import ParcelError
from fundoscope.pipeline import ParcelHistory, parcel_history
from fundoscope.pipeline.parcel_history import Fetcher
from fundoscope.sentinel import SentinelError

TABLE_HEADERS = {
    "peak_ndvi": "peak NDVI",
    "mean_ndvi": "mean NDVI",
    "mean_ndwi": "mean NDWI",
    "integrated_ndvi": "integrated",
    "n_obs": "dates",
    "coverage": "coverage",
    "complete": "complete",
}


def run_history(
    file: Path, start: date, end: date | None, out: Path | None, fetch: Fetcher | None = None
) -> int:
    """Download, compute, print, and save a parcel's history. Returns the exit code."""
    try:
        data = file.read_bytes()
    except OSError as exc:
        print(f"Can't read {file}: {exc}")
        return 1

    print(f"Downloading Sentinel-2 statistics for {file.name} ...")
    try:
        history = parcel_history(data, file.name, start, end, **({"fetch": fetch} if fetch else {}))
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
        print("\n" + format_table(history.campaigns))

    out = out or default_output_dir(history.parcel.name or file.stem)
    save_outputs(history, out)
    print(f"\nSaved the table, the series, and the chart to {out}")
    return 0


def summary(history: ParcelHistory) -> str:
    parcel, daily, clean = history.parcel, history.daily, history.clean
    region = UTM_REGION.get(parcel.utm_epsg, f"EPSG:{parcel.utm_epsg}")
    obs = daily.observations
    lines = [f"Parcel: {parcel.name or '(unnamed)'} · {parcel.area_ha:.2f} ha · {region}"]
    if not obs.empty:
        lines.append(
            f"Satellite dates: {len(obs)} ({obs.index[0]:%Y-%m-%d} to {obs.index[-1]:%Y-%m-%d}); "
            f"removed {clean.dropped_cloudy} cloudy and {clean.dropped_dips} cloud dips"
        )
    lines.append(
        f"Cost: {daily.processing_units:.2f} processing units in {daily.requests} requests"
    )
    if parcel.repaired:
        lines.append("Note: the outline was invalid and was repaired; check the drawing.")
    if daily.failed_intervals:
        lines.append(f"Note: the service could not process {daily.failed_intervals} dates.")
    return "\n".join(lines)


def format_table(campaigns: pd.DataFrame) -> str:
    table = campaigns.rename(columns=TABLE_HEADERS)
    return table.to_string(
        formatters={
            "peak NDVI": "{:.2f}".format,
            "mean NDVI": "{:.2f}".format,
            "mean NDWI": "{:.2f}".format,
            "integrated": "{:.0f}".format,
            "coverage": "{:.0%}".format,
            "complete": lambda done: "yes" if done else "partial",
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
