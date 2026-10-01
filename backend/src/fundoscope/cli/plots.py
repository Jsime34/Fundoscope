"""PNG charts for the command line. (The web app will draw its own interactive charts.)

Uses matplotlib's object API (`Figure`), not `pyplot`, so it keeps no global state and is
safe to call from a server process later. Colors follow the project's chart palette
(light mode): one hue for the single measure, muted ink for text and chrome.
"""

from pathlib import Path

import pandas as pd

from fundoscope.pipeline import ParcelHistory
from fundoscope.timeseries import campaign_bounds

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, BAND = "#e1e0d9", "#c3c2b7", "#f0efec"
SERIES, SERIES_LIGHT = "#2a78d6", "#86b6ef"

UTM_REGION = {32717: "UTM 17S (Piura, La Libertad)", 32718: "UTM 18S (Ica)"}


def history_chart(history: ParcelHistory, path: Path) -> None:
    """Save the parcel's NDVI history (clear dates + smoothed curve, by campaign) as a PNG."""
    import matplotlib
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    grid = history.clean.grid
    clear = history.clean.observations
    if grid.empty:
        raise ValueError("No clear satellite images: nothing to plot.")

    style = {"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"]}
    with matplotlib.rc_context({**style, "font.size": 10}):
        fig = Figure(figsize=(12, 4.6), dpi=200, facecolor=SURFACE)
        ax = fig.subplots()
        ax.set_facecolor(SURFACE)

        for i, label in enumerate(history.campaigns.index):
            start, end = (pd.Timestamp(d) for d in campaign_bounds(label))
            start, end = max(start, grid.index[0]), min(end, grid.index[-1])
            if i % 2 == 0:
                ax.axvspan(start, end, color=BAND, zorder=0, lw=0)
            complete = bool(history.campaigns.loc[label, "complete"])
            ax.text(
                start + (end - start) / 2,
                1.02,
                label + ("" if complete else "\n(partial)"),
                ha="center",
                va="bottom",
                color=MUTED,
                fontsize=8.5,
                transform=ax.get_xaxis_transform(),
            )

        ax.scatter(clear.index, clear["ndvi"], s=9, color=SERIES_LIGHT, lw=0, zorder=2)
        ax.plot(grid.index, grid["ndvi_smooth"], color=SERIES, lw=2, zorder=3)

        ax.set_ylim(0, 1)
        ax.set_xlim(grid.index[0], grid.index[-1] + pd.Timedelta(days=20))
        ax.set_ylabel("NDVI", color=INK_2)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(colors=MUTED, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)

        parcel = history.parcel
        name = parcel.name or "Parcel"
        region = UTM_REGION.get(parcel.utm_epsg, f"EPSG:{parcel.utm_epsg}")
        fig.suptitle(
            f"{name}: NDVI since {grid.index[0]:%B %Y}",
            x=0.06,
            y=0.99,
            ha="left",
            color=INK,
            fontsize=13,
            fontweight="bold",
        )
        fig.text(
            0.06,
            0.915,
            f"{parcel.area_ha:.2f} ha · {region} · Sentinel-2 L2A · "
            "shaded bands = agricultural campaigns (Aug–Jul)",
            color=INK_2,
            fontsize=9,
        )
        handles = [
            Line2D(
                [], [], ls="", marker="o", ms=4, color=SERIES_LIGHT, label="clear satellite date"
            ),
            Line2D([], [], color=SERIES, lw=2, label="cleaned & smoothed"),
        ]
        fig.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(0.97, 0.955),
            ncol=2,
            frameon=False,
            labelcolor=INK_2,
            fontsize=9,
        )
        years = f"{grid.index[0]:%Y}–{grid.index[-1]:%Y}"
        fig.text(
            0.97,
            0.01,
            f"Contains modified Copernicus Sentinel data [{years}]",
            ha="right",
            va="bottom",
            color=MUTED,
            fontsize=7.5,
        )
        fig.subplots_adjust(left=0.06, right=0.97, top=0.80, bottom=0.10)
        fig.savefig(path, facecolor=SURFACE)
