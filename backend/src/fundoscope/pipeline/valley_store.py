"""The files of a counted valley, in data/valley/<slug>/ (FR-14).

- `aoi.geojson`: the area of interest, a polygon in WGS84 (drawn in QGIS);
- `grid.json`: the tile grid, saved by the first refresh. The months on disk belong to it, so
  later refreshes keep it: the area of interest may change, but only inside the grid;
- `composites/<tile>/<month>.npz`: each month of each tile, downloaded once (the 10 bands as
  compressed 16-bit integers, a NumPy file: faster to read than NetCDF);
- `landcover.nc`, `maps/`, `runs/`, `latest.json`: written by `valley_refresh`.

The refresh writes these files and the neighbor comparison (FR-08) reads them.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray
from shapely import union_all

from fundoscope.config import get_settings
from fundoscope.ingest import read_features, to_utm, utm_epsg
from fundoscope.sentinel import COMPOSITE_BANDS
from fundoscope.timeseries import ndvi
from fundoscope.valley import Valley, valley
from fundoscope.valley.season import MonthKey

AOI_FILE = "aoi.geojson"
GRID_FILE = "grid.json"
COMPOSITES_DIR = "composites"
MAPS_DIR = "maps"
RUNS_DIR = "runs"
LAND_COVER_FILE = "landcover.nc"
LATEST_FILE = "latest.json"
REFLECTANCE_STEP = np.float32(1e-4)  # month files hold reflectance x 10,000...
NO_VALUE = 65535  # ...as 16-bit integers, with this for "no clear pixel"
MAX_REFLECTANCE = (NO_VALUE - 1) * float(REFLECTANCE_STEP)


class ValleyError(RuntimeError):
    """The valley can't be refreshed. The message is meant for the user."""


def valley_dir(slug: str) -> Path:
    """data/valley/<slug>: git-ignored, like all project data."""
    return get_settings().data_dir / "valley" / slug


def load_valley(folder: Path, slug: str) -> Valley:
    """The valley whose area of interest is `folder/aoi.geojson`, on the grid of `grid.json`.

    Raises:
        ValleyError: there is no area of interest yet, or it goes beyond the saved grid.
        ParcelError: the file isn't a readable polygon in WGS84.
    """
    path = folder / AOI_FILE
    if not path.exists():
        raise ValleyError(f"No area of interest for {slug!r}: draw it and save it as {path}.")
    area = union_all([feature.geometry for feature in read_features(path.read_bytes(), path.name)])
    grid_path = folder / GRID_FILE
    saved = json.loads(grid_path.read_text(encoding="utf-8")) if grid_path.exists() else None
    epsg = saved["epsg"] if saved else utm_epsg(*area.centroid.coords[0])
    grid = {"bounds": tuple(saved["bounds"]), "max_tile_m": saved["max_tile_m"]} if saved else {}
    try:
        return valley(slug, to_utm(area, epsg), epsg, **grid)
    except ValueError as exc:
        raise ValleyError(
            f"The area of interest in {path} goes beyond the tiles already downloaded "
            f"({grid_path}). Keep it inside them, or move {folder / COMPOSITES_DIR} and "
            f"{grid_path} away to start over (a full download)."
        ) from exc


def save_grid(valley_: Valley, folder: Path) -> None:
    """Save the valley's tile grid, unless one is saved already (`load_valley` reads it)."""
    path = folder / GRID_FILE
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        grid = {
            "epsg": valley_.epsg,
            "bounds": list(valley_.bounds),
            "max_tile_m": valley_.max_tile_m,
        }
        path.write_text(json.dumps(grid, indent=2), encoding="utf-8")


def save_month(path: Path, layer: xr.DataArray) -> None:
    """Write one month of a tile, (band, y, x) reflectance, as compressed 16-bit integers."""
    values = layer.transpose("band", "y", "x").clip(0, MAX_REFLECTANCE).to_numpy()
    scaled = np.where(np.isnan(values), NO_VALUE, np.round(values / REFLECTANCE_STEP))
    arrays = {
        str(name): band
        for name, band in zip(layer["band"].to_numpy(), scaled.astype(np.uint16), strict=True)
    }
    partial = path.with_suffix(".tmp")
    with partial.open("wb") as file:
        np.savez_compressed(file, x=layer["x"].to_numpy(), y=layer["y"].to_numpy(), **arrays)
    partial.replace(path)  # only a complete file counts as saved


def load_month(path: Path, bands: tuple[str, ...] = COMPOSITE_BANDS) -> xr.DataArray:
    """One saved month: (band, y, x) reflectance, NaN where no clear pixel was seen."""
    with np.load(path) as saved:
        x, y = saved["x"], saved["y"]
    return xr.DataArray(
        load_bands(path, bands),
        dims=("band", "y", "x"),
        coords={"band": list(bands), "y": y, "x": x},
    )


def load_bands(path: Path, bands: tuple[str, ...]) -> NDArray[np.float32]:
    """The bands of one saved month as a plain (band, y, x) array, like `load_month`'s values.

    Faster when the coordinates aren't needed (e.g. the neighbor comparison, which reads many
    months on each report).
    """
    with np.load(path) as saved:
        values = np.stack([saved[name] for name in bands])
    return np.where(values == NO_VALUE, np.nan, values * REFLECTANCE_STEP).astype(np.float32)


def load_stack(tile_dir: Path, keys: list[MonthKey]) -> xr.DataArray:
    """Saved months of a tile as a (month, band, y, x) stack, for `pixel_features`."""
    months = pd.DatetimeIndex([key.month for key in keys], name="month")
    layers = [load_month(tile_dir / key.file_name) for key in keys]
    return xr.concat(layers, dim=months).transpose("month", "band", "y", "x")


def peak_ndvi(tile_dir: Path, keys: list[MonthKey]) -> NDArray[np.float64]:
    """Each pixel's highest monthly NDVI over the saved months `keys` (NaN if never clear)."""
    peak = None
    for key in keys:
        red, nir = load_bands(tile_dir / key.file_name, ("B04", "B08"))
        value = ndvi(red, nir)
        peak = value if peak is None else np.fmax(peak, value)
    if peak is None:
        raise ValueError("No months to take the peak NDVI from.")
    return peak
