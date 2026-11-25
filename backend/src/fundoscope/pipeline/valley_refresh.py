"""Refresh the valley crop count (FR-14): download what is new, classify, add up the hectares.

Files, in data/valley/<slug>/ (git-ignored):

- `aoi.geojson`: the area of interest, a polygon in WGS84 (drawn in QGIS);
- `grid.json`: the tile grid, saved by the first refresh. The months on disk belong to it, so
  later refreshes keep it: the area of interest may change, but only inside the grid;
- `composites/<tile>/<month>.npz`: each month of each tile, downloaded once (the 10 bands as
  compressed 16-bit integers, a NumPy file: faster to read than NetCDF);
- `landcover.nc`: ESA WorldCover 2021 on the valley grid;
- `maps/<count>.nc`: the class of every pixel of each count (opens in QGIS);
- `maps/<count>/<class>.png` and `maps/<count>/farmland.png`: the images of the dashboard;
- `runs/<time>.json` and `latest.json`: the hectares of each count;
- `requests.csv`: every openEO request with its cost (FR-25).

A refresh first downloads the months it lacks (see `fundoscope.valley.season`), two requests
at a time, within a credit budget. A run stopped by the budget or by failures counts nothing
and resumes where it stopped. When every month is on disk, it computes the farmland of each
tile and classifies the tile for every count, then adds up the hectares with their 95 %
ranges. The first run downloads three campaigns (about 55 requests, two hours); a weekly
refresh downloads the month in progress and the same days one year earlier (about 18
requests, 15 minutes).
"""

import json
import time
from collections import Counter, deque
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray
from pyproj import CRS
from shapely import union_all

from fundoscope.config import get_settings
from fundoscope.ingest import from_utm, read_features, to_utm, utm_epsg
from fundoscope.ml import MONTHS, CropModel
from fundoscope.pipeline.train_model import (
    MODEL_FILE,
    SEASON_DIR,
    load_crop_model,
    load_model_card,
    model_confusion,
    season_model_file,
)
from fundoscope.pipeline.training_data import MAX_FAILURES_IN_A_ROW, WORKERS
from fundoscope.sentinel import COMPOSITE_BANDS, CreditBudget, SentinelError
from fundoscope.sentinel.catalog import newest_image_date
from fundoscope.sentinel.composites import fetch_monthly_composites
from fundoscope.sentinel.landcover import NO_DATA_CLASS, LandCover, fetch_land_cover
from fundoscope.timeseries import ndvi
from fundoscope.valley import (
    PIXEL_HA,
    PIXEL_M,
    ClassArea,
    Tile,
    Valley,
    class_areas,
    classify_tile,
    farmland_mask,
    map_classes,
    valley,
)
from fundoscope.valley.classify import PLANTED
from fundoscope.valley.render import (
    FARMLAND,
    class_color,
    class_png,
    corners,
    farmland_png,
    image_name,
)
from fundoscope.valley.season import Count, MonthKey, Season, group_months, season

AOI_FILE = "aoi.geojson"
GRID_FILE = "grid.json"
COMPOSITES_DIR = "composites"
MAPS_DIR = "maps"
RUNS_DIR = "runs"
LAND_COVER_FILE = "landcover.nc"
LATEST_FILE = "latest.json"
LOG_FILE = "requests.csv"
LAND_COVER = "land cover"
REFLECTANCE_STEP = np.float32(1e-4)  # month files hold reflectance x 10,000...
NO_VALUE = 65535  # ...as 16-bit integers, with this for "no clear pixel"
MAX_REFLECTANCE = (NO_VALUE - 1) * float(REFLECTANCE_STEP)
_CLASSES = {"dtype": "uint8", "zlib": True, "complevel": 4}

CompositeFetcher = Callable[..., Any]  # fetch_monthly_composites
LandCoverFetcher = Callable[..., LandCover]


class ValleyError(RuntimeError):
    """The valley can't be refreshed. The message is meant for the user."""


@dataclass(frozen=True)
class Request:
    """Consecutive months of one tile, downloaded in one openEO request."""

    tile: Tile
    months: tuple[MonthKey, ...]

    @property
    def start(self) -> date:
        return self.months[0].month

    @property
    def end(self) -> date:
        return self.months[-1].through

    @property
    def name(self) -> str:
        """ "r0c1 2024-08 to 2025-01", or "r0c1 2025-11-01 to 2025-11-22" for a cut month."""
        last = self.months[-1]
        end = f"{last.month:%Y-%m}" if last.complete else f"{last.through:%Y-%m-%d}"
        start = f"{self.start:%Y-%m}" if last.complete else f"{self.start:%Y-%m-%d}"
        return f"{self.tile.id} {start} to {end}"


@dataclass(frozen=True)
class Plan:
    """What a refresh will download and count.

    Attributes:
        valley: the valley and its tiles.
        season: the date of the newest images and the counts.
        requests: composites still to download.
        needs_land_cover: WorldCover isn't on disk yet (one more request).
    """

    valley: Valley
    season: Season
    requests: tuple[Request, ...]
    needs_land_cover: bool

    @property
    def downloads(self) -> int:
        return len(self.requests) + self.needs_land_cover


@dataclass(frozen=True)
class Download:
    """One finished openEO request (`error` is empty when it worked)."""

    name: str
    credits: float
    seconds: float
    error: str = ""


@dataclass(frozen=True)
class CountResult:
    """The hectares of one count.

    Attributes:
        count: what was counted.
        model: the model file used, relative to the models folder.
        areas: the area of every class, in map order.
        map_file: the class map, relative to the valley folder.
        images: the dashboard's image of each class and of the farmland ("farmland"),
            relative to the valley folder.
    """

    count: Count
    model: str
    areas: list[ClassArea]
    map_file: str
    images: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Refresh:
    """What a refresh did.

    Attributes:
        plan: what it set out to do.
        downloads: the requests it made, in the order they finished.
        stopped_at_cap: the budget stopped the downloads before the end.
        stopped_by_failures: 3 requests in a row failed.
        counts: the hectares of each count (empty when downloads are missing).
        farmland_ha: hectares of farmland in the valley.
        newest_image: the day of the newest image the first count could use (None if the
            catalog couldn't say).
        run_file: the saved run (None when nothing was counted).
        seconds: how long the refresh took.
    """

    plan: Plan
    downloads: list[Download]
    stopped_at_cap: bool = False
    stopped_by_failures: bool = False
    counts: list[CountResult] = field(default_factory=list)
    farmland_ha: float = 0.0
    newest_image: date | None = None
    run_file: Path | None = None
    seconds: float = 0.0

    @property
    def credits(self) -> float:
        return sum(item.credits for item in self.downloads)

    @property
    def failed(self) -> list[Download]:
        return [item for item in self.downloads if item.error]

    @property
    def missing(self) -> int:
        """Downloads still to do: failed or never started."""
        return self.plan.downloads - (len(self.downloads) - len(self.failed))


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


def plan_refresh(valley_: Valley, folder: Path, today: date) -> Plan:
    """The requests a refresh on `today` needs: the months of each tile not on disk yet."""
    when = season(today)
    keys = when.month_keys()
    requests = []
    for tile in valley_.tiles:
        tile_dir = folder / COMPOSITES_DIR / tile.id
        missing = [key for key in keys if not (tile_dir / key.file_name).exists()]
        requests += [Request(tile, tuple(group)) for group in group_months(missing)]
    return Plan(valley_, when, tuple(requests), not (folder / LAND_COVER_FILE).exists())


def refresh_valley(
    plan: Plan,
    folder: Path,
    models_dir: Path,
    budget: CreditBudget,
    *,
    fetch: CompositeFetcher = fetch_monthly_composites,
    fetch_cover: LandCoverFetcher = fetch_land_cover,
    newest_image: Callable[..., date | None] = newest_image_date,
    on_download: Callable[[Download], None] | None = None,
    on_tile: Callable[[Tile], None] | None = None,
    workers: int = WORKERS,
    now: Callable[[], datetime] = datetime.now,
) -> Refresh:
    """Download what `plan` lacks within `budget`, then count every crop of the valley.

    Raises:
        ValleyError: there is no crop model or model card (the counts need them).
    """
    started = time.monotonic()
    models, card = _models(plan.season.counts, models_dir)
    save_grid(plan.valley, folder)
    downloads, at_cap, by_failures = _download(
        plan, folder, budget, fetch, fetch_cover, on_download, workers
    )
    refresh = Refresh(plan, downloads, at_cap, by_failures)
    if refresh.missing:
        return refresh

    _prune(plan, folder)
    counts, farmland = _count(plan, folder, models, on_tile)
    results = []
    for count in plan.season.counts:
        areas = class_areas(counts[count.key], model_confusion(card, count.months))
        images = _image_files(count, map_classes(models[count.months]))
        results.append(
            CountResult(count, _model_file(count.months), areas, _map_file(count), images)
        )
    refresh = Refresh(
        plan,
        downloads,
        counts=results,
        farmland_ha=farmland * PIXEL_HA,
        newest_image=_newest_image(plan, newest_image),
        seconds=time.monotonic() - started,
    )
    run_file = _save_run(refresh, folder, card, now())
    return replace(refresh, run_file=run_file)


def _newest_image(plan: Plan, newest_image: Callable[..., date | None]) -> date | None:
    """The newest Sentinel-2 image over the valley up to the end of the first count."""
    bounds = from_utm(plan.valley.area_utm, plan.valley.epsg).bounds
    try:
        return newest_image(bounds, plan.season.counts[0].through)
    except SentinelError:
        return None  # only a date on the dashboard: not worth failing the count


# --- downloads ------------------------------------------------------------------------------


def _download(
    plan: Plan,
    folder: Path,
    budget: CreditBudget,
    fetch: CompositeFetcher,
    fetch_cover: LandCoverFetcher,
    on_download: Callable[[Download], None] | None,
    workers: int,
) -> tuple[list[Download], bool, bool]:
    """Run the plan's requests, `workers` at a time, until done, at the cap, or failing."""
    jobs: deque[tuple[str, Callable[[], float]]] = deque()
    if plan.needs_land_cover:
        jobs.append((LAND_COVER, lambda: _get_land_cover(plan.valley, folder, fetch_cover)))
    for request in plan.requests:
        jobs.append((request.name, _composites_job(request, plan.valley.epsg, folder, fetch)))

    downloads: list[Download] = []
    at_cap = by_failures = False
    failures_in_a_row = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        running: dict[Future[Download], float] = {}
        while jobs or running:
            while jobs and len(running) < workers and not (at_cap or by_failures):
                hold = budget.reserve()
                if hold is None:
                    at_cap = True
                    break
                running[pool.submit(_run, *jobs.popleft())] = hold
            if not running:
                break
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                download = future.result()
                budget.settle(running.pop(future), download.credits)
                failures_in_a_row = failures_in_a_row + 1 if download.error else 0
                by_failures = by_failures or failures_in_a_row >= MAX_FAILURES_IN_A_ROW
                downloads.append(download)
                _log(folder, download)
                if on_download:
                    on_download(download)
    return downloads, at_cap, by_failures


class _PaidError(SentinelError):
    """openEO answered (and charged for) a request whose result can't be used."""

    def __init__(self, message: str, credits: float):
        super().__init__(message)
        self.credits = credits


def _run(name: str, job: Callable[[], float]) -> Download:
    started = time.monotonic()
    try:
        credits = job()
    except SentinelError as exc:
        spent = exc.credits if isinstance(exc, _PaidError) else 0.0
        return Download(name, spent, time.monotonic() - started, str(exc))
    return Download(name, credits, time.monotonic() - started)


def _composites_job(
    request: Request, epsg: int, folder: Path, fetch: CompositeFetcher
) -> Callable[[], float]:
    def job() -> float:
        result = fetch(
            request.tile.geometry, epsg, request.start, request.end, bands=COMPOSITE_BANDS
        )
        months = pd.DatetimeIndex([key.month for key in request.months])
        try:
            data = on_grid(result.data, request.tile).reindex(month=months)
        except SentinelError as exc:
            raise _PaidError(str(exc), float(result.credits)) from exc
        tile_dir = folder / COMPOSITES_DIR / request.tile.id
        tile_dir.mkdir(parents=True, exist_ok=True)
        for key, layer in zip(request.months, data.transpose("month", ...), strict=True):
            save_month(tile_dir / key.file_name, layer)
        return float(result.credits)

    return job


def _get_land_cover(valley_: Valley, folder: Path, fetch_cover: LandCoverFetcher) -> float:
    grid = Tile("valley", valley_.bounds)
    cover = fetch_cover(grid.geometry, valley_.epsg)
    try:
        data = on_grid(cover.data, grid, NO_DATA_CLASS, aligned=False)
    except SentinelError as exc:
        raise _PaidError(str(exc), float(cover.credits)) from exc
    data = data.astype(np.uint8).rename("land_cover")
    folder.mkdir(parents=True, exist_ok=True)
    _write(
        _georeferenced(data.to_dataset(), valley_.epsg),
        folder / LAND_COVER_FILE,
        {"land_cover": _CLASSES},
    )
    return float(cover.credits)


def on_grid(
    data: xr.DataArray, tile: Tile, fill_value: float = np.nan, *, aligned: bool = True
) -> xr.DataArray:
    """`data` on the tile's pixel grid (openEO adds one pixel on each side).

    Composites come on the tile's 10 m grid, and `aligned` checks it. WorldCover comes on a
    10 m grid shifted by a few meters (it is stored in latitude and longitude): with
    `aligned=False`, each pixel takes the value of the nearest one.

    Raises:
        SentinelError: there are no pixels, or `aligned` and they are off the tile's grid.
    """
    x, y = data["x"].to_numpy(), data["y"].to_numpy()
    if not len(x) or not len(y):
        raise SentinelError(f"openEO returned no pixels for tile {tile.id}.")
    if aligned and (_off_grid(x[0], tile.x[0]) or _off_grid(y[0], tile.y[0])):
        raise SentinelError(f"openEO returned pixels off the 10 m grid of tile {tile.id}.")
    tolerance = PIXEL_M / 4 if aligned else PIXEL_M / 2
    return data.reindex(
        x=tile.x, y=tile.y, method="nearest", tolerance=tolerance, fill_value=fill_value
    )


def _off_grid(a: float, b: float) -> bool:
    shift = (a - b) % PIXEL_M
    return min(shift, PIXEL_M - shift) > PIXEL_M / 4


def _log(folder: Path, download: Download) -> None:
    row = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "request": download.name,
        "credits": download.credits,
        "seconds": round(download.seconds, 1),
        "error": download.error,
    }
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / LOG_FILE
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


# --- month files ----------------------------------------------------------------------------


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
        values = np.stack([saved[name] for name in bands])
        x, y = saved["x"], saved["y"]
    reflectance = np.where(values == NO_VALUE, np.nan, values * REFLECTANCE_STEP)
    return xr.DataArray(
        reflectance.astype(np.float32),
        dims=("band", "y", "x"),
        coords={"band": list(bands), "y": y, "x": x},
    )


def load_stack(tile_dir: Path, keys: list[MonthKey]) -> xr.DataArray:
    """Saved months of a tile as a (month, band, y, x) stack, for `pixel_features`."""
    months = pd.DatetimeIndex([key.month for key in keys], name="month")
    layers = [load_month(tile_dir / key.file_name) for key in keys]
    return xr.concat(layers, dim=months).transpose("month", "band", "y", "x")


def peak_ndvi(tile_dir: Path, keys: list[MonthKey]) -> NDArray[np.float64]:
    """Each pixel's highest monthly NDVI over the saved months `keys` (NaN if never clear)."""
    peak = None
    for key in keys:
        red, nir = load_month(tile_dir / key.file_name, ("B04", "B08")).to_numpy()
        value = ndvi(red, nir)
        peak = value if peak is None else np.fmax(peak, value)
    if peak is None:
        raise ValueError("No months to take the peak NDVI from.")
    return peak


def _check_grid(tile_dir: Path, keys: list[MonthKey], tile: Tile) -> None:
    """Every month file of the tile must be on the tile's pixel grid."""
    for key in keys:
        with np.load(tile_dir / key.file_name) as saved:
            x, y = saved["x"], saved["y"]
        if not (np.array_equal(x, tile.x) and np.array_equal(y, tile.y)):
            raise ValleyError(
                f"{tile_dir / key.file_name} is on another tile grid than {tile.id}: the tiles "
                f"changed since it was downloaded. Restore {GRID_FILE}, or move "
                f"{tile_dir.parent} away to start over (a full download)."
            )


def _prune(plan: Plan, folder: Path) -> None:
    """Delete the months cut short that no count needs any more (a newer cut replaced them)."""
    needed = {key.file_name for key in plan.season.month_keys()}
    for tile in plan.valley.tiles:
        for path in (folder / COMPOSITES_DIR / tile.id).glob("????-??-??.npz"):
            if path.name not in needed:
                path.unlink()


def _georeferenced(dataset: xr.Dataset, epsg: int) -> xr.Dataset:
    """Add the CRS the way GDAL (and so QGIS) reads it from NetCDF."""
    crs = CRS.from_epsg(epsg).to_wkt()
    dataset = dataset.assign(
        spatial_ref=xr.DataArray(0, attrs={"crs_wkt": crs, "spatial_ref": crs})
    )
    for name in dataset.data_vars:
        if name != "spatial_ref":
            dataset[name].attrs["grid_mapping"] = "spatial_ref"
    dataset["x"].attrs.update(units="m", standard_name="projection_x_coordinate")
    dataset["y"].attrs.update(units="m", standard_name="projection_y_coordinate")
    return dataset


def _write(dataset: xr.Dataset, path: Path, encoding: dict[str, dict[str, Any]]) -> None:
    """Write through a temporary file, so only a complete file counts as saved."""
    partial = path.with_suffix(".tmp")
    dataset.to_netcdf(partial, engine="h5netcdf", encoding=encoding)
    partial.replace(path)


# --- counts ---------------------------------------------------------------------------------


def _models(
    counts: tuple[Count, ...], models_dir: Path
) -> tuple[dict[int, CropModel], dict[str, Any]]:
    """The model of each count's length, and the model card (for the confusion matrices)."""
    card = load_model_card(models_dir)
    models = {}
    for months in {count.months for count in counts}:
        model = load_crop_model(models_dir, months)
        if model is None or card is None:
            raise ValleyError(
                f"The crop models aren't in {models_dir}: run `fundoscope model train` first."
            )
        models[months] = model
    return models, card  # type: ignore[return-value]


def _model_file(months: int) -> str:
    if months == len(MONTHS):
        return MODEL_FILE
    return f"{SEASON_DIR}/{season_model_file(months)}"


def _map_file(count: Count) -> str:
    return f"{MAPS_DIR}/{count.key}.nc"


def _count(
    plan: Plan,
    folder: Path,
    models: dict[int, CropModel],
    on_tile: Callable[[Tile], None] | None,
) -> tuple[dict[str, dict[str, int]], int]:
    """Classify every tile for every count; save the class maps; pixels per class."""
    valley_ = plan.valley
    west, _, _, north = valley_.bounds
    grid = Tile("valley", valley_.bounds)
    with xr.open_dataset(folder / LAND_COVER_FILE, engine="h5netcdf") as dataset:
        land_cover = dataset["land_cover"].load()
    counts = {count.key: Counter[str]() for count in plan.season.counts}
    maps = {
        count.key: np.zeros((len(grid.y), len(grid.x)), np.uint8) for count in plan.season.counts
    }
    farmland_pixels = 0
    keys = plan.season.month_keys()
    for tile in valley_.tiles:
        tile_dir = folder / COMPOSITES_DIR / tile.id
        _check_grid(tile_dir, keys, tile)
        cover = land_cover.reindex(x=tile.x, y=tile.y, fill_value=NO_DATA_CLASS).to_numpy()
        farmland = farmland_mask(peak_ndvi(tile_dir, keys), cover, valley_.inside(tile))
        farmland_pixels += int(farmland.sum())
        row = round((north - tile.bounds[3]) / PIXEL_M)
        col = round((tile.bounds[0] - west) / PIXEL_M)
        for count in plan.season.counts:
            stack = load_stack(tile_dir, count.month_keys())
            classes = classify_tile(
                stack, farmland, models[count.months], in_progress=count.in_progress
            )
            counts[count.key].update(classes.counts())
            maps[count.key][row : row + len(tile.y), col : col + len(tile.x)] = classes.codes
        if on_tile:
            on_tile(tile)
    (folder / MAPS_DIR).mkdir(exist_ok=True)
    for count in plan.season.counts:
        classes = map_classes(models[count.months])
        _save_map(folder / _map_file(count), maps[count.key], grid, classes, valley_.epsg)
        _save_images(folder, count, maps[count.key], classes)
    ordered = {
        count.key: {name: counts[count.key][name] for name in map_classes(models[count.months])}
        for count in plan.season.counts
    }
    return ordered, farmland_pixels


def _save_map(
    path: Path, codes: NDArray[np.uint8], grid: Tile, classes: tuple[str, ...], epsg: int
) -> None:
    """The class of every pixel: 0 outside the farmland, else 1 + the index in `classes`."""
    data = xr.DataArray(
        codes,
        coords={"y": grid.y, "x": grid.x},
        dims=("y", "x"),
        name="crop_class",
        attrs={"classes": json.dumps(classes), "codes": "0 = not farmland, n = classes[n - 1]"},
    )
    _write(_georeferenced(data.to_dataset(), epsg), path, {"crop_class": _CLASSES})


def _image_files(count: Count, classes: tuple[str, ...]) -> dict[str, str]:
    folder = f"{MAPS_DIR}/{count.key}"
    images = {name: f"{folder}/{image_name(name)}.png" for name in classes}
    return {FARMLAND: f"{folder}/{FARMLAND}.png", **images}


def _save_images(
    folder: Path, count: Count, codes: NDArray[np.uint8], classes: tuple[str, ...]
) -> None:
    """The dashboard's images: the farmland, and each class in its color (see `render`)."""
    files = _image_files(count, classes)
    (folder / MAPS_DIR / count.key).mkdir(exist_ok=True)
    (folder / files[FARMLAND]).write_bytes(farmland_png(codes))
    for code, name in enumerate(classes, start=1):
        color = class_color(name, classes)
        image = class_png(codes, code, color, striped=name == PLANTED)
        (folder / files[name]).write_bytes(image)


def _save_run(refresh: Refresh, folder: Path, card: dict[str, Any], run_at: datetime) -> Path:
    """Save the run as runs/<time>.json and latest.json."""
    plan = refresh.plan
    run = {
        "valley": plan.valley.slug,
        "run_at": run_at.isoformat(timespec="seconds"),
        "as_of": plan.season.as_of.isoformat(),
        "newest_image": None if refresh.newest_image is None else refresh.newest_image.isoformat(),
        "area_km2": round(plan.valley.area_km2, 1),
        "farmland_ha": round(refresh.farmland_ha, 2),
        "credits": refresh.credits,
        "downloads": len(refresh.downloads),
        "seconds": round(refresh.seconds, 1),
        "model": {"version": card["version"], "trained_at": card["trained_at"]},
        "corners": corners(plan.valley.bounds, plan.valley.epsg),
        "colors": {
            area.name: class_color(area.name, tuple(a.name for a in refresh.counts[0].areas))
            for area in refresh.counts[0].areas
        },
        "counts": [_count_json(result) for result in refresh.counts],
    }
    text = json.dumps(run, indent=2)
    (folder / RUNS_DIR).mkdir(exist_ok=True)
    path = folder / RUNS_DIR / f"{run_at:%Y%m%d-%H%M%S}.json"
    path.write_text(text, encoding="utf-8")
    (folder / LATEST_FILE).write_text(text, encoding="utf-8")
    return path


def _count_json(result: CountResult) -> dict[str, Any]:
    count = result.count
    return {
        "key": count.key,
        "campaign": count.campaign,
        "months": count.months,
        "through": count.through.isoformat(),
        "in_progress": count.in_progress,
        "model": result.model,
        "map": result.map_file,
        "images": result.images,
        "classes": [
            {
                "name": area.name,
                "pixels": area.pixels,
                "hectares": round(area.hectares, 2),
                "adjusted": None if area.adjusted is None else round(area.adjusted, 2),
                "low": None if area.low is None else round(area.low, 2),
                "high": None if area.high is None else round(area.high, 2),
            }
            for area in result.areas
        ],
    }
