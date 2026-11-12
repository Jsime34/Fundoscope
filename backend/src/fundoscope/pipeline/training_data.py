"""Training data for the crop model: composites -> pixel features per label (FR-06, FR-25).

For every labeled parcel and campaign (Street View and satellite labels; the field-verified
parcels are kept for the final test), download the campaign's monthly composites, keep the
pixels at least 15 m inside the parcel, compute their 150 features, and save up to 50 of them
in `<out>/pixels/<parcel>_<campaign>.csv`. A saved file means "done", so a second run continues
where the first stopped. Every request is logged with its credits in
`<out>/extraction_log.csv`, and no request starts once it could pass the budget's cap.

The files hold no crop: training joins them with the current labels, so correcting a label
doesn't need a new download.
"""

from collections import deque
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry

from fundoscope.ingest import TRAINING_SOURCES, ParcelFeature, validate_parcel
from fundoscope.ml import (
    FEATURE_COLUMNS,
    INTERIOR_BUFFER_M,
    MONTHS,
    FeatureError,
    pixel_features,
    training_sample,
)
from fundoscope.pipeline.parcel_history import CompositeFetcher
from fundoscope.sentinel import (
    COMPOSITE_BANDS,
    CreditBudget,
    SentinelError,
    fetch_monthly_composites,
)
from fundoscope.timeseries import campaign_bounds, campaign_of, parse_campaign

WORKERS = 2
MAX_FAILURES_IN_A_ROW = 3
PIXEL_AREA_M2 = 100.0  # one 10 m pixel
LOG_FILE = "extraction_log.csv"

DONE = "done"
NO_CLEAR_PIXELS = "no clear pixels"
TOO_NARROW = "too narrow"
FAILED = "failed"


@dataclass(frozen=True)
class Task:
    """One labeled parcel and campaign. `geometry_utm` is the whole parcel, in meters."""

    parcel_id: str
    campaign: str
    geometry_utm: BaseGeometry
    epsg: int

    @property
    def file_name(self) -> str:
        return f"{self.parcel_id}_{self.campaign.replace('/', '-')}.csv"


@dataclass(frozen=True)
class TaskResult:
    """What happened to one task.

    Attributes:
        status: `DONE`, `NO_CLEAR_PIXELS` (saved, but every pixel was too cloudy),
            `TOO_NARROW` (nothing left 15 m inside; not downloaded), or `FAILED`.
        pixels: rows saved.
        credits: openEO credits the request used.
        error: why it failed.
    """

    task: Task
    status: str
    pixels: int
    credits: float
    error: str = ""


@dataclass(frozen=True)
class Extraction:
    """What one run did.

    Attributes:
        total: parcel-campaigns in the plan.
        already_done: saved by earlier runs and skipped.
        results: one per task handled in this run, in the order they finished.
        stopped_at_cap: the budget stopped the run before the end.
        stopped_by_failures: 3 requests in a row failed (Copernicus down, or bad credentials).
    """

    total: int
    already_done: int
    results: list[TaskResult]
    stopped_at_cap: bool
    stopped_by_failures: bool

    @property
    def credits(self) -> float:
        return sum(result.credits for result in self.results)

    @property
    def remaining(self) -> int:
        """Tasks still to download: failed or not started (too narrow ones never are)."""
        handled = sum(result.status != FAILED for result in self.results)
        return self.total - self.already_done - handled

    def count(self, status: str) -> int:
        return sum(result.status == status for result in self.results)


def extraction_tasks(labels: gpd.GeoDataFrame) -> list[Task]:
    """One task per training label (WGS84 geometry) and campaign, in label order.

    Expects labels that passed `fundoscope.ingest.check_labels`.
    """
    tasks = []
    for row in labels[labels["source"].isin(TRAINING_SOURCES)].itertuples(index=False):
        parcel = validate_parcel([ParcelFeature(None, row.geometry)])
        for year in range(parse_campaign(row.from_campaign), parse_campaign(row.to_campaign) + 1):
            campaign = campaign_of(date(year, 8, 1))
            tasks.append(Task(row.parcel_id, campaign, parcel.geometry_utm, parcel.utm_epsg))
    return tasks


def pending(tasks: list[Task], out_dir: Path) -> list[Task]:
    """The tasks without a saved file in `out_dir/pixels`."""
    return [task for task in tasks if not (out_dir / "pixels" / task.file_name).exists()]


def load_training_pixels(tasks: list[Task], out_dir: Path) -> pd.DataFrame:
    """The saved pixels of `tasks`: `parcel_id`, `campaign`, `x`, `y` (UTM m), and the features.

    Tasks without a file (not downloaded yet, or too narrow) and empty files (no clear
    pixels) add no rows. Files of labels no longer in `tasks` are ignored.
    """
    tables = [
        pd.read_csv(path, dtype={"parcel_id": str, "campaign": str})
        for path in (out_dir / "pixels" / task.file_name for task in tasks)
        if path.exists()
    ]
    tables = [table for table in tables if not table.empty]
    if not tables:
        return pd.DataFrame(columns=["parcel_id", "campaign", "x", "y", *FEATURE_COLUMNS])
    return pd.concat(tables, ignore_index=True)


def extract_training_data(
    tasks: list[Task],
    out_dir: Path,
    budget: CreditBudget,
    *,
    fetch: CompositeFetcher = fetch_monthly_composites,
    on_result: Callable[[TaskResult], None] | None = None,
    workers: int = WORKERS,
) -> Extraction:
    """Download and save the training pixels of every pending task, within `budget`.

    Runs `workers` requests at a time. Failed tasks are logged and retried by the next run.
    """
    (out_dir / "pixels").mkdir(parents=True, exist_ok=True)
    todo = deque(pending(tasks, out_dir))
    already_done = len(tasks) - len(todo)
    results: list[TaskResult] = []
    stopped_at_cap = stopped_by_failures = False
    failures_in_a_row = 0

    def record(result: TaskResult) -> None:
        results.append(result)
        _log(out_dir, result)
        if on_result:
            on_result(result)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        running: dict[Future[TaskResult], float] = {}
        while todo or running:
            while todo and len(running) < workers and not (stopped_at_cap or stopped_by_failures):
                if too_narrow(todo[0]):
                    record(TaskResult(todo.popleft(), TOO_NARROW, 0, 0.0))
                    continue
                hold = budget.reserve()
                if hold is None:
                    stopped_at_cap = True
                    break
                running[pool.submit(_extract, todo.popleft(), fetch, out_dir)] = hold
            if not running:
                break
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                result = future.result()
                budget.settle(running.pop(future), result.credits)
                failures_in_a_row = failures_in_a_row + 1 if result.status == FAILED else 0
                stopped_by_failures = stopped_by_failures or (
                    failures_in_a_row >= MAX_FAILURES_IN_A_ROW
                )
                record(result)

    return Extraction(len(tasks), already_done, results, stopped_at_cap, stopped_by_failures)


def too_narrow(task: Task) -> bool:
    """True when no pixel lies 15 m inside the parcel: it is never downloaded."""
    return task.geometry_utm.buffer(-INTERIOR_BUFFER_M).area < PIXEL_AREA_M2


def _extract(task: Task, fetch: CompositeFetcher, out_dir: Path) -> TaskResult:
    start, end = campaign_bounds(task.campaign)
    try:
        composites = fetch(task.geometry_utm, task.epsg, start, end, bands=COMPOSITE_BANDS)
    except SentinelError as exc:
        return TaskResult(task, FAILED, 0, 0.0, str(exc))
    # A month without any image may be missing from the result; give it an empty slot.
    months = pd.date_range(start, periods=len(MONTHS), freq="MS")
    interior = task.geometry_utm.buffer(-INTERIOR_BUFFER_M)
    stack = composites.masked_to(interior).data.reindex(month=months)
    try:
        features = pixel_features(stack)
    except FeatureError as exc:
        return TaskResult(task, FAILED, 0, composites.credits, str(exc))
    sample = training_sample(features, f"{task.parcel_id} {task.campaign}")
    _save(sample, task, out_dir / "pixels" / task.file_name)
    status = DONE if len(sample) else NO_CLEAR_PIXELS
    return TaskResult(task, status, len(sample), composites.credits)


def _save(sample: pd.DataFrame, task: Task, path: Path) -> None:
    table = sample.reset_index()
    table.insert(0, "campaign", task.campaign)
    table.insert(0, "parcel_id", task.parcel_id)
    partial = path.with_suffix(".tmp")
    table.to_csv(partial, index=False)
    partial.replace(path)  # only a complete file counts as done


def _log(out_dir: Path, result: TaskResult) -> None:
    path = out_dir / LOG_FILE
    row = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "parcel_id": result.task.parcel_id,
        "campaign": result.task.campaign,
        "status": result.status,
        "pixels": result.pixels,
        "credits": result.credits,
        "error": result.error,
    }
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
