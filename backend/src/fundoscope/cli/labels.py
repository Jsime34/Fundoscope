"""`fundoscope labels check` and `labels extract`: the labeled parcels of the crop model (FR-06).

`check` lists what still has to be fixed; `extract` downloads the training pixels of every
label within the monthly free quota (FR-25).
"""

from collections.abc import Callable
from pathlib import Path

from fundoscope.config import get_settings
from fundoscope.ingest import LabelError, LabelReport, check_labels, read_labels
from fundoscope.pipeline import (
    DONE,
    FAILED,
    LOG_FILE,
    NO_CLEAR_PIXELS,
    TOO_NARROW,
    Extraction,
    TaskResult,
    extract_training_data,
    extraction_tasks,
    pending,
)
from fundoscope.pipeline.parcel_history import CompositeFetcher
from fundoscope.sentinel import CreditBudget, batch_cap

MAX_IDS_SHOWN = 6
TYPICAL_REQUEST_CREDITS = 7.0  # measured: one campaign of the 3 ha demo parcel


def run_labels_check(file: Path | None) -> int:
    """Check the labels file and print what still has to be fixed. Returns the exit code."""
    path = file or default_labels_file()
    try:
        report = check_labels(read_labels(path))
    except LabelError as exc:
        print(str(exc))
        return 1
    print(format_report(report))
    return 0 if report.ok else 1


def run_labels_extract(
    file: Path | None,
    used_credits: float | None,
    cap: float | None,
    *,
    dry_run: bool = False,
    out: Path | None = None,
    fetch: CompositeFetcher | None = None,
) -> int:
    """Download the training pixels of every label within the quota. Returns the exit code.

    The run may spend at most 80 % of the month's free openEO credits minus `used_credits`,
    and at most `cap` credits; at least one of the two is required.
    """
    path = file or default_labels_file()
    try:
        labels = read_labels(path)
    except LabelError as exc:
        print(str(exc))
        return 1
    if not check_labels(labels).ok:
        print("The labels aren't ready: run `fundoscope labels check` and fix what it lists.")
        return 1

    out = out or default_features_dir()
    tasks = extraction_tasks(labels)
    todo = pending(tasks, out)
    estimate = len(todo) * TYPICAL_REQUEST_CREDITS
    print(
        f"{len(tasks)} parcel-campaigns: {len(tasks) - len(todo)} already saved, "
        f"{len(todo)} to download (about {estimate:,.0f} openEO credits)."
    )
    if dry_run or not todo:
        return 0
    limits = [cap, None if used_credits is None else batch_cap(used_credits)]
    if limits == [None, None]:
        print(
            "Pass --used-credits N, the openEO credits already used this month (see the "
            "Copernicus Data Space dashboard), or --cap N."
        )
        return 1
    limit = min(value for value in limits if value is not None)
    print(f"This run may spend up to {limit:,.0f} credits.")

    options = {"fetch": fetch} if fetch else {}
    extraction = extract_training_data(
        tasks, out, CreditBudget(limit), on_result=_progress(len(todo)), **options
    )
    print(format_extraction(extraction, out))
    return 0 if extraction.remaining == 0 else 1


def default_labels_file() -> Path:
    """data/labels/labels.gpkg: git-ignored, because labels are private."""
    return get_settings().data_dir / "labels" / "labels.gpkg"


def default_features_dir() -> Path:
    """data/features: git-ignored, like all project data."""
    return get_settings().data_dir / "features"


def format_extraction(extraction: Extraction, out: Path) -> str:
    lines = [
        f"\nSaved {extraction.count(DONE)} parcel-campaigns in this run "
        f"({extraction.credits:,.1f} openEO credits)."
    ]
    if extraction.count(NO_CLEAR_PIXELS):
        lines.append(
            f"{extraction.count(NO_CLEAR_PIXELS)} had no clear pixels; their files are empty."
        )
    if extraction.count(TOO_NARROW):
        lines.append(
            f"{extraction.count(TOO_NARROW)} are too narrow: no pixel is 15 m inside the parcel."
        )
    if extraction.count(FAILED):
        lines.append(
            f"{extraction.count(FAILED)} failed; the next run retries them "
            f"(details in {out / LOG_FILE})."
        )
    if extraction.stopped_by_failures:
        lines.append(
            "Stopped: 3 requests in a row failed. Check the connection with "
            "`fundoscope check-cdse`."
        )
    elif extraction.stopped_at_cap:
        lines.append(
            "Stopped at the credit cap. Run the command again to continue, with the new "
            "--used-credits from the dashboard."
        )
    if extraction.remaining:
        lines.append(f"{extraction.remaining} parcel-campaigns are still to download.")
    else:
        lines.append(f"All training pixels are saved in {out / 'pixels'}.")
    return "\n".join(lines)


def _progress(total: int) -> Callable[[TaskResult], None]:
    finished = 0

    def show(result: TaskResult) -> None:
        nonlocal finished
        finished += 1
        detail = f"{result.pixels} pixels" if result.status == DONE else result.status
        if result.error:
            detail += f" ({result.error})"
        print(
            f"  [{finished}/{total}] {result.task.parcel_id} {result.task.campaign}: "
            f"{detail}, {result.credits:.1f} credits"
        )

    return show


def format_report(report: LabelReport) -> str:
    regions = ", ".join(f"{code} {count}" for code, count in report.regions.items())
    lines = [
        f"{report.total} labels: {report.training_parcels} for training, "
        f"{report.field_verified} field-verified (kept for the final test)",
        f"Training parcels per region: {regions or 'none'}",
        "",
        f"{'Model class':<20}{'Parcels':>8}  Status",
    ]
    for item in report.classes:
        crops = ", ".join(f"{crop} {count}" for crop, count in item.crops.items())
        detail = f"  ({crops})" if len(item.crops) > 1 else ""
        lines.append(f"{item.name:<20}{item.parcels:>8}  {item.status}{detail}")
    if report.problems:
        lines += ["", "Fix these labels:"]
        for message, parcel_ids in report.problems.items():
            lines.append(f"  [{len(parcel_ids)}] {message}: {_shorten(parcel_ids)}")
    if report.unmet:
        lines += ["", "Not met yet (FR-06):"]
        lines += [f"  - {item}" for item in report.unmet]
    lines += ["", "Ready for the training-data step." if report.ok else "Not ready yet."]
    return "\n".join(lines)


def _shorten(parcel_ids: list[str]) -> str:
    shown = ", ".join(parcel_ids[:MAX_IDS_SHOWN])
    extra = len(parcel_ids) - MAX_IDS_SHOWN
    return f"{shown} and {extra} more" if extra > 0 else shown
