"""`fundoscope labels check`: are the labeled parcels ready to train the crop model? (FR-06)"""

from pathlib import Path

from fundoscope.config import get_settings
from fundoscope.ingest import LabelError, LabelReport, check_labels, read_labels

MAX_IDS_SHOWN = 6


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


def default_labels_file() -> Path:
    """data/labels/labels.gpkg: git-ignored, because labels are private."""
    return get_settings().data_dir / "labels" / "labels.gpkg"


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
