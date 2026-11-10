"""Check the labeled training parcels before any satellite data is spent on them (FR-06).

Reads the GeoPackage drawn in QGIS (layer `labels`, WGS84 lon/lat) and checks each label and
the whole set against FR-06. Comparing each label with its NDVI curve needs the composites,
so that check belongs to the training-data step, not here.
"""

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely import STRtree
from shapely.geometry.base import BaseGeometry

from fundoscope.ingest.parcels import validate_parcel
from fundoscope.ingest.readers import ParcelError, ParcelFeature

LAYER = "labels"
COLUMNS = (
    "parcel_id",
    "region",
    "crop",
    "from_campaign",
    "to_campaign",
    "source",
    "confidence",
    "evidence",
)

MIN_TRAINING_PARCELS = 150
MIN_REGIONS = 3
REQUIRED_REGION = "ARE"  # Arequipa: FR-06 requires labels from Majes.
MIN_PARCELS_PER_CLASS = 15

OTHER_ANNUAL = "other annual crops"
FALLOW = "fallow/bare"
MODEL_CLASSES = (
    "grape",
    "asparagus",
    "avocado",
    "blueberry",
    "citrus",
    "mango",
    "sugarcane",
    "rice",
    "alfalfa",
    OTHER_ANNUAL,
    FALLOW,
)

# Values of the QGIS `crop` drop-down -> the model class each one trains.
CROP_CLASSES = {
    **{crop: crop for crop in MODEL_CLASSES if crop not in (OTHER_ANNUAL, FALLOW)},
    "mandarine": "citrus",
    "bare": FALLOW,
    "fallow": FALLOW,
    **dict.fromkeys(
        (
            "artichoke",
            "corn",
            "garlic",
            "onion",
            "other",
            "paprika",
            "potato",
            "quinoa",
            "strawberry",
            "tomato",
        ),
        OTHER_ANNUAL,
    ),
}

SOURCES = ("photo", "field")
CONFIDENCES = ("high", "medium")
# young = planted or growing; mature = full or harvest; bare = no crop.
STAGES = ("young", "mature", "bare")

# The composites start in August 2018, the first month of campaign 2018/19.
FIRST_CAMPAIGN_YEAR = 2018
FIRST_PHOTO_MONTH = "2018-08"

# Two labels sharing more than this of the smaller one are the same field drawn twice, which
# would put one field in both the training and the test folds.
MAX_OVERLAP_SHARE = 0.1

EVIDENCE_FORMAT = "SV YYYY-MM <stage> <link>"
_PHOTO_EVIDENCE = re.compile(r"SV (\d{4}-(\d{2})) (\w+) https://\S+")
_PARCEL_ID = re.compile(r"[A-Z]{3}-\d{4}")
_CAMPAIGN = re.compile(r"(\d{4})/(\d{2})")


class LabelError(ValueError):
    """The labels file can't be read. The message is meant for the user."""


@dataclass(frozen=True)
class ClassCount:
    """The training parcels of one model class.

    Attributes:
        name: model class, one of `MODEL_CLASSES`.
        parcels: number of training parcels.
        crops: parcels per crop of the QGIS drop-down, most common first.
    """

    name: str
    parcels: int
    crops: dict[str, int]

    @property
    def status(self) -> str:
        """Whether the class trains on its own, must be merged, or is absent."""
        if self.parcels == 0:
            return "no parcels"
        if self.parcels < MIN_PARCELS_PER_CLASS:
            return f"below {MIN_PARCELS_PER_CLASS}: will be merged"
        return "own class"


@dataclass(frozen=True)
class LabelReport:
    """The result of `check_labels`.

    Attributes:
        total: rows in the file.
        training_parcels: photo labels with a known crop.
        field_verified: labels with `source = field`, reserved for the final test.
        regions: training parcels per region code, largest first.
        classes: every model class, largest first.
        problems: what is wrong with individual labels: message -> parcel IDs.
        unmet: FR-06 commitments that the set as a whole does not meet.
    """

    total: int
    training_parcels: int
    field_verified: int
    regions: dict[str, int]
    classes: list[ClassCount]
    problems: dict[str, list[str]]
    unmet: list[str]

    @property
    def ok(self) -> bool:
        """True when the labels are ready for the training-data step."""
        return not self.problems and not self.unmet


def read_labels(path: Path) -> gpd.GeoDataFrame:
    """Read the `labels` layer of a GeoPackage, in WGS84 lon/lat.

    Raises:
        LabelError: the file or layer can't be read, or a column is missing.
    """
    try:
        labels = gpd.read_file(path, layer=LAYER).to_crs(4326)
    except (OSError, RuntimeError, ValueError) as exc:
        raise LabelError(f"Can't read layer '{LAYER}' of {path}: {exc}") from exc
    missing = [column for column in COLUMNS if column not in labels.columns]
    if missing:
        raise LabelError(f"{path} has no column {', '.join(missing)}.")
    return labels


def check_labels(labels: gpd.GeoDataFrame) -> LabelReport:
    """Check labels against FR-06. Geometries are in WGS84; areas are measured in UTM.

    Args:
        labels: one row per label, with `COLUMNS` and a `geometry` column.

    Returns:
        The counts, the problems of individual labels, and the unmet commitments.
    """
    problems: dict[str, list[str]] = defaultdict(list)
    id_counts = Counter(_text(value) for value in labels["parcel_id"])
    shapes: list[tuple[str, BaseGeometry]] = []
    for position, row in enumerate(labels.itertuples(index=False), start=1):
        name = _text(row.parcel_id) or f"row {position}"
        for message in _label_problems(row, id_counts):
            problems[message].append(name)
        try:
            shapes.append((name, _valid_shape(row.geometry)))
        except ParcelError as exc:
            problems[str(exc)].append(name)
    overlapping = _overlapping(shapes)
    if overlapping:
        share = f"{MAX_OVERLAP_SHARE:.0%}"
        problems[f"overlaps another label by more than {share}"] = overlapping

    sources = labels["source"].map(_text)
    crops = labels["crop"].map(_text)
    training = labels[(sources == "photo") & crops.isin(list(CROP_CLASSES))]
    regions = Counter(_text(value) for value in training["region"] if _text(value))
    classes = _class_counts(_text(value) for value in training["crop"])
    field_verified = int((sources == "field").sum())

    return LabelReport(
        total=len(labels),
        training_parcels=len(training),
        field_verified=field_verified,
        regions=dict(regions.most_common()),
        classes=classes,
        problems=dict(problems),
        unmet=_unmet(len(training), field_verified, regions, classes),
    )


def _text(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _label_problems(row: Any, id_counts: Counter[str]) -> list[str]:
    found = _id_problems(_text(row.parcel_id), _text(row.region), id_counts)
    crop = _text(row.crop)
    if crop not in CROP_CLASSES:
        found.append(f"crop '{crop}' has no model class" if crop else "has no crop")
    found += _campaign_problems(_text(row.from_campaign), _text(row.to_campaign))
    source = _text(row.source)
    if source not in SOURCES:
        found.append(f"source must be one of {', '.join(SOURCES)}")
    if _text(row.confidence) not in CONFIDENCES:
        found.append(f"confidence must be one of {', '.join(CONFIDENCES)}")
    found += _evidence_problems(source, _text(row.evidence))
    return found


def _id_problems(parcel_id: str, region: str, id_counts: Counter[str]) -> list[str]:
    found = [] if region else ["has no region"]
    if not parcel_id:
        return [*found, "has no parcel_id"]
    if not _PARCEL_ID.fullmatch(parcel_id) or parcel_id[:3] != region:
        found.append("parcel_id must be the region code and 4 digits, like ARE-0001")
    if id_counts[parcel_id] > 1:
        found.append("parcel_id is used more than once")
    return found


def _campaign_problems(first: str, last: str) -> list[str]:
    start, end = _campaign_year(first), _campaign_year(last)
    if start is None or end is None:
        return ["from_campaign and to_campaign must look like 2023/24"]
    if start > end:
        return ["from_campaign comes after to_campaign"]
    if start < FIRST_CAMPAIGN_YEAR:
        return ["campaigns before 2018/19 have no satellite composites"]
    return []


def _campaign_year(campaign: str) -> int | None:
    match = _CAMPAIGN.fullmatch(campaign)
    if match is None or (int(match[1]) + 1) % 100 != int(match[2]):
        return None
    return int(match[1])


def _evidence_problems(source: str, evidence: str) -> list[str]:
    if source == "field":
        return [] if evidence else ["a field-verified label needs a note in evidence"]
    if source != "photo":
        return []
    match = _PHOTO_EVIDENCE.fullmatch(evidence)
    if match is None:
        return [f'evidence must read "{EVIDENCE_FORMAT}" (needs photo month and crop stage)']
    found = []
    if not 1 <= int(match[2]) <= 12 or match[1] < FIRST_PHOTO_MONTH:
        found.append("photo month must be a real month, August 2018 or later")
    if match[3] not in STAGES:
        found.append(f"crop stage must be one of {', '.join(STAGES)}")
    return found


def _valid_shape(geometry: BaseGeometry | None) -> BaseGeometry:
    if geometry is None or geometry.is_empty:
        raise ParcelError("has no polygon")
    return validate_parcel([ParcelFeature(None, geometry)]).geometry


def _overlapping(shapes: list[tuple[str, BaseGeometry]]) -> list[str]:
    if not shapes:
        return []
    names = [name for name, _ in shapes]
    geometries = [geometry for _, geometry in shapes]
    found: set[str] = set()
    pairs = STRtree(geometries).query(geometries, predicate="intersects")
    for left, right in zip(*pairs, strict=True):
        if left < right:
            a, b = geometries[left], geometries[right]
            if a.intersection(b).area > MAX_OVERLAP_SHARE * min(a.area, b.area):
                found |= {names[left], names[right]}
    return sorted(found)


def _class_counts(crops: Iterable[str]) -> list[ClassCount]:
    per_class: dict[str, Counter[str]] = {name: Counter() for name in MODEL_CLASSES}
    for crop in crops:
        per_class[CROP_CLASSES[crop]][crop] += 1
    classes = [
        ClassCount(name, counts.total(), dict(counts.most_common()))
        for name, counts in per_class.items()
    ]
    return sorted(classes, key=lambda item: -item.parcels)


def _unmet(
    training: int, field_verified: int, regions: Counter[str], classes: list[ClassCount]
) -> list[str]:
    unmet = []
    if training < MIN_TRAINING_PARCELS:
        unmet.append(f"{training} training parcels; at least {MIN_TRAINING_PARCELS} are required.")
    if len(regions) < MIN_REGIONS or REQUIRED_REGION not in regions:
        unmet.append(
            f"Training parcels come from {len(regions)} region(s); "
            f"at least {MIN_REGIONS} are required, including {REQUIRED_REGION}."
        )
    fallow = next(item.parcels for item in classes if item.name == FALLOW)
    if fallow < MIN_PARCELS_PER_CLASS:
        unmet.append(
            f"{FALLOW} has {fallow} parcels; it needs {MIN_PARCELS_PER_CLASS}, "
            "because it has no broader class to merge into."
        )
    if field_verified == 0:
        unmet.append("No field-verified parcels (source = field) for the final test.")
    return unmet
