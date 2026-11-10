"""Parcel ingestion.

Read KML/KMZ/GeoJSON uploads, repair the geometry, pick the UTM zone, and compute the area
(FR-01). Pure: takes bytes in and returns shapes; no database or network.

Submodules: `utm` (zones and reprojection), `readers` (file formats), `parcels` (validation),
`labels` (checks the labeled training parcels, FR-06).
"""

from fundoscope.ingest.labels import (
    CROP_CLASSES,
    MODEL_CLASSES,
    ClassCount,
    LabelError,
    LabelReport,
    check_labels,
    read_labels,
)
from fundoscope.ingest.parcels import (
    MAX_AREA_HA,
    MIN_AREA_HA,
    Parcel,
    parse_parcel,
    validate_parcel,
)
from fundoscope.ingest.readers import ParcelError, ParcelFeature, read_features
from fundoscope.ingest.utm import from_utm, to_utm, utm_epsg, zone_label

__all__ = [
    "CROP_CLASSES",
    "MAX_AREA_HA",
    "MIN_AREA_HA",
    "MODEL_CLASSES",
    "ClassCount",
    "LabelError",
    "LabelReport",
    "Parcel",
    "ParcelError",
    "ParcelFeature",
    "check_labels",
    "from_utm",
    "parse_parcel",
    "read_features",
    "read_labels",
    "to_utm",
    "utm_epsg",
    "validate_parcel",
    "zone_label",
]
