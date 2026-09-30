"""Parcel ingestion.

Read KML/KMZ/GeoJSON uploads, repair the geometry, pick the UTM zone, and compute the area
(FR-01). Pure: takes bytes in and returns shapes; no database or network.

Submodules: `utm` (zones and reprojection), `readers` (file formats), `parcels` (validation).
"""

from fundoscope.ingest.parcels import (
    MAX_AREA_HA,
    MIN_AREA_HA,
    Parcel,
    parse_parcel,
    validate_parcel,
)
from fundoscope.ingest.readers import ParcelError, ParcelFeature, read_features
from fundoscope.ingest.utm import from_utm, to_utm, utm_epsg

__all__ = [
    "MAX_AREA_HA",
    "MIN_AREA_HA",
    "Parcel",
    "ParcelError",
    "ParcelFeature",
    "from_utm",
    "parse_parcel",
    "read_features",
    "to_utm",
    "utm_epsg",
    "validate_parcel",
]
