import json

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box, mapping

from fundoscope.ingest import from_utm, to_utm
from fundoscope.sentinel import DailyStats


@pytest.fixture
def parcel_geojson() -> bytes:
    """A 300 m x 300 m (9 ha) parcel near Ica, as an uploaded GeoJSON file."""
    center = to_utm(Point(-75.73, -14.07), 32718)
    square = box(center.x - 150, center.y - 150, center.x + 150, center.y + 150)
    return json.dumps(
        {
            "type": "Feature",
            "properties": {"name": "Lote demo"},
            "geometry": mapping(from_utm(square, 32718)),
        }
    ).encode()


class FakeFetch:
    """Replaces `fetch_daily_stats`: a vineyard-like yearly NDVI cycle, no network."""

    def __init__(self):
        self.calls: list[tuple] = []

    def __call__(self, geometry_utm, epsg, start, end):
        self.calls.append((geometry_utm, epsg, start, end))
        index = pd.date_range(start, end, freq="5D", name="date")
        day_of_year = index.dayofyear.to_numpy()
        # Low (0.2) around June, full canopy (0.75) around October.
        ndvi = 0.475 + 0.275 * np.cos(2 * np.pi * (day_of_year - 280) / 365)
        observations = pd.DataFrame(
            {"ndvi": ndvi, "ndwi": 0.15, "ndvi_std": 0.05, "valid_fraction": 1.0, "pixels": 900},
            index=index,
        )
        return DailyStats(observations, processing_units=2.5, requests=2, failed_intervals=0)


@pytest.fixture
def fake_fetch() -> FakeFetch:
    return FakeFetch()
