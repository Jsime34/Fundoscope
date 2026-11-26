"""What every parcel store must do (in memory and in PostGIS)."""

import math
from datetime import date

import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, box

from fundoscope.db import OBSERVATION_COLUMNS, StoredCampaign, empty_observations

SQUARE = MultiPolygon([box(-75.731, -14.071, -75.729, -14.069)])  # public farmland near Ica


def add(store, key="abc", name="Lote demo"):
    return store.add(key, name, SQUARE, 32718, 4.6, False)


def observations(*days: str, ndvi: float = 0.5) -> pd.DataFrame:
    index = pd.DatetimeIndex(pd.to_datetime(list(days)), name="date")
    return pd.DataFrame(
        {"ndvi": ndvi, "ndwi": 0.1, "ndvi_std": 0.05, "valid_fraction": 1.0, "pixels": 400},
        index=index,
    )[list(OBSERVATION_COLUMNS)]


def test_a_parcel_is_found_by_its_key_and_its_id(any_store):
    parcel = add(any_store)

    assert any_store.find("abc") == parcel
    assert any_store.find("other") is None
    assert any_store.get(parcel.id) == parcel
    assert (parcel.name, parcel.utm_epsg, parcel.area_ha, parcel.repaired) == (
        "Lote demo",
        32718,
        4.6,
        False,
    )
    assert parcel.geometry.equals(SQUARE)
    assert parcel.observed_through is None and parcel.created_at is not None


def test_parcels_are_listed_newest_first(any_store):
    first = add(any_store, "a", "first")
    second = add(any_store, "b", "second")

    assert [p.id for p in any_store.parcels()] == [second.id, first.id]


def test_a_new_parcel_has_no_history(any_store):
    history = add(any_store) and any_store.load(any_store.find("abc").id)

    assert history.observations.empty and list(history.observations.columns) == list(
        OBSERVATION_COLUMNS
    )
    assert (history.observed_through, history.campaigns) == (None, {})


def test_observations_and_campaigns_are_saved_and_replaced(any_store):
    parcel = add(any_store)
    campaign = StoredCampaign(0.8, "grape", 0.9, date(2024, 7, 31), "v1")
    any_store.save(
        parcel.id,
        observations("2024-07-01", "2024-07-06"),
        date(2024, 7, 31),
        {"2023/24": campaign},
    )

    redone = StoredCampaign(float("nan"), None, float("nan"), date(2024, 8, 10))
    any_store.save(
        parcel.id,
        observations("2024-07-06", "2024-08-10", ndvi=0.6),
        date(2024, 8, 10),
        {"2024/25": redone},
    )

    history = any_store.load(parcel.id)
    assert history.observed_through == date(2024, 8, 10)
    assert any_store.get(parcel.id).observed_through == date(2024, 8, 10)
    assert list(history.observations.index.strftime("%Y-%m-%d")) == [
        "2024-07-01",
        "2024-07-06",
        "2024-08-10",
    ]
    assert history.observations["ndvi"].tolist() == pytest.approx([0.5, 0.6, 0.6])  # 07-06 replaced
    assert history.campaigns["2023/24"] == campaign
    again = history.campaigns["2024/25"]
    assert math.isnan(again.active_area) and again.crop is None and again.crop_model is None


def test_missing_values_survive_the_round_trip(any_store):
    parcel = add(any_store)
    cloudy = observations("2024-07-01", ndvi=float("nan"))
    any_store.save(parcel.id, cloudy, date(2024, 7, 1), {})

    assert math.isnan(any_store.load(parcel.id).observations["ndvi"].iloc[0])


def test_saving_nothing_new_still_moves_the_day(any_store):
    parcel = add(any_store)

    any_store.save(parcel.id, empty_observations(), date(2024, 7, 31), {})

    assert any_store.load(parcel.id).observed_through == date(2024, 7, 31)
