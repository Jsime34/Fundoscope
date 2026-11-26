"""Analyzing and reopening stored parcels (FR-02), with the in-memory store and fakes."""

from datetime import date

import pytest

from fundoscope.db import MemoryParcelStore
from fundoscope.ingest import parse_parcel
from fundoscope.pipeline.parcel_store import analyze_parcel, parcel_key, reopen_parcel
from fundoscope.sentinel import ARCHIVE_START, SentinelError

TODAY = date(2024, 7, 31)  # campaigns 2018/19 to 2023/24, all over


def analyze(data, store, fetch, composites=None, today=TODAY, **options):
    return analyze_parcel(
        data,
        "lote.geojson",
        store,
        today=today,
        fetch=fetch,
        fetch_composites=composites,
        **options,
    )


def test_the_same_outline_is_the_same_parcel_whatever_the_file(parcel_geojson):
    from shapely.geometry import mapping

    geojson = parse_parcel(parcel_geojson, "lote.geojson")
    coords = mapping(geojson.geometry)["coordinates"][0][0]
    points = " ".join(f"{lon},{lat}" for lon, lat in coords)
    kml = (
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Placemark><Polygon><outerBoundaryIs>'
        f"<LinearRing><coordinates>{points}</coordinates></LinearRing>"
        "</outerBoundaryIs></Polygon></Placemark></kml>"
    ).encode()

    assert parcel_key(parse_parcel(kml, "lote.kml")) == parcel_key(geojson)


def test_the_first_analysis_stores_everything(parcel_geojson, fake_fetch, fake_composites):
    store = MemoryParcelStore()
    seen = []

    analysis = analyze(parcel_geojson, store, fake_fetch, fake_composites, on_parcel=seen.append)

    assert seen == [analysis.parcel_id] and analysis.downloaded
    assert fake_fetch.calls[0][2:] == (ARCHIVE_START, TODAY)
    stored = store.load(analysis.parcel_id)
    assert stored.observed_through == TODAY
    assert len(stored.observations) == len(analysis.history.daily.observations)
    assert set(stored.campaigns) == set(analysis.history.campaigns.index)
    assert all(c.computed_through.month == 7 for c in stored.campaigns.values())


def test_the_same_day_downloads_nothing(parcel_geojson, fake_fetch, fake_composites):
    store = MemoryParcelStore()
    first = analyze(parcel_geojson, store, fake_fetch, fake_composites)

    again = analyze(parcel_geojson, store, fake_fetch, fake_composites)

    assert again.parcel_id == first.parcel_id and not again.downloaded
    assert len(fake_fetch.calls) == 1 and len(fake_composites.calls) == 6
    assert again.history.campaigns.equals(first.history.campaigns)
    assert again.history.daily.processing_units == 0


def test_later_only_the_new_dates_and_the_campaign_in_progress_are_downloaded(
    parcel_geojson, fake_fetch, fake_composites
):
    store = MemoryParcelStore()
    analyze(parcel_geojson, store, fake_fetch, fake_composites)

    later = analyze(parcel_geojson, store, fake_fetch, fake_composites, today=date(2024, 8, 20))

    assert fake_fetch.calls[1][2:] == (date(2024, 8, 1), date(2024, 8, 20))
    assert [call[0] for call in fake_composites.calls[6:]] == [date(2024, 8, 1)]
    assert later.history.campaigns.index[-1] == "2024/25"
    assert later.history.daily.observations.index[0].date() < date(2018, 9, 1)  # the stored ones
    stored = store.load(later.parcel_id)
    assert stored.observed_through == date(2024, 8, 20)
    assert stored.campaigns["2024/25"].computed_through == date(2024, 8, 20)


def test_crops_from_another_model_version_are_predicted_again(
    parcel_geojson, fake_fetch, fake_composites, fake_crop_model
):
    store = MemoryParcelStore()
    options = {"model": fake_crop_model}
    analyze(parcel_geojson, store, fake_fetch, fake_composites, model_version="v1", **options)
    analyze(parcel_geojson, store, fake_fetch, fake_composites, model_version="v1", **options)
    assert len(fake_composites.calls) == 6  # same version: nothing redone

    redone = analyze(
        parcel_geojson, store, fake_fetch, fake_composites, model_version="v2", **options
    )

    assert len(fake_composites.calls) == 12
    assert {c.crop_model for c in store.load(redone.parcel_id).campaigns.values()} == {"v2"}


def test_a_stored_report_is_rebuilt_without_any_request(
    parcel_geojson, fake_fetch, fake_composites
):
    store = MemoryParcelStore()
    first = analyze(parcel_geojson, store, fake_fetch, fake_composites)

    reopened = reopen_parcel(first.parcel_id, store)

    assert len(fake_fetch.calls) == 1 and len(fake_composites.calls) == 6
    assert not reopened.downloaded
    assert reopened.history.campaigns.equals(first.history.campaigns)
    assert reopened.history.parcel.area_ha == pytest.approx(first.history.parcel.area_ha)


def test_nothing_to_reopen(parcel_geojson):
    from uuid import uuid4

    store = MemoryParcelStore()
    never = store.add(
        "k", None, parse_parcel(parcel_geojson, "l.geojson").geometry, 32718, 9.0, False
    )

    assert reopen_parcel(uuid4(), store) is None
    assert reopen_parcel(never.id, store) is None  # stored but never analyzed


def test_a_failed_download_saves_no_history(parcel_geojson):
    store = MemoryParcelStore()

    def failing(*_args):
        raise SentinelError("status 'FAILED'")

    with pytest.raises(SentinelError):
        analyze(parcel_geojson, store, failing)

    [parcel] = store.parcels()
    assert parcel.observed_through is None
