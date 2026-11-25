"""Tests for the newest-image search. No network: a fake catalog."""

from datetime import date

import pytest
from requests import ConnectionError as RequestsConnectionError

import fundoscope.sentinel.catalog
from fundoscope.sentinel import SentinelError
from fundoscope.sentinel.catalog import newest_image_date

BOUNDS = (-75.74, -14.08, -75.72, -14.06)  # public farmland near Ica


def items(*days: str) -> list[dict]:
    return [{"properties": {"datetime": f"{day}T15:12:34.024Z"}} for day in days]


def test_the_newest_image_up_to_the_day():
    found = []

    def search(bounds, start, end):
        found.append((bounds, start, end))
        return items("2026-11-16", "2026-11-21", "2026-11-11")

    assert newest_image_date(BOUNDS, date(2026, 11, 22), search=search) == date(2026, 11, 21)
    assert found == [(BOUNDS, date(2026, 10, 22), date(2026, 11, 22))]


def test_images_after_the_day_are_ignored():
    search = lambda *_: items("2026-11-21", "2026-11-23")  # noqa: E731

    assert newest_image_date(BOUNDS, date(2026, 11, 22), search=search) == date(2026, 11, 21)


def test_no_image_in_a_month():
    assert newest_image_date(BOUNDS, date(2026, 11, 22), search=lambda *_: []) is None


def test_a_catalog_error_is_a_sentinel_error():
    def search(*_):
        raise RequestsConnectionError("no network")

    with pytest.raises(SentinelError, match="Could not search the Copernicus catalog"):
        newest_image_date(BOUNDS, date(2026, 11, 22), search=search)


def test_the_real_search_asks_the_catalog_for_sentinel2_dates(monkeypatch):
    calls = []

    class FakeCatalog:
        def __init__(self, config):
            calls.append(config.sh_base_url)

        def search(self, collection, bbox, time, fields):
            calls.append((collection.api_id, list(bbox), time, fields["include"]))
            return items("2026-11-21")

    monkeypatch.setattr(fundoscope.sentinel.catalog, "SentinelHubCatalog", FakeCatalog)

    assert newest_image_date(BOUNDS, date(2026, 11, 22)) == date(2026, 11, 21)
    assert calls[0] == "https://sh.dataspace.copernicus.eu"
    assert calls[1] == (
        "sentinel-2-l2a",
        list(BOUNDS),
        ("2026-10-22", "2026-11-22"),
        ["properties.datetime"],
    )
