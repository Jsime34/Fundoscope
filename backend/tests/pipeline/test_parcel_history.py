from datetime import date

import pytest

from fundoscope.ingest import ParcelError
from fundoscope.pipeline import parcel_history


def test_history_chains_ingest_sentinel_and_timeseries(parcel_geojson, fake_fetch):
    history = parcel_history(
        parcel_geojson, "lote.geojson", date(2022, 8, 1), date(2024, 7, 31), fetch=fake_fetch
    )

    # ingest: the parcel was validated and measured
    assert history.parcel.name == "Lote demo"
    assert history.parcel.area_ha == pytest.approx(9.0, rel=1e-3)

    # sentinel: called once with the UTM geometry, its zone, and the requested period
    [(geometry, epsg, start, end)] = fake_fetch.calls
    assert geometry.equals(history.parcel.geometry_utm)
    assert (epsg, start, end) == (32718, date(2022, 8, 1), date(2024, 7, 31))
    assert history.daily.processing_units == 2.5

    # timeseries: two complete campaigns with a vineyard-like peak
    assert list(history.campaigns.index) == ["2022/23", "2023/24"]
    assert history.campaigns["complete"].all()
    assert history.campaigns["peak_ndvi"].to_numpy() == pytest.approx([0.75, 0.75], abs=0.02)


def test_a_bad_file_is_rejected_before_downloading_anything(fake_fetch):
    with pytest.raises(ParcelError):
        parcel_history(b"<kml><Placemark>", "broken.kml", fetch=fake_fetch)
    assert fake_fetch.calls == []  # no processing units spent on an invalid upload
