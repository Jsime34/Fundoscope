from datetime import date

import numpy as np
import pytest
import shapely

from fundoscope.ingest import ParcelError, parse_parcel
from fundoscope.pipeline import parcel_history
from fundoscope.pipeline.parcel_history import interior_stack
from fundoscope.sentinel import COMPOSITE_BANDS


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


# --- % active area (monthly composites) ----------------------------------------------------

BARE_SHARE = 7 / 30  # the fake parcel is 30 pixels wide and its western 7 columns are bare


def test_history_adds_active_area_per_campaign(parcel_geojson, fake_fetch, fake_composites):
    history = parcel_history(
        parcel_geojson,
        "lote.geojson",
        date(2022, 8, 1),
        date(2024, 7, 31),
        fetch=fake_fetch,
        fetch_composites=fake_composites,
    )
    assert history.campaigns["active_area"].to_numpy() == pytest.approx([1 - BARE_SHARE] * 2)
    assert history.credits == 14.0
    assert history.active_area_failures == 0
    assert sorted(fake_composites.calls) == [
        (date(2022, 8, 1), date(2023, 7, 31), ("B04", "B08")),  # one request per campaign,
        (date(2023, 8, 1), date(2024, 7, 31), ("B04", "B08")),  # only the two NDVI bands
    ]


def test_campaign_requests_are_clipped_to_the_period(parcel_geojson, fake_fetch, fake_composites):
    parcel_history(
        parcel_geojson,
        "lote.geojson",
        date(2023, 1, 1),
        date(2023, 12, 31),
        fetch=fake_fetch,
        fetch_composites=fake_composites,
    )
    assert sorted(call[:2] for call in fake_composites.calls) == [
        (date(2023, 1, 1), date(2023, 7, 31)),
        (date(2023, 8, 1), date(2023, 12, 31)),
    ]


def test_a_failed_campaign_request_is_retried_once(parcel_geojson, fake_fetch, make_composites):
    flaky = make_composites(failures={"2022-08-01": 1})
    history = parcel_history(
        parcel_geojson,
        "lote.geojson",
        date(2022, 8, 1),
        date(2023, 7, 31),
        fetch=fake_fetch,
        fetch_composites=flaky,
    )
    assert len(flaky.calls) == 2
    assert history.campaigns.loc["2022/23", "active_area"] == pytest.approx(1 - BARE_SHARE)


def test_a_campaign_failing_twice_gets_no_active_area(parcel_geojson, fake_fetch, make_composites):
    broken = make_composites(failures={"2022-08-01": 2})
    history = parcel_history(
        parcel_geojson,
        "lote.geojson",
        date(2022, 8, 1),
        date(2024, 7, 31),
        fetch=fake_fetch,
        fetch_composites=broken,
    )
    assert np.isnan(history.campaigns.loc["2022/23", "active_area"])
    assert history.campaigns.loc["2023/24", "active_area"] == pytest.approx(1 - BARE_SHARE)
    assert history.active_area_failures == 1  # the rest of the history is still delivered


def test_without_composites_active_area_is_empty_and_free(parcel_geojson, fake_fetch):
    history = parcel_history(
        parcel_geojson, "lote.geojson", date(2022, 8, 1), date(2023, 7, 31), fetch=fake_fetch
    )
    assert history.campaigns["active_area"].isna().all()
    assert history.credits == 0.0


def test_a_fully_cloudy_campaign_is_unknown_not_a_failure(
    parcel_geojson, fake_fetch, make_composites
):
    cloudy = make_composites(cloudy={"2022-08-01"})
    history = parcel_history(
        parcel_geojson,
        "lote.geojson",
        date(2022, 8, 1),
        date(2023, 7, 31),
        fetch=fake_fetch,
        fetch_composites=cloudy,
    )
    assert np.isnan(history.campaigns.loc["2022/23", "active_area"])
    assert history.active_area_failures == 0  # openEO delivered; there was just nothing clear
    assert history.credits == 7.0


# Inside the 15 m margin the fake parcel has about 26 x 26 pixels, 5 of the 26 columns bare
# (the square is slightly tilted after the trip through WGS84, hence the tolerance).
GRAPE_CONFIDENCE = pytest.approx((21 * 0.9 + 5 * 0.1) / 26, abs=0.01)


def test_with_a_model_each_campaign_gets_its_crop(
    parcel_geojson, fake_fetch, fake_composites, fake_crop_model
):
    history = parcel_history(
        parcel_geojson,
        "lote.geojson",
        date(2022, 8, 1),
        date(2024, 7, 31),
        fetch=fake_fetch,
        fetch_composites=fake_composites,
        model=fake_crop_model,
    )

    assert history.campaigns["crop"].tolist() == ["grape", "grape"]
    assert history.campaigns["crop_confidence"].tolist() == [GRAPE_CONFIDENCE] * 2
    assert history.campaigns["active_area"].to_numpy() == pytest.approx([1 - BARE_SHARE] * 2)
    assert {call[2] for call in fake_composites.calls} == {COMPOSITE_BANDS}  # one request each
    assert history.credits == 14.0


def test_a_campaign_in_progress_gets_no_crop(
    parcel_geojson, fake_fetch, fake_composites, fake_crop_model
):
    history = parcel_history(
        parcel_geojson,
        "lote.geojson",
        date(2022, 8, 1),
        date(2023, 12, 31),  # 2023/24 has only August to December so far
        fetch=fake_fetch,
        fetch_composites=fake_composites,
        model=fake_crop_model,
    )

    assert history.campaigns["crop"].tolist() == ["grape", None]
    assert np.isnan(history.campaigns.loc["2023/24", "crop_confidence"])


def test_without_a_model_there_is_no_crop(parcel_geojson, fake_fetch, fake_composites):
    history = parcel_history(
        parcel_geojson,
        "lote.geojson",
        date(2022, 8, 1),
        date(2024, 7, 31),
        fetch=fake_fetch,
        fetch_composites=fake_composites,
    )

    assert history.campaigns["crop"].isna().all()
    assert history.campaigns["crop_confidence"].isna().all()


def test_composites_without_every_band_give_no_crop_but_still_the_active_area(
    parcel_geojson, fake_fetch, make_composites, fake_crop_model
):
    fake = make_composites(only_bands=("B04", "B08"))

    history = parcel_history(
        parcel_geojson,
        "lote.geojson",
        date(2022, 8, 1),
        date(2023, 7, 31),
        fetch=fake_fetch,
        fetch_composites=fake,
        model=fake_crop_model,
    )

    assert history.campaigns["crop"].tolist() == [None]
    assert history.campaigns["active_area"].to_numpy() == pytest.approx([1 - BARE_SHARE])


def test_the_model_sees_twelve_months_of_the_interior_pixels(parcel_geojson, make_composites):
    parcel = parse_parcel(parcel_geojson, "lote.geojson")
    fake = make_composites()
    composites = fake(
        parcel.geometry_utm, parcel.utm_epsg, date(2023, 8, 1), date(2024, 1, 31), bands=("B04",)
    )

    stack = interior_stack(composites, parcel.geometry_utm, "2023/24")

    assert len(stack["month"]) == 12
    assert str(stack["month"].to_numpy()[0])[:10] == "2023-08-01"
    assert np.isnan(stack.sel(band="B04")[6:]).all()  # February to July are not there yet
    clear = ~np.isnan(stack.sel(band="B04")[0].to_numpy())
    xx, yy = np.meshgrid(stack["x"].to_numpy(), stack["y"].to_numpy())
    interior = shapely.contains_xy(parcel.geometry_utm.buffer(-15), xx, yy)
    assert (clear == interior).all()  # pixels within 15 m of the edge are left out
    assert 0 < clear.sum() < clear.size
