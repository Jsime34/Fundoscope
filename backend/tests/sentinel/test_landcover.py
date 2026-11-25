"""Tests for the WorldCover land cover. No network: offline graphs and synthetic files."""

import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box

from fundoscope.sentinel import SentinelError
from fundoscope.sentinel.landcover import (
    LAND_COVER_COLLECTION,
    build_land_cover_graph,
    fetch_land_cover,
    parse_land_cover,
)


def netcdf(tmp_path, dataset: xr.Dataset) -> bytes:
    path = tmp_path / "cover.nc"
    dataset.to_netcdf(path, engine="h5netcdf")
    return path.read_bytes()


def cover_file(tmp_path, with_time: bool = True) -> bytes:
    values = np.array([[40.0, 50.0], [80.0, np.nan]])
    coords = {"y": [8_400_015.0, 8_400_005.0], "x": [400_005.0, 400_015.0]}
    if with_time:
        data = xr.DataArray(values[None], dims=("t", "y", "x"), coords={"t": [0], **coords})
    else:
        data = xr.DataArray(values, dims=("y", "x"), coords=coords)
    return netcdf(tmp_path, data.to_dataset(name="MAP"))


def test_the_graph_takes_worldcover_2021_by_nearest_neighbor():
    graph = build_land_cover_graph((400_000, 8_400_000, 400_020, 8_400_020), 32718)

    load = graph["loadcollection1"]["arguments"]
    assert load["id"] == LAND_COVER_COLLECTION
    assert load["bands"] == ["MAP"]
    assert load["spatial_extent"] == {
        "west": 399_990,
        "south": 8_399_990,
        "east": 400_030,
        "north": 8_400_030,
        "crs": 32718,
    }
    resample = graph["resamplespatial1"]["arguments"]
    assert (resample["method"], resample["resolution"], resample["projection"]) == (
        "near",
        10,
        32718,
    )


@pytest.mark.parametrize("with_time", [True, False])
def test_parse_gives_classes_with_no_data_as_zero(tmp_path, with_time):
    cover = parse_land_cover(cover_file(tmp_path, with_time))

    assert cover.dims == ("y", "x")
    assert cover.dtype == np.uint8
    assert cover.to_numpy().tolist() == [[40, 50], [80, 0]]


def test_unreadable_content_is_an_error():
    with pytest.raises(SentinelError, match="not readable NetCDF"):
        parse_land_cover(b"<html>Service unavailable</html>")


def test_a_result_without_the_map_band_is_an_error(tmp_path):
    other = xr.DataArray(np.zeros((1, 1)), dims=("y", "x")).to_dataset(name="B04")

    with pytest.raises(SentinelError, match="no MAP band"):
        parse_land_cover(netcdf(tmp_path, other))


def test_fetch_reports_the_credits(tmp_path):
    content = cover_file(tmp_path)
    sent = []

    def send(graph):
        sent.append(graph)
        return content, 7.0

    cover = fetch_land_cover(box(400_000, 8_400_000, 400_020, 8_400_020), 32718, send=send)

    assert (cover.epsg, cover.credits, cover.data.shape) == (32718, 7.0, (2, 2))
    assert len(sent) == 1
