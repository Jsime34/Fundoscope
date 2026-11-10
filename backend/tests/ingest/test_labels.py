import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from fundoscope.ingest import (
    CROP_CLASSES,
    MODEL_CLASSES,
    ClassCount,
    LabelError,
    check_labels,
    from_utm,
    read_labels,
    to_utm,
)

LINK = "https://maps.app.goo.gl/abc123"


def square(index: int = 0, side: float = 100.0, shift: float = 0.0):
    """A square near Ica in WGS84; squares with different indexes are 500 m apart."""
    center = to_utm(Point(-75.73, -14.07), 32718)
    x = center.x + index * 500 + shift
    return from_utm(box(x, center.y, x + side, center.y + side), 32718)


def label(index: int = 0, region: str = "ICA", **changes) -> dict:
    row = {
        "parcel_id": f"{region}-{index + 1:04d}",
        "region": region,
        "crop": "grape",
        "from_campaign": "2023/24",
        "to_campaign": "2024/25",
        "source": "photo",
        "confidence": "high",
        "evidence": f"SV 2024-08 mature {LINK}",
        "geometry": square(index),
    }
    return row | changes


def table(rows: list[dict]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)


def complete_rows() -> list[dict]:
    """150 training parcels from 3 regions, with every required class, plus one test parcel."""
    regions = ["ARE", "ICA", "LLI"]
    crops = ["grape"] * 45 + ["avocado"] * 45 + ["corn"] * 45 + ["fallow"] * 15
    rows = [
        label(i, regions[i % 3], crop=crop) | (_bare() if crop == "fallow" else {})
        for i, crop in enumerate(crops)
    ]
    return [*rows, label(200, "ARE", source="field", crop="paprika", evidence="Family records")]


def _bare() -> dict:
    return {"evidence": f"SV 2024-08 bare {LINK}"}


def test_a_complete_set_is_ready():
    report = check_labels(table(complete_rows()))

    assert report.ok
    assert report.problems == {}
    assert report.unmet == []
    assert (report.total, report.training_parcels, report.field_verified) == (151, 150, 1)
    assert report.regions == {"ARE": 50, "ICA": 50, "LLI": 50}


def test_field_verified_parcels_never_count_for_training():
    report = check_labels(table(complete_rows()))

    other_annual = next(c for c in report.classes if c.name == "other annual crops")
    assert report.training_parcels == 150
    assert "paprika" not in other_annual.crops


def test_every_crop_maps_to_a_model_class():
    assert set(CROP_CLASSES.values()) == set(MODEL_CLASSES)


def test_crops_are_grouped_into_the_model_classes():
    rows = [
        label(0, crop="corn"),
        label(1, crop="corn"),
        label(2, crop="onion"),
        label(3, crop="mandarine"),
        label(4, crop="bare", evidence=f"SV 2024-08 bare {LINK}"),
    ]
    classes = {c.name: c for c in check_labels(table(rows)).classes}

    assert classes["other annual crops"].crops == {"corn": 2, "onion": 1}
    assert classes["citrus"].parcels == 1
    assert classes["fallow/bare"].crops == {"bare": 1}
    assert list(classes)[0] == "other annual crops"  # largest first


@pytest.mark.parametrize(
    ("parcels", "status"),
    [(0, "no parcels"), (14, "below 15: will be merged"), (15, "own class")],
)
def test_class_status(parcels, status):
    assert ClassCount("citrus", parcels, {}).status == status


def test_too_few_training_parcels():
    report = check_labels(table(complete_rows()[1:]))
    assert any("149 training parcels" in item for item in report.unmet)
    assert not report.ok


def test_too_few_regions():
    rows = [r for r in complete_rows() if r["region"] != "LLI"]
    assert any("from 2 region(s)" in item for item in check_labels(table(rows)).unmet)


def test_majes_must_be_one_of_the_regions():
    rows = [label(i, "ICA") for i in range(3)] + [label(3, "LLI"), label(4, "PIU")]
    assert any("including ARE" in item for item in check_labels(table(rows)).unmet)


def test_fallow_needs_its_own_parcels():
    rows = [r for r in complete_rows() if r["crop"] != "fallow"]
    unmet = check_labels(table(rows)).unmet
    assert any("fallow/bare has 0 parcels" in item for item in unmet)


def test_the_final_test_needs_field_verified_parcels():
    rows = complete_rows()[:-1]
    assert any("field-verified" in item for item in check_labels(table(rows)).unmet)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"parcel_id": None}, "has no parcel_id"),
        ({"parcel_id": "ICA-12"}, "parcel_id must be"),
        ({"parcel_id": "ARE-0001"}, "parcel_id must be"),
        ({"region": None, "parcel_id": "ICA-0001"}, "has no region"),
        ({"crop": None}, "has no crop"),
        ({"crop": "kiwi"}, "crop 'kiwi' has no model class"),
        ({"from_campaign": "2023"}, "must look like 2023/24"),
        ({"to_campaign": "2024/26"}, "must look like 2023/24"),
        ({"from_campaign": "2024/25", "to_campaign": "2023/24"}, "comes after"),
        ({"from_campaign": "2017/18"}, "before 2018/19"),
        ({"source": "wayback"}, "source must be"),
        ({"confidence": "low"}, "confidence must be"),
        ({"evidence": LINK}, "needs photo month and crop stage"),
        ({"evidence": None}, "needs photo month and crop stage"),
        ({"evidence": f"SV 2018-07 mature {LINK}"}, "August 2018 or later"),
        ({"evidence": f"SV 2024-13 mature {LINK}"}, "a real month"),
        ({"evidence": f"SV 2024-08 ripe {LINK}"}, "crop stage must be"),
        ({"source": "field", "evidence": ""}, "needs a note in evidence"),
        ({"geometry": None}, "has no polygon"),
        ({"geometry": square(side=50.0)}, "the minimum is 0.5 ha"),
        ({"geometry": box(10, 10, 11, 11)}, "outside Peru"),
    ],
)
def test_problems_with_a_single_label(changes, expected):
    report = check_labels(table([label(**changes)]))
    assert any(expected in message for message in report.problems), report.problems
    assert not report.ok


def test_a_field_verified_label_needs_no_street_view_link():
    report = check_labels(table([label(source="field", evidence="Family records")]))
    assert report.problems == {}


def test_a_label_without_an_id_is_named_by_its_row():
    report = check_labels(table([label(0), label(1, parcel_id=None)]))
    assert report.problems["has no parcel_id"] == ["row 2"]


def test_repeated_ids_are_reported():
    report = check_labels(table([label(0), label(1, parcel_id="ICA-0001")]))
    assert report.problems["parcel_id is used more than once"] == ["ICA-0001", "ICA-0001"]


def test_the_same_field_drawn_twice_is_reported():
    rows = [label(0), label(1, geometry=square(0, shift=20.0))]
    assert check_labels(table(rows)).problems["overlaps another label by more than 10%"] == [
        "ICA-0001",
        "ICA-0002",
    ]


def test_neighbors_that_barely_touch_are_fine():
    rows = [label(0), label(1, geometry=square(0, shift=95.0))]
    assert check_labels(table(rows)).problems == {}


def test_no_labels_at_all():
    report = check_labels(table([label()]).iloc[0:0])
    assert report.total == 0
    assert not report.ok


def test_read_labels_round_trip(tmp_path):
    path = tmp_path / "labels.gpkg"
    table([label(0), label(1)]).to_file(path, layer="labels", driver="GPKG")

    labels = read_labels(path)

    assert labels.crs.to_epsg() == 4326
    assert list(labels["parcel_id"]) == ["ICA-0001", "ICA-0002"]
    assert check_labels(labels).problems == {}


def test_read_labels_reprojects_to_wgs84(tmp_path):
    path = tmp_path / "labels.gpkg"
    table([label()]).to_crs(32718).to_file(path, layer="labels", driver="GPKG")
    assert read_labels(path).crs.to_epsg() == 4326


def test_read_labels_missing_file(tmp_path):
    with pytest.raises(LabelError, match="Can't read layer"):
        read_labels(tmp_path / "nope.gpkg")


def test_read_labels_wrong_layer(tmp_path):
    path = tmp_path / "labels.gpkg"
    table([label()]).to_file(path, layer="other", driver="GPKG")
    with pytest.raises(LabelError, match="Can't read layer"):
        read_labels(path)


def test_read_labels_missing_column(tmp_path):
    path = tmp_path / "labels.gpkg"
    table([label()]).drop(columns="evidence").to_file(path, layer="labels", driver="GPKG")
    with pytest.raises(LabelError, match="has no column evidence"):
        read_labels(path)
