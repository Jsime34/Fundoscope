import geopandas as gpd
from shapely.geometry import Point, box

import fundoscope.cli
from fundoscope.cli import main
from fundoscope.cli.labels import default_labels_file, run_labels_check
from fundoscope.ingest import from_utm, to_utm


def write_labels(
    path, evidence="SV 2024-08 mature https://maps.app.goo.gl/abc", count=150, satellite=False
):
    """`count` training labels (the last 15 fallow) and 1 field-verified one.

    With `satellite`, the fallow labels come from satellite images instead of Street View.
    """
    center = to_utm(Point(-75.73, -14.07), 32718)
    regions = ["ARE", "ICA", "LLI"]
    crops = ["grape"] * (count - 15) + ["fallow"] * 15
    rows = [
        {
            "parcel_id": f"{regions[i % 3]}-{i + 1:04d}",
            "region": regions[i % 3],
            "crop": crop,
            "from_campaign": "2024/25",
            "to_campaign": "2024/25",
            "source": "photo",
            "confidence": "high",
            "evidence": evidence,
            "geometry": from_utm(
                box(center.x + i * 500, center.y, center.x + i * 500 + 100, center.y + 100), 32718
            ),
        }
        for i, crop in enumerate(crops)
    ]
    rows[0] = rows[0] | {"source": "field", "evidence": "Family records"}
    if satellite:
        pin = {"source": "satellite", "evidence": "SAT 2024-08 bare https://maps.app.goo.gl/pin"}
        rows[-15:] = [row | pin for row in rows[-15:]]
    extra = rows[1] | {
        "parcel_id": "ARE-0999",
        "region": "ARE",
        "geometry": from_utm(box(center.x, center.y + 900, center.x + 100, center.y + 1000), 32718),
    }
    gpd.GeoDataFrame([*rows, extra], geometry="geometry", crs=4326).to_file(
        path, layer="labels", driver="GPKG"
    )


def test_ready_labels_exit_with_success(tmp_path, capsys):
    path = tmp_path / "labels.gpkg"
    write_labels(path)

    code = run_labels_check(path)

    printed = capsys.readouterr().out
    assert code == 0
    assert "151 labels: 150 for training, 1 field-verified" in printed
    assert "Ready for the training-data step." in printed
    assert "Fix these labels" not in printed


def test_satellite_labels_are_counted_apart(tmp_path, capsys):
    path = tmp_path / "labels.gpkg"
    write_labels(path, satellite=True)

    assert run_labels_check(path) == 0
    assert "150 for training (15 from satellite images)" in capsys.readouterr().out


def test_problems_are_grouped_and_shortened(tmp_path, capsys):
    path = tmp_path / "labels.gpkg"
    write_labels(path, evidence="https://maps.app.goo.gl/abc")

    code = run_labels_check(path)

    printed = capsys.readouterr().out
    assert code == 1
    assert "[150] evidence must read" in printed
    assert "and 144 more" in printed
    assert "Not ready yet." in printed


def test_too_few_parcels_are_reported(tmp_path, capsys):
    path = tmp_path / "labels.gpkg"
    write_labels(path, count=20)

    assert run_labels_check(path) == 1

    printed = capsys.readouterr().out
    assert "Not met yet (FR-06):" in printed
    assert "21 labels: 20 for training" in printed


def test_a_short_problem_list_shows_every_id(tmp_path, capsys):
    path = tmp_path / "labels.gpkg"
    write_labels(path)
    labels = gpd.read_file(path, layer="labels")
    labels.loc[labels.index[1:3], "evidence"] = "https://maps.app.goo.gl/abc"
    labels.to_file(path, layer="labels", driver="GPKG")

    assert run_labels_check(path) == 1

    printed = capsys.readouterr().out
    assert "crop stage): ICA-0002, LLI-0003\n" in printed


def test_a_mixed_class_lists_its_crops(tmp_path, capsys):
    path = tmp_path / "labels.gpkg"
    write_labels(path)
    labels = gpd.read_file(path, layer="labels")
    labels.loc[labels.index[1], "crop"] = "corn"
    labels.loc[labels.index[2], "crop"] = "onion"
    labels.to_file(path, layer="labels", driver="GPKG")

    run_labels_check(path)

    assert "(corn 1, onion 1)" in capsys.readouterr().out


def test_an_unreadable_file_is_reported(tmp_path, capsys):
    assert run_labels_check(tmp_path / "nope.gpkg") == 1
    assert "Can't read layer" in capsys.readouterr().out


def test_default_labels_file_is_in_the_data_folder():
    assert default_labels_file().parts[-3:] == ("data", "labels", "labels.gpkg")


def test_main_runs_the_check(monkeypatch, tmp_path):
    received = []
    monkeypatch.setattr(fundoscope.cli, "run_labels_check", lambda file: received.append(file) or 0)

    assert main(["labels", "check", str(tmp_path / "x.gpkg")]) == 0
    assert main(["labels", "check"]) == 0

    assert received == [tmp_path / "x.gpkg", None]
