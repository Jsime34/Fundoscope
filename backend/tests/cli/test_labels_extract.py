import geopandas as gpd
from shapely.geometry import Point, box

import fundoscope.cli
from fundoscope.cli import main
from fundoscope.cli.labels import default_features_dir, run_labels_extract
from fundoscope.ingest import from_utm, to_utm

LINK = "https://maps.app.goo.gl/abc"
CENTER = to_utm(Point(-75.73, -14.07), 32718)  # public farmland near Ica
FIRST_FALLOW = 135  # labels 0-134 are grape, 135-149 fallow


def write_labels(path, evidence=f"SV 2024-08 mature {LINK}", narrow=False, notes=None):
    """150 photo labels from 3 regions that pass `labels check`, plus a field-verified one.

    `notes` maps a label's index to its notes.
    """
    regions = ["ARE", "ICA", "LLI"]
    crops = ["grape"] * FIRST_FALLOW + ["fallow"] * (150 - FIRST_FALLOW)
    notes = notes or {}

    def square(i, width=100.0):
        x = CENTER.x + i * 500
        return from_utm(box(x, CENTER.y, x + width, CENTER.y + 250), 32718)

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
            "notes": notes.get(i),
            "geometry": square(i, 25.0 if narrow and i == 0 else 100.0),
        }
        for i, crop in enumerate(crops)
    ]
    rows.append(rows[1] | {"parcel_id": "ICA-0999", "source": "field", "geometry": square(999)})
    gpd.GeoDataFrame(rows, geometry="geometry", crs=4326).to_file(
        path, layer="labels", driver="GPKG"
    )


def parcel_id(i):
    return f"{['ARE', 'ICA', 'LLI'][i % 3]}-{i + 1:04d}"


def fallow_is_bare(make_campaign_composites, **options):
    """Fake composites where the fallow labels stay bare all year, as they should."""
    first_fallow_x = CENTER.x + FIRST_FALLOW * 500 - 250
    return make_campaign_composites(
        bare=lambda geometry: geometry.bounds[0] > first_fallow_x, **options
    )


def run(tmp_path, fetch=None, used=0.0, cap=None, **options):
    return run_labels_extract(
        tmp_path / "labels.gpkg", used, cap, out=tmp_path / "features", fetch=fetch, **options
    )


def test_a_dry_run_only_counts(tmp_path, make_campaign_composites, capsys):
    write_labels(tmp_path / "labels.gpkg")
    fake = make_campaign_composites()

    assert run(tmp_path, fake, used=None, dry_run=True) == 0

    printed = capsys.readouterr().out
    assert "150 parcel-campaigns: 0 already saved, 150 to download" in printed
    assert "about 1,050 openEO credits" in printed
    assert fake.calls == []


def test_the_quota_must_be_given(tmp_path, make_campaign_composites, capsys):
    write_labels(tmp_path / "labels.gpkg")
    assert run(tmp_path, make_campaign_composites(), used=None) == 1
    assert "Pass --used-credits N" in capsys.readouterr().out


def test_labels_must_pass_the_check_first(tmp_path, capsys):
    write_labels(tmp_path / "labels.gpkg", evidence=LINK)
    assert run(tmp_path) == 1
    assert "run `fundoscope labels check`" in capsys.readouterr().out


def test_an_unreadable_file_is_reported(tmp_path, capsys):
    assert run(tmp_path) == 1
    assert "Can't read layer" in capsys.readouterr().out


def test_a_full_run(tmp_path, make_campaign_composites, capsys):
    write_labels(tmp_path / "labels.gpkg")

    fake = fallow_is_bare(make_campaign_composites)

    assert run(tmp_path, fake, used=1_500) == 0

    printed = capsys.readouterr().out
    assert "This run may spend up to 6,500 credits." in printed
    assert "[150/150]" in printed
    assert "Saved 150 parcel-campaigns in this run (1,050.0 openEO credits)." in printed
    assert "All training pixels are saved" in printed
    assert "NDVI check: all 150 labeled campaigns fit their crop. Ready for training." in printed
    assert (tmp_path / "features" / "pixels" / "ARE-0001_2024-25.csv").exists()

    assert run(tmp_path, fake, used=None) == 0  # nothing left to download: no quota needed
    printed = capsys.readouterr().out
    assert "150 already saved, 0 to download" in printed
    assert "NDVI check: all 150" in printed
    assert len(fake.calls) == 150


def test_the_cap_stops_the_run_and_a_second_run_finishes(
    tmp_path, make_campaign_composites, capsys
):
    write_labels(tmp_path / "labels.gpkg")
    fake = fallow_is_bare(make_campaign_composites)

    assert run(tmp_path, fake, used=None, cap=15) == 1
    printed = capsys.readouterr().out
    assert "This run may spend up to 15 credits." in printed
    assert "Stopped at the credit cap" in printed
    assert "149 parcel-campaigns are still to download." in printed

    assert run(tmp_path, fake, used=15, cap=5_000) == 0
    assert "1 already saved, 149 to download" in capsys.readouterr().out
    assert len(fake.calls) == 150


def test_problems_are_summarized(tmp_path, make_campaign_composites, capsys):
    write_labels(tmp_path / "labels.gpkg", narrow=True)
    fake = make_campaign_composites(cloudy={"2024-08-01"}, fail={"2024-08-01": 3})

    assert run(tmp_path, fake) == 1

    printed = capsys.readouterr().out
    assert "ARE-0001 2024/25: too narrow, 0.0 credits" in printed
    assert "simulated openEO failure" in printed
    assert "1 are too narrow" in printed
    assert "3 failed; the next run retries them" in printed
    assert "Stopped: 3 requests in a row failed." in printed

    assert run(tmp_path, fake) == 0
    printed = capsys.readouterr().out
    assert "had no clear pixels; their files are empty." in printed
    assert "NDVI check: all 0 labeled campaigns fit their crop." in printed


def test_labels_that_do_not_fit_their_ndvi_curve_must_be_reviewed(
    tmp_path, make_campaign_composites, capsys
):
    write_labels(tmp_path / "labels.gpkg")
    fake = make_campaign_composites()  # every parcel is green, the fallow ones too

    assert run(tmp_path, fake) == 1

    printed = capsys.readouterr().out
    assert (
        "NDVI check: 15 of 150 labeled campaigns don't fit their crop; 0 reviewed, 15 to review."
        in printed
    )
    assert "  Aug  Sep  Oct" in printed
    assert f"{parcel_id(FIRST_FALLOW)} 2024/25 fallow" + " " * 8 + "0.64 0.64" in printed
    assert "    a crop grew: NDVI 0.64 in Aug" in printed
    assert 'write "NDVI ok: <your reason>" in its notes' in printed

    fallow = range(FIRST_FALLOW, 150)
    notes = {i: "NDVI ok: checked, a crop was planted" for i in fallow if i != 149}
    write_labels(tmp_path / "labels.gpkg", notes=notes)

    assert run(tmp_path, fake) == 1

    printed = capsys.readouterr().out
    assert "14 reviewed, 1 to review." in printed
    assert f"{parcel_id(149)} 2024/25 fallow" in printed
    assert f"{parcel_id(FIRST_FALLOW)} 2024/25 fallow" not in printed
    assert f"Reviewed and kept: {parcel_id(FIRST_FALLOW)} 2024/25" in printed
    assert "and 8 more" in printed

    write_labels(tmp_path / "labels.gpkg", notes=dict.fromkeys(fallow, "NDVI ok: checked"))

    assert run(tmp_path, fake) == 0

    printed = capsys.readouterr().out
    assert "15 reviewed, 0 to review." in printed
    assert "Monthly NDVI" not in printed
    assert "Ready for training." in printed
    assert len(fake.calls) == 150  # the check downloads nothing


def test_default_features_dir_is_in_the_data_folder():
    assert default_features_dir().parts[-2:] == ("data", "features")


def test_main_runs_the_extraction(monkeypatch, tmp_path):
    received = []
    monkeypatch.setattr(
        fundoscope.cli,
        "run_labels_extract",
        lambda *args, **kwargs: received.append((args, kwargs)) or 0,
    )

    assert main(["labels", "extract", "--used-credits", "1200", "--cap", "300"]) == 0
    assert main(["labels", "extract", str(tmp_path / "x.gpkg"), "--dry-run"]) == 0

    assert received == [
        ((None, 1200.0, 300.0), {"dry_run": False}),
        ((tmp_path / "x.gpkg", None, None), {"dry_run": True}),
    ]
