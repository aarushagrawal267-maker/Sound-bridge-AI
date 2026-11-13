import numpy as np
import pandas as pd
import pytest

from src.config import DNA_FEATURES
from src.data_loader import detect_columns, file_sha256, load_tracks, schema_report
from src.data_validation import invalid_row_mask, range_violations, validate


def test_detect_columns_maps_alternative_schemas():
    m = detect_columns(["id", "name", "artist_name", "track_genre", "Popularity", "danceability"])
    assert m["id"] == "track_id"
    assert m["name"] == "track_name"
    assert m["artist_name"] == "artists"
    assert m["track_genre"] == "genre"
    assert m["Popularity"] == "popularity"


def test_detect_columns_does_not_invent_fields():
    m = detect_columns(["track_name", "artists", "energy"])
    assert set(m.values()) == {"track_name", "artists", "energy"}


def test_load_real_dataset(tmp_path):
    from src.config import ROOT, load_config

    p = ROOT / load_config()["paths"]["raw_data"]
    if not p.exists():
        pytest.skip("raw dataset not downloaded")
    assert file_sha256(p) == load_config()["dataset"]["sha256"]
    df = load_tracks(p)
    assert len(df) == 114_000
    for col in ["track_id", "track_name", "artists", "genre", "popularity", *DNA_FEATURES]:
        assert col in df.columns
    rep = schema_report(df)
    assert rep["artist_id"] is False  # absent in this dataset and must not be fabricated
    assert df["explicit"].isin([0, 1]).all()


def test_loader_handles_custom_schema(tmp_path):
    p = tmp_path / "other.csv"
    pd.DataFrame({"name": ["a"], "artist": ["b"], "energy": [0.5], "Unnamed: 0": [0]}).to_csv(p, index=False)
    df = load_tracks(p)
    assert list(df.columns) == ["track_name", "artists", "energy"]


def test_loader_requires_core_columns(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"energy": [0.5]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_tracks(p)


def test_validation_flags_invalid_values(raw_frame):
    df = raw_frame.rename(columns={"track_genre": "genre"})
    df.loc[0, "popularity"] = 150
    df.loc[1, "energy"] = 1.4
    rv = range_violations(df)
    assert rv["popularity"].sum() == 1
    assert rv["energy"].sum() == 1
    assert rv["tempo"].sum() == 1  # tempo 0 is an analysis failure
    mask = invalid_row_mask(df, DNA_FEATURES)
    assert mask.sum() == 4  # pop 150, energy 1.4, tempo 0, missing artist


def test_validation_report_counts(raw_frame):
    df = raw_frame.rename(columns={"track_genre": "genre"})
    rep = validate(df, DNA_FEATURES)
    assert rep.n_rows == 6
    assert rep.duplicate_track_ids == 1
    assert rep.missing_artist == 1
    assert any("tempo = 0" in n for n in rep.notes)


def test_non_finite_values_are_invalid(raw_frame):
    df = raw_frame.rename(columns={"track_genre": "genre"})
    df.loc[2, "loudness"] = np.inf
    assert invalid_row_mask(df, DNA_FEATURES).iloc[2]
