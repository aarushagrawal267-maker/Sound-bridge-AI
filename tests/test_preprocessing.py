import numpy as np

from src.feature_engineering import FeaturePipeline, display_scale, genre_matrix
from src.preprocessing import base_title, clean_tracks, normalise_title, split_artists


def _clean(raw_frame):
    return clean_tracks(raw_frame.rename(columns={"track_genre": "genre"}))


def test_clean_merges_genres_and_versions(raw_frame):
    df, log = _clean(raw_frame)
    # D (tempo 0) and E (no artist) excluded; A rows merged; B is a live version of A -> merged
    assert set(df["track_id"]) == {"A", "C"}
    a = df[df.track_id == "A"].iloc[0]
    assert a["genre"] == ["rock", "indie"]
    assert a["n_versions"] == 2
    assert a["popularity"] == 40  # most popular version kept
    steps = log.to_frame()
    assert steps["removed"].sum() == len(raw_frame) - len(df)


def test_primary_artist_and_idx(raw_frame):
    df, _ = _clean(raw_frame)
    c = df[df.track_id == "C"].iloc[0]
    assert c["primary_artist"] == "Singer Y"
    assert c["artist_list"] == ["Singer Y", "Band X"]
    assert list(df["idx"]) == list(range(len(df)))


def test_title_helpers():
    assert normalise_title("Song - 2011 Remaster") == "song"
    assert normalise_title("Alive") == "alive"
    assert base_title("The Night We Met (feat. Phoebe Bridgers)") == "the night we met"
    assert split_artists("A; B;C") == ["A", "B", "C"]


def test_feature_pipeline_outputs(raw_frame):
    df, _ = _clean(raw_frame)
    pipe = FeaturePipeline.fit(df)
    X = pipe.model_inputs(df)
    assert X.shape == (len(df), len(pipe.input_names))
    assert np.isfinite(X).all()
    R = pipe.raw_matrix(df)
    assert R.shape == (len(df), 9)


def test_pipeline_tolerates_missing_feature(raw_frame):
    df, _ = _clean(raw_frame.drop(columns=["liveness", "key"]))
    pipe = FeaturePipeline.fit(df)
    assert "liveness" not in pipe.dna_features
    assert "key_sin" not in pipe.input_names
    assert np.isfinite(pipe.model_inputs(df)).all()


def test_identifiers_not_in_similarity_features(raw_frame):
    df, _ = _clean(raw_frame)
    pipe = FeaturePipeline.fit(df)
    for banned in ("track_id", "popularity", "idx", "genre", "artists"):
        assert banned not in pipe.input_names


def test_display_scale_bounds(raw_frame):
    df, _ = _clean(raw_frame)
    d = display_scale(df)
    assert ((d >= 0) & (d <= 1)).all().all()


def test_genre_matrix(raw_frame):
    df, _ = _clean(raw_frame)
    m, genres = genre_matrix(df)
    assert m.shape == (len(df), len(genres))
    assert m.sum() == sum(len(g) for g in df["genre"])
