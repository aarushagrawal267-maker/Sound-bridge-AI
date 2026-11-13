import numpy as np
import pytest

from src.musical_dna import build_profile, describe_profile, feature_summary, taste_cohesion


def _emb(rng, n=50, d=8):
    e = rng.normal(size=(n, d))
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def test_profile_is_normalised(rng):
    e = _emb(rng)
    p = build_profile(e, [0, 1, 2])
    assert np.isclose(np.linalg.norm(p.centroid), 1.0)
    assert p.n_seeds == 3
    assert p.excluded == {0, 1, 2}


def test_profile_requires_seed(rng):
    with pytest.raises(ValueError):
        build_profile(_emb(rng), [])


def test_rocchio_moves_towards_liked_and_away_from_disliked(rng):
    e = _emb(rng)
    base = build_profile(e, [0, 1])
    liked = build_profile(e, [0, 1], liked_idx=[5])
    disliked = build_profile(e, [0, 1], disliked_idx=[5])
    assert liked.centroid @ e[5] > base.centroid @ e[5] > disliked.centroid @ e[5]
    assert 5 in liked.excluded and 5 in disliked.excluded


def test_cohesion_single_seed(rng):
    assert taste_cohesion(build_profile(_emb(rng), [3])) == 1.0


def test_feature_summary_matches_seed_means(engine, seeds):
    s = feature_summary(engine.tracks, seeds).set_index("feature")
    expected = engine.tracks.loc[seeds, "energy"].mean()
    assert np.isclose(s.loc["energy", "mean"], expected)
    assert (s["display_mean"].between(0, 1)).all()


def test_profile_changes_when_seeds_change(engine, seeds):
    a = feature_summary(engine.tracks, seeds[:1])["mean"].to_numpy()
    b = feature_summary(engine.tracks, seeds)["mean"].to_numpy()
    assert not np.allclose(a, b)


def test_describe_profile(engine, seeds):
    d = describe_profile(engine.tracks, seeds)
    t = engine.tracks.loc[seeds, "tempo"]
    assert d["tempo_range"] == (t.min(), t.max())
    assert len(d["traits"]) == 4
    assert d["genres"].sum() == sum(len(g) for g in engine.tracks.loc[seeds, "genre"])
