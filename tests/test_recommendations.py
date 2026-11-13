import numpy as np
import pytest

from src.hidden_gems import find_hidden_gems
from src.hybrid_recommender import STRATEGIES, RankingWeights, Recommender, weights_from_preferences
from src.listening_modes import MODES, recommend_for_mode


@pytest.fixture(scope="module")
def rec(engine):
    return Recommender(engine)


@pytest.fixture(scope="module")
def profile(engine, seeds):
    return engine.profile(seeds)


@pytest.mark.parametrize("strategy", list(STRATEGIES))
def test_every_strategy_returns_valid_list(engine, rec, profile, seeds, strategy):
    df = rec.recommend(profile, strategy, k=10)
    assert len(df) == 10
    assert df["idx"].isin(engine.tracks["idx"]).all()           # tracks exist in the dataset
    assert not set(df["idx"]) & set(seeds)                       # seeds excluded
    assert df["idx"].is_unique                                   # no duplicate recommendations
    assert not set(engine.song_key[df["idx"]]) & set(engine.song_key[seeds])  # no alt versions of seeds
    assert df["popularity"].between(0, 100).all()
    assert df["relevance"].between(-1, 1).all()
    assert df["discovery_score"].between(0, 1).all()


def test_content_similarity_is_cosine(engine, rec, profile):
    df = rec.recommend(profile, "content", k=5)
    i = int(df.iloc[0]["idx"])
    v = engine.embeddings[i]
    expected = 0.5 * v @ profile.centroid + 0.5 * (profile.seed_embeddings @ v).max()
    assert df.iloc[0]["relevance"] == pytest.approx(expected, abs=1e-5)
    assert df["relevance"].is_monotonic_decreasing


def test_artist_repetition_controlled(rec, profile):
    w = RankingWeights(max_per_artist=1)
    for s in ("hybrid", "diversity_mmr", "popularity_aware"):
        df = rec.recommend(profile, s, w, k=15)
        assert df["primary_artist"].is_unique, s


def test_popularity_ranked_is_more_popular(rec, profile):
    c = rec.recommend(profile, "content", k=10)
    p = rec.recommend(profile, "popularity_ranked", k=10)
    assert p["popularity"].mean() > c["popularity"].mean()


def test_discovery_preference_changes_ranking(rec, profile):
    low = rec.recommend(profile, "hybrid", weights_from_preferences(0.6, 0.0, 0.1, 0.5), k=12)
    high = rec.recommend(profile, "hybrid", weights_from_preferences(0.6, 1.0, 0.1, 0.5), k=12)
    assert list(low["idx"]) != list(high["idx"])
    assert high["novelty"].mean() > low["novelty"].mean()


def test_genre_exploration_changes_genres(rec, profile):
    stay = rec.recommend(profile, "hybrid", weights_from_preferences(0.6, 0.3, 0.0, 0.5), k=12)
    go = rec.recommend(profile, "hybrid", weights_from_preferences(0.6, 0.3, 1.0, 0.5), k=12)
    assert stay["genre_affinity"].mean() > go["genre_affinity"].mean()


def test_diversity_preference_increases_spread(engine, rec, profile):
    from src.recommendation_metrics import intra_list_diversity

    a = rec.recommend(profile, "hybrid", weights_from_preferences(0.6, 0.3, 0.1, 0.0), k=12)
    b = rec.recommend(profile, "hybrid", weights_from_preferences(0.6, 0.3, 0.1, 1.0), k=12)
    assert intra_list_diversity(engine.embeddings[b["idx"]]) > intra_list_diversity(engine.embeddings[a["idx"]])


def test_raw_baseline_uses_raw_space(engine, rec, profile, seeds):
    df = rec.recommend(profile, "content_raw", k=5)
    rp = engine.raw_profile(seeds)
    v = engine.raw_z[int(df.iloc[0]["idx"])]
    assert df.iloc[0]["raw_similarity"] == pytest.approx(v @ rp / (np.linalg.norm(v) * np.linalg.norm(rp)), abs=1e-5)


def test_unknown_strategy_raises(rec, profile):
    with pytest.raises(ValueError):
        rec.recommend(profile, "magic")


@pytest.mark.parametrize("level", ["familiar", "balanced", "deep"])
def test_hidden_gems(engine, profile, seeds, level):
    res = find_hidden_gems(engine, profile, level, k=10)
    cfg = engine.cfg["hidden_gems"]["levels"][level]
    g = res.tracks
    assert len(g) > 0
    assert (g["relevance"] >= res.relevance_floor - 1e-6).all()         # relevance first
    assert (g["artist_exposure"] <= cfg["max_artist_exposure"]).all()
    assert (g["popularity"] > 0).all()                                   # popularity 0 is not "lesser-known"
    assert g["primary_artist"].is_unique                                  # one per artist
    assert not set(g["primary_artist"]) & set(engine.artists[seeds])     # not the seed artists


def test_hidden_gems_not_simply_least_popular(engine, profile):
    g = find_hidden_gems(engine, profile, "deep", k=10).tracks
    least = engine.tracks[engine.exposure_known].nsmallest(10, "popularity")
    assert set(g["idx"]) != set(least["idx"])


@pytest.mark.parametrize("mode", [m for m in MODES if m != "genre_exploration"])
def test_listening_modes_respect_constraints(engine, mode):
    df, info = recommend_for_mode(engine, mode, k=8)
    assert len(df) == 8
    t = engine.tracks.loc[df["idx"]]
    for f, (lo, hi) in MODES[mode].constraints.items():
        assert t[f].between(lo, hi).all(), f


def test_genre_exploration_mode(engine, profile):
    df, info = recommend_for_mode(engine, "genre_exploration", profile=profile, k=6)
    assert len(df) > 0
    assert all(set(engine.tracks.at[i, "genre"]) & set(info["genres"]) for i in df["idx"])


def test_explanations_use_real_values(engine, rec, profile, seeds):
    from src.explainability import build_explanation

    i = int(rec.recommend(profile, "hybrid", k=3).iloc[0]["idx"])
    exp = build_explanation(tracks=engine.tracks, seed_idx=seeds, track_idx=i, catalog_std=engine.catalog_std,
                            raw_track=engine.raw_z[i], raw_profile=engine.raw_profile(seeds),
                            raw_names=engine.dna_features, relevance=0.8, seed_genres=engine.seed_genres(seeds),
                            attribution=engine.attribution(i, profile.centroid))
    assert exp.reasons
    comp = exp.comparison.set_index("feature")
    assert comp.loc["energy", "this_track"] == pytest.approx(engine.tracks.at[i, "energy"])
    # exact decomposition: contributions sum to the raw cosine similarity
    rz, rp = engine.raw_z[i], engine.raw_profile(seeds)
    assert exp.contributions["contribution"].sum() == pytest.approx(rz @ rp / (np.linalg.norm(rz) * np.linalg.norm(rp)), abs=1e-5)


def test_missing_genre_data_does_not_crash(engine):
    from src.genre_recommender import GenreModel

    gm = GenreModel(np.zeros((len(engine.embeddings), 0)), [], engine.embeddings)
    assert not gm.available
    assert gm.affinity(np.arange(3), gm.user_vector([0])).tolist() == [0, 0, 0]
