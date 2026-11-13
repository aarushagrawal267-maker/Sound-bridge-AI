import numpy as np
import pytest

from src.recommendation_metrics import (
    artist_coverage, catalog_coverage, duplicate_track_pct, genre_diversity, gini, intra_list_diversity,
    list_metrics, mean_similarity, ndcg_at_k, novelty_score, precision_at_k, recall_at_k, repeated_artist_pct,
)


def test_precision_recall():
    assert precision_at_k([1, 2, 3, 4], {2, 4, 9}, 4) == 0.5
    assert recall_at_k([1, 2, 3, 4], {2, 4, 9}, 4) == pytest.approx(2 / 3)


def test_ndcg_perfect_and_worst():
    gains = {1: 1.0, 2: 1.0}
    assert ndcg_at_k([1, 2, 3], gains, 3) == pytest.approx(1.0)
    assert ndcg_at_k([3, 4, 5], gains, 3) == 0.0
    assert ndcg_at_k([3, 1], gains, 2) == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)))


def test_intra_list_diversity():
    same = np.array([[1, 0], [1, 0]], dtype=float)
    orth = np.array([[1, 0], [0, 1]], dtype=float)
    assert intra_list_diversity(same) == pytest.approx(0.0)
    assert intra_list_diversity(orth) == pytest.approx(1.0)


def test_mean_similarity():
    assert mean_similarity(np.array([[1.0, 0.0]]), np.array([2.0, 0.0])) == pytest.approx(1.0)


def test_repetition_metrics():
    assert repeated_artist_pct(["a", "a", "b", "c"]) == 0.25
    assert duplicate_track_pct(["x", "x", "y"]) == pytest.approx(1 / 3)


def test_coverage_and_gini():
    assert catalog_coverage([[1, 2], [2, 3]], 10) == 0.3
    assert artist_coverage([["a"], ["a", "b"]], 4) == 0.5
    assert gini([5, 5, 5, 5]) == pytest.approx(0.0)
    assert gini([0, 0, 0, 10]) == pytest.approx(0.75)


def test_genre_diversity_and_novelty():
    g = genre_diversity([["rock"], ["rock"], ["jazz", "rock"]])
    assert g["genre_unique_ratio"] == pytest.approx(2 / 3)
    assert novelty_score([0, 100]) == 0.5


def test_list_metrics_on_real_recommendations(engine, seeds):
    from src.hybrid_recommender import Recommender

    p = engine.profile(seeds)
    recs = Recommender(engine).recommend(p, "hybrid", k=10)
    m = list_metrics(recs["idx"].tolist(), tracks=engine.tracks, embeddings=engine.embeddings, raw_z=engine.raw_z,
                     profile_embedding=p.centroid, raw_profile=engine.raw_profile(seeds),
                     seed_genres=engine.seed_genres(seeds), artist_exposure=engine.artist_exposure)
    assert 0 <= m["relevance_learned"] <= 1
    assert 0 <= m["proxy_genre_precision"] <= 1
    assert m["duplicate_track_pct"] == 0
    assert 0 <= m["lesser_known_share"] <= 1
