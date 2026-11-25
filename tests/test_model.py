import joblib
import numpy as np
import pytest
import torch

from src.model import SoundBridgeEncoder, encode, similarity_attribution
from src.trainer import (
    _PositiveSampler,
    artist_split,
    cluster_bootstrap_ci,
    retrieval_metrics,
    retrieval_per_query,
    train_encoder,
)


def test_forward_shapes_and_norm():
    m = SoundBridgeEncoder(in_dim=15, n_genres=7, hidden=32, emb_dim=8)
    z, logits, pop = m(torch.randn(5, 15))
    assert z.shape == (5, 8) and logits.shape == (5, 7) and pop.shape == (5,)
    assert torch.allclose(z.norm(dim=1), torch.ones(5), atol=1e-5)
    assert ((pop >= 0) & (pop <= 100)).all()


def test_save_load_roundtrip(tmp_path):
    m = SoundBridgeEncoder(in_dim=6, n_genres=3, hidden=16, emb_dim=4)
    m.eval()
    x = np.random.default_rng(0).normal(size=(10, 6)).astype(np.float32)
    m.save(tmp_path / "m.pt", extra={"genres": ["a", "b", "c"]})
    m2, extra = SoundBridgeEncoder.load(tmp_path / "m.pt")
    assert extra["genres"] == ["a", "b", "c"]
    assert np.allclose(encode(m, x)[0], encode(m2, x)[0], atol=1e-6)


def test_attribution_completeness():
    torch.manual_seed(0)
    m = SoundBridgeEncoder(in_dim=6, n_genres=3, hidden=16, emb_dim=4).eval()
    x = np.random.default_rng(1).normal(size=6).astype(np.float32)
    target = np.random.default_rng(2).normal(size=4).astype(np.float32)
    attr, delta = similarity_attribution(m, x, target, steps=128)
    assert attr.sum() == pytest.approx(delta, abs=0.02)


def test_artist_split_disjoint():
    codes = np.repeat(np.arange(50), 4)
    tr, te = artist_split(codes, 0.2, seed=0)
    assert not set(codes[tr]) & set(codes[te])
    assert len(tr) + len(te) == len(codes)


def test_positive_sampler_same_artist():
    codes = np.array([0, 0, 1, 2, 2, 2])
    s = _PositiveSampler(codes, np.random.default_rng(0))
    pos, has = s.sample(np.arange(6))
    assert (codes[pos] == codes).all()
    assert not has[2]  # artist 1 has a single track


def test_training_reduces_loss():
    rng = np.random.default_rng(0)
    n, d, g = 600, 6, 4
    artists = rng.integers(0, 60, n)
    centers = rng.normal(size=(60, d))
    X = (centers[artists] + 0.3 * rng.normal(size=(n, d))).astype(np.float32)
    genres = np.eye(g, dtype=np.float32)[artists % g]
    pop = (artists % 10 * 10 + 5).astype(np.float32)
    res = train_encoder(X, genres, pop, artists, np.arange(n), hidden=32, emb_dim=8, epochs=8, batch_size=128,
                        lr=5e-3, verbose=False)
    assert res.history[-1]["loss"] < res.history[0]["loss"]
    emb = encode(res.model, X)[0]
    m = retrieval_metrics(emb, np.arange(n), artists, genres, k=5, max_queries=200)
    raw = retrieval_metrics(np.random.default_rng(1).normal(size=(n, 8)), np.arange(n), artists, genres, k=5, max_queries=200)
    assert m["artist_precision@5"] > raw["artist_precision@5"]


def test_trained_artifacts_consistent(engine):
    """The persisted model reproduces the stored catalogue embeddings."""
    from src.config import path

    pipe = joblib.load(path("models_dir") / "feature_pipeline.joblib")
    X = pipe.model_inputs(engine.tracks.iloc[:200])
    emb, probs, pop = encode(engine.model, X)
    assert np.allclose(emb, engine.embeddings[:200], atol=1e-4)
    assert np.allclose(probs.sum(1), 1, atol=1e-3)
    assert len(engine.model_meta["genres"]) == engine.genre_probs.shape[1]


def test_retrieval_per_query_matches_aggregate():
    rng = np.random.default_rng(3)
    n, d = 600, 6
    artists = rng.integers(0, 80, n)
    X = (rng.normal(size=(80, d))[artists] + 0.5 * rng.normal(size=(n, d))).astype(np.float32)
    genres = np.eye(5, dtype=np.float32)[artists % 5]
    idx = np.arange(100, n)
    r = retrieval_per_query(X, idx, artists, genres, k=5, max_queries=300, seed=4)
    m = retrieval_metrics(X, idx, artists, genres, k=5, max_queries=300, seed=4)
    assert r["same_artist"].shape == (300, 5)
    assert m["artist_precision@5"] == pytest.approx(r["same_artist"].mean())
    assert m["artist_hit@5"] == pytest.approx(r["same_artist"].any(1).mean())
    assert m["genre_agreement@5"] == pytest.approx(r["genre_overlap"].mean())
    assert m["genre_agreement_chance@5"] == pytest.approx(r["chance_genre_overlap"].mean())


def test_cluster_bootstrap_ci_covers_mean_and_respects_clusters():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(50), 20)
    # values identical within a cluster: the effective sample size is 50, not 1,000
    x = np.repeat(rng.normal(size=50), 20)
    y = x + 1.0
    ci = cluster_bootstrap_ci({"x": x, "y": y}, clusters, statistic=lambda m: {**m, "d": m["y"] - m["x"]},
                              n_boot=500, seed=1)
    lo, hi = ci["x"]
    assert lo < x.mean() < hi
    assert ci["d"] == pytest.approx([1.0, 1.0])
    naive_half = 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    assert (hi - lo) / 2 > 2 * naive_half  # clustering widens the interval


def test_committed_confidence_report_matches_evaluation():
    """outputs/reports/evaluation_confidence.json was computed from the committed evaluation."""
    import json

    from src.config import path

    reports = path("outputs_dir") / "reports"
    ev = json.loads((reports / "model_evaluation.json").read_text())["retrieval_held_out_artists"]
    conf = json.loads((reports / "evaluation_confidence.json").read_text())["held_out_retrieval"]
    names = {"encoder": "SoundBridge Encoder (learned, 32-d)", "raw": "Raw audio features (z-score, 9-d)"}
    for key, name in names.items():
        for metric, field in (("precision", "artist_precision@10"), ("hit", "artist_hit@10"),
                              ("genre", "genre_agreement@10")):
            row = conf["metrics"][f"{key}_{metric}"]
            assert row["value"] == pytest.approx(ev[name][field], abs=1e-4)
            assert row["ci95"][0] <= row["value"] <= row["ci95"][1]
    assert conf["n_queries"] == ev[names["encoder"]]["n_queries"]
