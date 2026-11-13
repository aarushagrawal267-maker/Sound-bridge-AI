"""Recommendation evaluation metrics.

Metric families
* Relevance  - mean similarity (proxy), precision@k / NDCG@k (only with real labels or
               explicitly labelled proxy labels)
* Diversity  - intra-list diversity, genre / artist / audio-feature diversity
* Discovery  - catalogue coverage, artist coverage, novelty, popularity distribution
* Repetition - repeated-artist %, duplicate-track %, mean pairwise similarity

Proxy metrics are named with a ``proxy_`` prefix so they are never mistaken for
user-validated relevance.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


def _norm(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


# ---------------- relevance ----------------
def mean_similarity(rec_vectors: np.ndarray, profile_vector: np.ndarray) -> float:
    if len(rec_vectors) == 0:
        return float("nan")
    p = profile_vector / max(np.linalg.norm(profile_vector), 1e-12)
    return float((_norm(rec_vectors) @ p).mean())


def precision_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    rel = set(relevant)
    rec = list(recommended)[:k]
    return float(sum(r in rel for r in rec) / k) if k else float("nan")


def recall_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return float("nan")
    return float(sum(r in rel for r in list(recommended)[:k]) / len(rel))


def ndcg_at_k(recommended: Sequence[int], gains: dict[int, float], k: int) -> float:
    """Normalised discounted cumulative gain with graded relevance `gains`."""
    rec = list(recommended)[:k]
    dcg = sum(gains.get(r, 0.0) / np.log2(i + 2) for i, r in enumerate(rec))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else float("nan")


# ---------------- diversity ----------------
def intra_list_diversity(rec_vectors: np.ndarray) -> float:
    """Mean pairwise cosine distance between recommended items."""
    n = len(rec_vectors)
    if n < 2:
        return 0.0
    v = _norm(rec_vectors)
    s = v @ v.T
    return float(1 - s[np.triu_indices(n, 1)].mean())


def mean_pairwise_similarity(rec_vectors: np.ndarray) -> float:
    return 1.0 - intra_list_diversity(rec_vectors) if len(rec_vectors) > 1 else 1.0


def unique_ratio(values: Sequence) -> float:
    values = list(values)
    return float(len(set(values)) / len(values)) if values else 0.0


def genre_diversity(genre_lists: Sequence[Sequence[str]]) -> dict[str, float]:
    flat = [g for lst in genre_lists for g in (lst[:1] if len(lst) else [])]
    if not flat:
        return {"genre_unique_ratio": 0.0, "genre_entropy": 0.0}
    p = pd.Series(flat).value_counts(normalize=True).to_numpy()
    return {"genre_unique_ratio": float(len(set(flat)) / len(flat)), "genre_entropy": float(-(p * np.log2(p)).sum())}


def audio_feature_diversity(raw_z: np.ndarray) -> float:
    """Mean per-feature standard deviation of z-scored audio features in the list."""
    return float(raw_z.std(0).mean()) if len(raw_z) > 1 else 0.0


# ---------------- discovery ----------------
def novelty_score(popularity: np.ndarray) -> float:
    return float((1 - np.asarray(popularity, dtype=float) / 100).mean())


def popularity_stats(popularity: np.ndarray, lesser_known: int = 35) -> dict[str, float]:
    p = np.asarray(popularity, dtype=float)
    return {
        "mean_popularity": float(p.mean()),
        "median_popularity": float(np.median(p)),
        "lesser_known_share": float((p <= lesser_known).mean()),
    }


def catalog_coverage(all_recommendations: Iterable[Iterable[int]], catalog_size: int) -> float:
    items = {i for rec in all_recommendations for i in rec}
    return float(len(items) / catalog_size)


def artist_coverage(all_rec_artists: Iterable[Iterable[str]], n_artists: int) -> float:
    items = {a for rec in all_rec_artists for a in rec}
    return float(len(items) / n_artists)


def gini(counts: Sequence[float]) -> float:
    """Gini coefficient of exposure concentration (0 = equal, 1 = concentrated)."""
    x = np.sort(np.asarray(counts, dtype=float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


# ---------------- repetition ----------------
def repeated_artist_pct(artists: Sequence[str]) -> float:
    a = list(artists)
    return float(1 - len(set(a)) / len(a)) if a else 0.0


def duplicate_track_pct(track_keys: Sequence[str]) -> float:
    k = list(track_keys)
    return float(1 - len(set(k)) / len(k)) if k else 0.0


# ---------------- list-level bundle ----------------
def list_metrics(
    rec_idx: Sequence[int],
    *,
    tracks: pd.DataFrame,
    embeddings: np.ndarray,
    raw_z: np.ndarray,
    profile_embedding: np.ndarray,
    raw_profile: np.ndarray,
    seed_genres: set[str] | None = None,
    lesser_known: int = 35,
    artist_exposure: np.ndarray | None = None,
) -> dict[str, float]:
    rec_idx = list(rec_idx)
    if not rec_idx:
        return {}
    sub = tracks.loc[rec_idx]
    out = {
        "relevance_learned": mean_similarity(embeddings[rec_idx], profile_embedding),
        "relevance_raw_features": mean_similarity(raw_z[rec_idx], raw_profile),
        "intra_list_diversity": intra_list_diversity(embeddings[rec_idx]),
        "artist_diversity": unique_ratio(sub["primary_artist"]),
        "audio_feature_diversity": audio_feature_diversity(raw_z[rec_idx]),
        "novelty": novelty_score(sub["popularity"]),
        "repeated_artist_pct": repeated_artist_pct(sub["primary_artist"]),
        "duplicate_track_pct": duplicate_track_pct(
            (sub["track_name"].str.lower().str.strip() + "|" + sub["primary_artist"].str.lower()).tolist()
        ),
        "mean_pairwise_similarity": mean_pairwise_similarity(embeddings[rec_idx]),
        **popularity_stats(sub["popularity"], lesser_known),
    }
    if artist_exposure is not None:
        ae = np.asarray(artist_exposure)[rec_idx]
        pop = sub["popularity"].to_numpy()
        # lesser-known artist share: artist exposure known (>0) and at or under the threshold
        out["lesser_known_share"] = float(((ae > 0) & (pop > 0) & (ae <= lesser_known)).mean())
        out["mean_artist_exposure"] = float(ae.mean())
    if "genre" in sub.columns:
        out.update(genre_diversity(sub["genre"].tolist()))
        if seed_genres:
            hits = [bool(set(g) & seed_genres) for g in sub["genre"]]
            out["proxy_genre_precision"] = float(np.mean(hits))
    return out
