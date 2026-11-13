"""Diversity-aware reranking (Maximal Marginal Relevance and its weighted extension).

At every step the candidate maximising

    w_rel * relevance + w_nov * novelty + w_div * (1 - max sim to already chosen)
    + w_gen * genre_term - w_rep * repetition - w_pen * penalty

is appended to the list. Classic MMR is the special case w_rel = lambda,
w_div = 1 - lambda and every other weight 0.

Hard constraints applied at every step:
* at most `max_per_artist` tracks from the same primary artist
* near-duplicates (embedding similarity >= `dup_threshold` to a chosen track by the
  same artist) are skipped
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RerankResult:
    idx: np.ndarray
    score: np.ndarray
    relevance: np.ndarray
    novelty: np.ndarray
    diversity: np.ndarray
    genre: np.ndarray
    repetition: np.ndarray


def greedy_rerank(
    idx: np.ndarray,
    vectors: np.ndarray,
    relevance: np.ndarray,
    artists: np.ndarray,
    k: int,
    *,
    novelty: np.ndarray | None = None,
    genre: np.ndarray | None = None,
    penalty: np.ndarray | None = None,
    albums: np.ndarray | None = None,
    w_rel: float = 1.0,
    w_nov: float = 0.0,
    w_div: float = 0.0,
    w_gen: float = 0.0,
    w_rep: float = 0.0,
    w_pen: float = 0.5,
    max_per_artist: int | None = 2,
    dup_threshold: float = 0.995,
    div_scale: float = 1.0,
) -> RerankResult:
    """`div_scale` rescales (1 - max similarity) so it is comparable to relevance; a
    natural choice is the mean pairwise distance inside the candidate pool."""
    n = len(idx)
    zeros = np.zeros(n, dtype=np.float32)
    novelty = zeros if novelty is None else np.asarray(novelty, dtype=np.float32)
    genre = zeros if genre is None else np.asarray(genre, dtype=np.float32)
    penalty = zeros if penalty is None else np.asarray(penalty, dtype=np.float32)
    v = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

    base = w_rel * relevance + w_nov * novelty + w_gen * genre - w_pen * penalty
    max_sim = np.full(n, -1.0, dtype=np.float32)  # similarity to closest chosen track
    artist_count: dict = {}
    album_count: dict = {}
    available = np.ones(n, dtype=bool)
    chosen, scores, divs, reps = [], [], [], []

    for _ in range(min(k, n)):
        div = np.where(max_sim < -0.5, 1.0, np.clip((1.0 - max_sim) / max(div_scale, 1e-6), 0.0, 1.0))
        rep = np.array([artist_count.get(a, 0) for a in artists], dtype=np.float32)
        if albums is not None:
            rep = rep + 0.5 * np.array([album_count.get(b, 0) for b in albums], dtype=np.float32)
        rep = np.minimum(rep / 2.0, 1.0)
        score = base + w_div * div - w_rep * rep
        mask = available.copy()
        if max_per_artist is not None:
            mask &= np.array([artist_count.get(a, 0) < max_per_artist for a in artists])
        if not mask.any():
            break
        score = np.where(mask, score, -np.inf)
        j = int(np.argmax(score))
        chosen.append(j)
        scores.append(float(score[j]))
        divs.append(float(div[j]))
        reps.append(float(rep[j]))
        available[j] = False
        artist_count[artists[j]] = artist_count.get(artists[j], 0) + 1
        if albums is not None:
            album_count[albums[j]] = album_count.get(albums[j], 0) + 1
        sim_j = v @ v[j]
        max_sim = np.maximum(max_sim, sim_j)
        # remove near-duplicates of the chosen track by the same artist
        available &= ~((sim_j >= dup_threshold) & (artists == artists[j]))

    c = np.array(chosen, dtype=np.int64)
    return RerankResult(
        idx=idx[c],
        score=np.array(scores, dtype=np.float32),
        relevance=relevance[c],
        novelty=novelty[c],
        diversity=np.array(divs, dtype=np.float32),
        genre=genre[c],
        repetition=np.array(reps, dtype=np.float32),
    )


def mmr(idx, vectors, relevance, artists, k, lam: float = 0.7, max_per_artist: int | None = 2,
        dup_threshold: float = 0.995) -> RerankResult:
    """Classic Maximal Marginal Relevance."""
    return greedy_rerank(
        idx, vectors, relevance, artists, k, w_rel=lam, w_div=1 - lam, max_per_artist=max_per_artist,
        dup_threshold=dup_threshold,
    )
