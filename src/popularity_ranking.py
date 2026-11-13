"""Popularity-aware ranking.

Popularity is treated strictly as an *exposure proxy* within this dataset: it reflects
how much a track has been played recently relative to others on Spotify, not artistic
quality. Discovery-oriented ranking therefore never simply sorts by lowest popularity;
candidates must first clear a relevance floor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def novelty(popularity: np.ndarray) -> np.ndarray:
    """Exposure-based novelty in [0, 1]: 1 - popularity/100."""
    return 1.0 - np.clip(np.asarray(popularity, dtype=np.float32), 0, 100) / 100.0


def artist_exposure(primary_artist: pd.Series, popularity: pd.Series) -> np.ndarray:
    """Highest track popularity of each track's primary artist *within this dataset*.

    Track popularity alone is misleading: an alternate release of a famous song can have
    popularity 0 while its artist is very widely heard. Artist exposure guards against
    labelling such tracks as coming from lesser-known artists.
    """
    return primary_artist.map(popularity.groupby(primary_artist).max()).to_numpy(dtype=np.float32)


def blended_novelty(track_popularity: np.ndarray, artist_exp: np.ndarray, artist_weight: float = 0.7) -> np.ndarray:
    """Novelty that mostly reflects the artist's exposure, partly the track's.

    Popularity 0 is treated as *missing* rather than as "unknown to everyone": in this
    dataset it mostly marks alternate releases or tracks whose plays are credited
    elsewhere. Rules:
    * track popularity 0 but artist exposure > 0 -> use the artist exposure alone
    * artist exposure 0 (no reliable signal at all) -> neutral novelty 0.5
    """
    tp = np.asarray(track_popularity, dtype=np.float32)
    ae = np.asarray(artist_exp, dtype=np.float32)
    tp_eff = np.where(tp > 0, tp, ae)
    exp = (1 - artist_weight) * tp_eff + artist_weight * ae
    return np.where(ae > 0, novelty(exp), 0.5).astype(np.float32)


def exposure_known(track_popularity: np.ndarray, artist_exp: np.ndarray) -> np.ndarray:
    """True where popularity gives a usable exposure signal (both values above 0)."""
    return (np.asarray(track_popularity) > 0) & (np.asarray(artist_exp) > 0)


def exposure_gap(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Predicted-from-sound popularity minus actual popularity, scaled to [0, 1].

    A large gap means the track sounds like music that is usually more exposed in this
    dataset. It is a heuristic discovery signal, not evidence of quality.
    """
    return np.clip((np.asarray(predicted) - np.asarray(actual)) / 100.0, 0, 1).astype(np.float32)


def popularity_ranked(idx: np.ndarray, relevance: np.ndarray, popularity: np.ndarray, pool: int = 100, k: int = 10):
    """Baseline: take the `pool` most relevant candidates and sort them by popularity."""
    top = np.argsort(-relevance)[:pool]
    order = top[np.argsort(-popularity[idx[top]], kind="stable")]
    return idx[order][:k], relevance[order][:k]


def discovery_ranked(
    idx: np.ndarray,
    relevance: np.ndarray,
    popularity: np.ndarray,
    min_relevance: float,
    novelty_weight: float = 0.3,
    max_popularity: float | None = None,
    gap: np.ndarray | None = None,
    gap_weight: float = 0.0,
):
    """Score = (1 - w) * relevance + w * novelty (+ optional exposure-gap bonus).

    Candidates below `min_relevance` (or above `max_popularity`) are excluded first,
    which is what prevents the least popular tracks from winning by default.
    """
    pop = popularity[idx]
    keep = relevance >= min_relevance
    if max_popularity is not None:
        keep &= pop <= max_popularity
    score = (1 - novelty_weight) * relevance + novelty_weight * novelty(pop)
    if gap is not None and gap_weight:
        score = score + gap_weight * gap
    score = np.where(keep, score, -np.inf)
    order = np.argsort(-score)
    order = order[np.isfinite(score[order])]
    return idx[order], score[order]
