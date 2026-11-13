"""HIDDEN GEMS: relevant tracks by lesser-known artists *in this dataset*.

Pipeline
1. Score every catalogue track against the listener's profile (learned embeddings).
2. Keep only tracks in the top X% of the catalogue by match (relevance floor) - so a track
   never qualifies just for being obscure.
3. Keep only tracks whose artist exposure (max popularity of the artist's tracks in this
   dataset) and own popularity are under the level's threshold.
4. Rank by (1 - w) * relevance + w * novelty (+ a small exposure-gap bonus).
5. Allow one track per artist and add mild diversity.

Labels are deliberately neutral: "lesser-known in this dataset". Popularity measures
exposure on Spotify at collection time, not quality, and the dataset holds no metadata
on independence or career stage, so no such claims are made.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .diversity_ranking import greedy_rerank
from .musical_dna import TasteProfile

LEVEL_LABELS = {
    "familiar": "Familiar discoveries",
    "balanced": "Balanced discoveries",
    "deep": "Deep discoveries",
}


@dataclass
class HiddenGemsResult:
    tracks: pd.DataFrame
    level: str
    relevance_floor: float
    n_relevant: int            # tracks above the relevance floor
    n_lesser_known: int        # of those, how many are lesser-known
    n_excluded_irrelevant: int # lesser-known tracks excluded for not matching


def find_hidden_gems(engine, profile: TasteProfile, level: str = "balanced", k: int = 12,
                     exclude_artists: set[str] | None = None) -> HiddenGemsResult:
    cfg = engine.cfg["hidden_gems"]["levels"][level]
    rel_all = engine.content.score_all(profile)
    floor = float(np.quantile(rel_all, 1 - cfg["top_percent"] / 100))
    excluded = np.zeros(len(rel_all), dtype=bool)
    excluded[list(profile.excluded)] = True
    seed_artists = set(engine.artists[profile.seed_idx]) | set(exclude_artists or ())
    excluded |= np.isin(engine.artists, list(seed_artists))
    excluded |= np.isin(engine.song_key, list(set(engine.song_key[profile.seed_idx])))

    # popularity 0 is treated as missing exposure data, never as evidence of being lesser-known
    lesser = (
        engine.exposure_known
        & (engine.artist_exposure <= cfg["max_artist_exposure"])
        & (engine.popularity <= cfg["max_artist_exposure"])
    )
    relevant = (rel_all >= floor) & ~excluded
    cand = np.where(relevant & lesser)[0]
    n_excl = int(((rel_all < floor) & lesser & ~excluded).sum())

    if len(cand) == 0:
        return HiddenGemsResult(pd.DataFrame(), level, floor, int(relevant.sum()), 0, n_excl)

    rel = rel_all[cand]
    rel_n = (rel - rel.min()) / max(rel.max() - rel.min(), 1e-9)
    nw = cfg["novelty_weight"]
    nov = engine.novelty[cand] + 0.1 * engine.gap[cand]
    penalty = engine.genre_model.penalty(cand, profile.penalised_genres)
    res = greedy_rerank(cand, engine.embeddings[cand], rel_n, engine.artists[cand], k,
                        novelty=nov, penalty=penalty, w_rel=1 - nw, w_nov=nw, w_div=0.1,
                        max_per_artist=1)
    t = engine.tracks.loc[res.idx]
    seed_vecs = engine.embeddings[profile.seed_idx]
    best_seed = np.array(profile.seed_idx)[(engine.embeddings[res.idx] @ seed_vecs.T).argmax(1)]
    df = pd.DataFrame(
        {
            "idx": res.idx,
            "rank": np.arange(1, len(res.idx) + 1),
            "track_id": t["track_id"].to_numpy(),
            "track_name": t["track_name"].to_numpy(),
            "artists": t["artists"].str.replace(";", ", ").to_numpy(),
            "primary_artist": t["primary_artist"].to_numpy(),
            "album_name": t["album_name"].to_numpy(),
            "genre": t["genre_str"].to_numpy(),
            "popularity": t["popularity"].to_numpy(),
            "artist_exposure": engine.artist_exposure[res.idx],
            "relevance": rel_all[res.idx],
            "catalog_percentile": [(rel_all < r).mean() * 100 for r in rel_all[res.idx]],
            "novelty": engine.novelty[res.idx],
            "exposure_gap": engine.gap[res.idx],
            "predicted_popularity": engine.pred_popularity[res.idx],
            "score": res.score,
            "best_seed": best_seed,
            "strategy": f"hidden_gems:{level}",
        }
    )
    df["discovery_score"] = df["novelty"]
    return HiddenGemsResult(df, level, floor, int(relevant.sum()), int(len(cand)), n_excl)
