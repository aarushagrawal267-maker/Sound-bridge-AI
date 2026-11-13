"""WHAT SHOULD I LISTEN TO TODAY? - listening preferences mapped to audio-feature targets.

Each preference is translated into (a) hard constraints on audio features and (b) a
target point. These are approximate descriptors: "relaxed" here means low energy and
loudness as estimated by audio analysis, not a guaranteed mood or a measurement of how a
listener will feel.

Ranking uses the SoundBridge Encoder: the tracks closest to the target point form an
anchor set whose embedding centroid defines the "sound" of the preference; candidates are
then scored by closeness to the target, encoder similarity to that anchor and, if the
listener has seeds, similarity to their Musical DNA.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .diversity_ranking import greedy_rerank
from .feature_engineering import display_scale


@dataclass(frozen=True)
class ListeningMode:
    key: str
    title: str
    icon: str
    description: str
    constraints: dict
    target: dict


MODES: dict[str, ListeningMode] = {
    m.key: m
    for m in [
        ListeningMode("energetic", "Energetic", "bolt", "High energy, driving tempo, loud mix.",
                      {"energy": (0.75, 1.0), "tempo": (110, 250)},
                      {"energy": 0.9, "danceability": 0.62, "loudness": 0.85, "tempo": 0.55}),
        ListeningMode("relaxed", "Relaxed", "spa", "Low energy, softer dynamics, unhurried tempo.",
                      {"energy": (0.0, 0.45), "loudness": (-60, -8)},
                      {"energy": 0.25, "acousticness": 0.65, "tempo": 0.3, "loudness": 0.45, "speechiness": 0.04}),
        ListeningMode("acoustic", "Acoustic", "music_note", "Predominantly acoustic instrumentation.",
                      {"acousticness": (0.7, 1.0)},
                      {"acousticness": 0.92, "energy": 0.35, "instrumentalness": 0.1}),
        ListeningMode("instrumental", "Instrumental", "piano", "Little or no vocal content detected.",
                      {"instrumentalness": (0.6, 1.0), "speechiness": (0.0, 0.1)},
                      {"instrumentalness": 0.92, "speechiness": 0.04}),
        ListeningMode("dance", "Dance-oriented", "nightlife", "Strong, regular beat and high danceability.",
                      {"danceability": (0.7, 1.0), "energy": (0.55, 1.0)},
                      {"danceability": 0.85, "energy": 0.78, "tempo": 0.48, "valence": 0.6}),
        ListeningMode("experimental", "Experimental", "science",
                      "Tracks that sound atypical for their own genre label (far from their genre's "
                      "centroid in the encoder's embedding space).",
                      {}, {}),
        ListeningMode("genre_exploration", "Genre exploration", "explore",
                      "Genres you have not picked that the model places close to your taste.",
                      {}, {}),
    ]
}


def atypicality(engine) -> np.ndarray:
    """1 - cosine similarity between a track's embedding and its primary genre centroid."""
    gm = engine.genre_model
    if not gm.available:
        return np.zeros(len(engine.tracks), dtype=np.float32)
    gidx = engine.tracks["primary_genre"].map({g: i for i, g in enumerate(gm.genres)}).fillna(0).astype(int).to_numpy()
    return (1 - (engine.embeddings * gm.centroids[gidx]).sum(1)).astype(np.float32)


def constraint_mask(tracks: pd.DataFrame, constraints: dict) -> np.ndarray:
    mask = np.ones(len(tracks), dtype=bool)
    for f, (lo, hi) in constraints.items():
        if f in tracks.columns:
            v = tracks[f].to_numpy(dtype=float)
            mask &= (v >= lo) & (v <= hi)
    return mask


def recommend_for_mode(engine, mode_key: str, profile=None, k: int = 12, explore_genre: str | None = None,
                       atyp: np.ndarray | None = None) -> tuple[pd.DataFrame, dict]:
    mode = MODES[mode_key]
    t = engine.tracks
    mask = constraint_mask(t, mode.constraints)
    if profile is not None:
        mask[list(profile.excluded)] = False
    info: dict = {"mode": mode.title, "n_matching": int(mask.sum())}

    if mode_key == "experimental":
        a = atyp if atyp is not None else atypicality(engine)
        thr = float(np.quantile(a, 0.85))
        mask &= a >= thr
        info["n_matching"] = int(mask.sum())
        info["atypicality_threshold"] = thr
        closeness = (a - a.min()) / max(a.max() - a.min(), 1e-9)
    elif mode_key == "genre_exploration":
        gm = engine.genre_model
        if profile is not None:
            uv = profile.genre_vector
            opp = gm.opportunities(uv, n=4)
            targets = opp["genre"].tolist()
        else:
            targets = [explore_genre] if explore_genre else []
            if explore_genre:
                targets += gm.related_genres(explore_genre, 3)["genre"].tolist()
        info["genres"] = targets
        gmask = t["genre"].map(lambda g: bool(set(g) & set(targets))).to_numpy() if targets else mask
        mask &= gmask
        info["n_matching"] = int(mask.sum())
        closeness = np.ones(len(t), dtype=np.float32)
    else:
        feats = list(mode.target)
        disp = display_scale(t, feats).to_numpy()
        tgt = np.array([mode.target[f] for f in feats])
        dist = np.sqrt(((disp - tgt) ** 2).mean(1))
        closeness = 1 - dist / max(dist.max(), 1e-9)

    cand = np.where(mask)[0]
    if len(cand) == 0:
        return pd.DataFrame(), info
    # encoder anchor: centroid of the 40 candidates closest to the target
    anchor = cand[np.argsort(-closeness[cand])[:40]]
    centroid = engine.embeddings[anchor].mean(0)
    centroid /= max(np.linalg.norm(centroid), 1e-12)
    enc_sim = engine.embeddings[cand] @ centroid
    score = 0.5 * closeness[cand] + 0.5 * (enc_sim + 1) / 2
    if profile is not None:
        user_sim = engine.embeddings[cand] @ profile.centroid
        score = 0.5 * score + 0.5 * (user_sim + 1) / 2
        info["personalised"] = True
    top = cand[np.argsort(-score)[:400]]
    s_top = score[np.argsort(-score)[:400]]
    res = greedy_rerank(top, engine.embeddings[top], s_top, engine.artists[top], k, w_rel=0.8, w_div=0.2,
                        max_per_artist=1)
    rows = t.loc[res.idx]
    df = pd.DataFrame(
        {
            "idx": res.idx,
            "rank": np.arange(1, len(res.idx) + 1),
            "track_id": rows["track_id"].to_numpy(),
            "track_name": rows["track_name"].to_numpy(),
            "artists": rows["artists"].str.replace(";", ", ").to_numpy(),
            "primary_artist": rows["primary_artist"].to_numpy(),
            "album_name": rows["album_name"].to_numpy(),
            "genre": rows["genre_str"].to_numpy(),
            "popularity": rows["popularity"].to_numpy(),
            "relevance": res.relevance,
            "novelty": engine.novelty[res.idx],
            "strategy": f"mode:{mode_key}",
        }
    )
    df["discovery_score"] = df["novelty"]
    return df, info
