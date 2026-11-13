"""Musical DNA: a listener profile built from selected seed tracks.

The profile has two layers:

* an interpretable layer - summary statistics of the seeds' audio features
  (mean, spread, tempo range, genre mix, descriptive labels)
* a learned layer - the SoundBridge Encoder's embedding centroid of the seeds, plus the
  individual seed embeddings (so mixed tastes are not averaged into a meaningless middle)

Audio features are computational descriptors from Spotify's audio analysis. They do not
measure how a person experiences a song; valence in particular is "musical
positiveness" as estimated by an algorithm, not a measurement of emotion.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import DNA_FEATURES, KEY_NAMES, SHORT_LABELS
from .feature_engineering import display_scale


@dataclass
class TasteProfile:
    seed_idx: list[int]
    centroid: np.ndarray                  # normalised embedding centroid (after feedback)
    seed_embeddings: np.ndarray           # (n_seeds, d)
    liked_idx: list[int] = field(default_factory=list)
    disliked_idx: list[int] = field(default_factory=list)
    penalised_genres: list[str] = field(default_factory=list)
    genre_vector: np.ndarray | None = None

    @property
    def n_seeds(self) -> int:
        return len(self.seed_idx)

    @property
    def excluded(self) -> set[int]:
        return set(self.seed_idx) | set(self.liked_idx) | set(self.disliked_idx)


def _normalise(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def build_profile(
    embeddings: np.ndarray,
    seed_idx: list[int],
    liked_idx: list[int] | None = None,
    disliked_idx: list[int] | None = None,
    learning_rate: float = 0.25,
    genre_vector: np.ndarray | None = None,
    penalised_genres: list[str] | None = None,
) -> TasteProfile:
    """Rocchio-style profile: seeds, pulled toward liked and pushed from disliked tracks."""
    if not seed_idx:
        raise ValueError("At least one seed track is required")
    liked_idx = [i for i in (liked_idx or []) if i not in seed_idx]
    disliked_idx = list(disliked_idx or [])
    seeds = embeddings[seed_idx]
    c = seeds.mean(0)
    if liked_idx:
        c = c + learning_rate * embeddings[liked_idx].mean(0)
    if disliked_idx:
        c = c - learning_rate * embeddings[disliked_idx].mean(0)
    return TasteProfile(
        seed_idx=list(seed_idx),
        centroid=_normalise(c),
        seed_embeddings=_normalise(np.vstack([seeds, embeddings[liked_idx]]) if liked_idx else seeds),
        liked_idx=liked_idx,
        disliked_idx=disliked_idx,
        penalised_genres=list(penalised_genres or []),
        genre_vector=genre_vector,
    )


def taste_cohesion(profile: TasteProfile) -> float:
    """Mean pairwise cosine similarity of seeds (1 = very consistent taste)."""
    e = profile.seed_embeddings
    if len(e) < 2:
        return 1.0
    s = e @ e.T
    return float(s[np.triu_indices(len(e), 1)].mean())


def feature_summary(tracks: pd.DataFrame, seed_idx: list[int]) -> pd.DataFrame:
    """Mean, spread and range of each available DNA feature across the seeds."""
    feats = [f for f in DNA_FEATURES if f in tracks.columns]
    s = tracks.loc[seed_idx, feats].astype(float)
    cat = tracks[feats].astype(float)
    pct = [(cat[f] <= s[f].mean()).mean() * 100 for f in feats]
    return pd.DataFrame(
        {
            "feature": feats,
            "label": [SHORT_LABELS[f] for f in feats],
            "mean": s.mean().values,
            "std": s.std(ddof=0).values,
            "min": s.min().values,
            "max": s.max().values,
            "catalog_mean": cat.mean().values,
            "catalog_percentile": pct,
            "display_mean": display_scale(s).mean().values,
            "display_catalog": display_scale(cat.sample(min(len(cat), 20000), random_state=0)).mean().values,
        }
    )


def _level(p: float) -> str:
    if p >= 80:
        return "very high"
    if p >= 60:
        return "high"
    if p >= 40:
        return "moderate"
    if p >= 20:
        return "low"
    return "very low"


def describe_profile(tracks: pd.DataFrame, seed_idx: list[int]) -> dict[str, object]:
    """Human-readable characteristics derived from the actual seed features."""
    summ = feature_summary(tracks, seed_idx).set_index("feature")
    seeds = tracks.loc[seed_idx]
    out: dict[str, object] = {}
    if "tempo" in summ.index:
        t = seeds["tempo"].astype(float)
        out["tempo_range"] = (float(t.min()), float(t.max()))
        out["tempo_mean"] = float(t.mean())
    if {"energy", "valence"} <= set(summ.index):
        e, v = summ.loc["energy", "mean"], summ.loc["valence", "mean"]
        quad = (
            "high-energy, bright-sounding" if e >= 0.6 and v >= 0.5 else
            "high-energy, darker-sounding" if e >= 0.6 else
            "calm, bright-sounding" if v >= 0.5 else
            "calm, darker-sounding"
        )
        out["energy_mood"] = quad
    traits = []
    for f in ("danceability", "energy", "acousticness", "instrumentalness", "speechiness", "valence"):
        if f in summ.index:
            traits.append((f, summ.loc[f, "catalog_percentile"]))
    traits.sort(key=lambda x: -abs(x[1] - 50))
    out["traits"] = [
        f"{_level(p)} {f} — higher than {p:.0f}% of the catalogue" for f, p in traits[:4]
    ]
    if "genre" in tracks.columns:
        g = pd.Series([x for lst in seeds["genre"] for x in lst]).value_counts()
        out["genres"] = g
    if "mode" in seeds.columns:
        out["major_share"] = float((seeds["mode"] == 1).mean())
    if "key" in seeds.columns:
        k = seeds["key"].dropna().astype(int)
        k = k[k >= 0]
        out["common_key"] = KEY_NAMES[k.mode().iloc[0]] if len(k) else None
    return out
