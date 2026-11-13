"""Feature preparation.

Four separate feature sets are produced so that identifiers and metadata never leak
into musical similarity:

1. ``similarity``  - audio features only (encoder inputs / raw baseline space)
2. ``artist_genre`` - multi-hot genre matrix and artist codes (never used as encoder input)
3. ``popularity``   - popularity and derived tiers (used only for discovery ranking)
4. ``diversity``    - artist, album, genre and sound-cluster identifiers for repetition
                       and diversity metrics
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, QuantileTransformer, StandardScaler

from .config import DISPLAY_RANGES, DNA_FEATURES, UNIT_FEATURES


@dataclass
class FeaturePipeline:
    """Fitted transformations from raw audio features to model / baseline spaces."""

    dna_features: list[str]
    structural: list[str]
    quantile: QuantileTransformer
    raw_scaler: StandardScaler
    duration_mean: float = 0.0
    duration_std: float = 1.0
    input_names: list[str] = field(default_factory=list)

    @classmethod
    def fit(cls, df: pd.DataFrame, seed: int = 42) -> "FeaturePipeline":
        dna = [f for f in DNA_FEATURES if f in df.columns]
        structural = [c for c in ("duration_ms", "mode", "key", "time_signature", "explicit") if c in df.columns]
        qt = QuantileTransformer(
            n_quantiles=1000, output_distribution="normal", subsample=200_000, random_state=seed
        ).fit(df[dna].to_numpy(dtype=float))
        sc = StandardScaler().fit(df[dna].to_numpy(dtype=float))
        pipe = cls(dna, structural, qt, sc)
        if "duration_ms" in structural:
            logd = np.log(df["duration_ms"].to_numpy(dtype=float))
            pipe.duration_mean, pipe.duration_std = float(logd.mean()), float(logd.std() + 1e-9)
        pipe.input_names = pipe._input_names()
        return pipe

    def _input_names(self) -> list[str]:
        names = list(self.dna_features)
        for c in self.structural:
            names += ["key_sin", "key_cos"] if c == "key" else [c]
        return names

    def model_inputs(self, df: pd.DataFrame) -> np.ndarray:
        """Encoder inputs: quantile-normalised audio + encoded structural features."""
        q = self.quantile.transform(df[self.dna_features].to_numpy(dtype=float))
        q = np.clip(q, -5.2, 5.2)
        cols = [q]
        for c in self.structural:
            v = df[c].to_numpy(dtype=float)
            if c == "duration_ms":
                cols.append(((np.log(np.maximum(v, 1.0)) - self.duration_mean) / self.duration_std)[:, None])
            elif c == "key":
                ang = 2 * np.pi * np.nan_to_num(v, nan=0.0) / 12.0
                has_key = (v >= 0).astype(float)
                cols.append((np.sin(ang) * has_key)[:, None])
                cols.append((np.cos(ang) * has_key)[:, None])
            elif c == "mode":
                cols.append((2 * np.nan_to_num(v, nan=0.5) - 1)[:, None])
            elif c == "time_signature":
                cols.append(((np.nan_to_num(v, nan=4.0) - 4.0) / 1.0)[:, None])
            elif c == "explicit":
                cols.append(np.nan_to_num(v, nan=0.0)[:, None])
        return np.hstack(cols).astype(np.float32)

    def raw_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Classic baseline space: z-scored raw audio features."""
        return self.raw_scaler.transform(df[self.dna_features].to_numpy(dtype=float)).astype(np.float32)

    def zscore(self, values: np.ndarray) -> np.ndarray:
        return self.raw_scaler.transform(np.atleast_2d(values))


def display_scale(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Map DNA features onto [0, 1] for charts (tempo/loudness via fixed display ranges)."""
    features = features or [f for f in DNA_FEATURES if f in df.columns]
    out = pd.DataFrame(index=df.index)
    for f in features:
        v = df[f].astype(float)
        if f in UNIT_FEATURES:
            out[f] = v.clip(0, 1)
        else:
            lo, hi = DISPLAY_RANGES[f]
            out[f] = ((v - lo) / (hi - lo)).clip(0, 1)
    return out


def genre_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Multi-hot genre labels (artist/genre feature set)."""
    if "genre" not in df.columns:
        return np.zeros((len(df), 0), dtype=np.float32), []
    mlb = MultiLabelBinarizer()
    m = mlb.fit_transform(df["genre"]).astype(np.float32)
    return m, list(mlb.classes_)


def artist_codes(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    codes, uniques = pd.factorize(df["primary_artist"])
    return codes.astype(np.int64), list(uniques)


def feature_sets(df: pd.DataFrame, pipe: FeaturePipeline) -> dict[str, object]:
    g, genres = genre_matrix(df)
    a, artists = artist_codes(df)
    sets: dict[str, object] = {
        "similarity": pipe.model_inputs(df),
        "artist_genre": {"genre_multi_hot": g, "genres": genres, "artist_codes": a, "artists": artists},
        "diversity": df[[c for c in ("primary_artist", "album_name", "primary_genre") if c in df.columns]].copy(),
    }
    if "popularity" in df.columns:
        sets["popularity"] = df[["popularity", "popularity_tier"]].copy()
    return sets
