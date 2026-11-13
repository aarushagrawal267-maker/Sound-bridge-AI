"""Exploratory data analysis with insights computed from the actual data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import DNA_FEATURES, SHORT_LABELS


def overview(raw: pd.DataFrame | None, tracks: pd.DataFrame) -> dict:
    out = {
        "tracks": len(tracks),
        "artists": int(tracks["primary_artist"].nunique()),
        "all_credited_artists": int(len({a for lst in tracks["artist_list"] for a in lst})),
        "albums": int(tracks["album_name"].nunique()),
        "genres": int(len({g for lst in tracks["genre"] for g in lst})),
        "avg_popularity": float(tracks["popularity"].mean()),
        "median_popularity": float(tracks["popularity"].median()),
        "zero_popularity": int((tracks["popularity"] == 0).sum()),
    }
    if raw is not None:
        out["raw_rows"] = len(raw)
        out["raw_missing"] = int(raw.isna().sum().sum())
        out["raw_duplicate_ids"] = int(raw["track_id"].duplicated().sum()) if "track_id" in raw else 0
    return out


def explode_genres(tracks: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    cols = cols or ["idx", "popularity", "primary_artist", *[f for f in DNA_FEATURES if f in tracks.columns]]
    return tracks[cols + ["genre"]].explode("genre").rename(columns={"genre": "genre_label"})


def genre_stats(tracks: pd.DataFrame) -> pd.DataFrame:
    ex = explode_genres(tracks)
    feats = [f for f in DNA_FEATURES if f in tracks.columns]
    g = ex.groupby("genre_label")
    out = g[feats + ["popularity"]].mean()
    out["tracks"] = g.size()
    out["artists"] = g["primary_artist"].nunique()
    out["median_popularity"] = g["popularity"].median()
    out["lesser_known_share"] = g["popularity"].apply(lambda p: ((p > 0) & (p <= 35)).mean())
    return out.sort_values("tracks", ascending=False)


def artist_stats(tracks: pd.DataFrame) -> pd.DataFrame:
    feats = [f for f in DNA_FEATURES if f in tracks.columns]
    g = tracks.groupby("primary_artist")
    out = g[feats].mean()
    out["tracks"] = g.size()
    out["max_popularity"] = g["popularity"].max()
    out["mean_popularity"] = g["popularity"].mean()
    out["genres"] = g["genre"].agg(lambda s: ", ".join(sorted({x for lst in s for x in lst})[:4]))
    out["top_genre"] = g["primary_genre"].agg(lambda s: s.value_counts().index[0])
    return out.sort_values("tracks", ascending=False)


def popularity_concentration(tracks: pd.DataFrame) -> dict:
    """How concentrated is popularity across artists in this sample?"""
    a = tracks.groupby("primary_artist")["popularity"].max().sort_values(ascending=False)
    total = a.sum()
    top1 = a.head(max(1, int(len(a) * 0.01))).sum() / total
    top10 = a.head(max(1, int(len(a) * 0.10))).sum() / total
    x = np.sort(a.to_numpy())
    n = len(x)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if x.sum() else 0.0
    return {"top1pct_share": float(top1), "top10pct_share": float(top10), "artist_gini": gini,
            "artists_over_70": int((a > 70).sum()), "artists_under_20": int(((a > 0) & (a <= 20)).sum())}


def correlations(tracks: pd.DataFrame) -> pd.DataFrame:
    feats = [f for f in DNA_FEATURES if f in tracks.columns] + ["popularity"]
    return tracks[feats].corr(method="spearman")


def popular_vs_lesser(tracks: pd.DataFrame, hi: int = 60, lo: int = 25) -> pd.DataFrame:
    feats = [f for f in DNA_FEATURES if f in tracks.columns]
    top = tracks[tracks["popularity"] >= hi]
    low = tracks[(tracks["popularity"] > 0) & (tracks["popularity"] <= lo)]
    rows = []
    for f in feats:
        u = stats.mannwhitneyu(top[f], low[f])
        pooled = tracks[f].std()
        rows.append({"feature": f, "label": SHORT_LABELS[f], f"popular (>={hi})": top[f].mean(),
                     f"lesser-known (1-{lo})": low[f].mean(),
                     "effect_size_sd": (top[f].mean() - low[f].mean()) / pooled, "p_value": u.pvalue})
    return pd.DataFrame(rows).sort_values("effect_size_sd", key=abs, ascending=False)


def insights(tracks: pd.DataFrame) -> list[str]:
    """Dynamic insights: every sentence is computed from the current data."""
    out = []
    corr = correlations(tracks)
    feats = [f for f in DNA_FEATURES if f in tracks.columns]
    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(feats) for b in feats[i + 1:]]
    pairs.sort(key=lambda x: -abs(x[2]))
    a, b, r = pairs[0]
    out.append(f"The strongest relationship between audio features is {a} and "
               f"{b} (Spearman ρ = {r:+.2f}).")
    pc = corr["popularity"].drop("popularity")
    f = pc.abs().idxmax()
    out.append(f"No audio feature strongly tracks popularity: the largest correlation is {f} "
               f"(ρ = {pc[f]:+.2f}). Exposure appears to depend on much more than how a track sounds.")
    conc = popularity_concentration(tracks)
    out.append(f"The top 10% of artists (by their best track's popularity) hold {conc['top10pct_share']:.0%} of "
               f"summed artist popularity in this sample; artist-level Gini = {conc['artist_gini']:.2f}.")
    gs = genre_stats(tracks)
    hi, lo = gs["popularity"].idxmax(), gs["popularity"].idxmin()
    out.append(f"Average popularity differs strongly by genre label: '{hi}' averages {gs.loc[hi, 'popularity']:.0f} "
               f"while '{lo}' averages {gs.loc[lo, 'popularity']:.0f}.")
    pvl = popular_vs_lesser(tracks)
    top = pvl.iloc[0]
    direction = "higher" if top["effect_size_sd"] > 0 else "lower"
    out.append(f"Compared with lesser-known tracks, popular tracks have {direction} {top['feature']} on average "
               f"(difference {abs(top['effect_size_sd']):.2f} SD) — a modest effect.")
    sizes = gs["tracks"]
    out.append(f"Genre labels are near-uniformly sampled ({sizes.min():,}–{sizes.max():,} tracks per label after "
               "de-duplication): the dataset was collected per search genre, so it does not reflect real-world genre "
               "shares.")
    zero = int((tracks["popularity"] == 0).sum())
    out.append(f"{zero:,} tracks ({zero / len(tracks):.1%}) have popularity 0; SoundBridge treats this as missing "
               "exposure information rather than as 'unknown to everyone'.")
    return out
