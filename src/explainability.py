"""WHY THIS SONG? - explanations generated from actual feature comparisons.

Three evidence sources, all computed (nothing is templated without data behind it):

1. Feature comparison  - the track's audio features vs. the seed profile, in units of
                          catalogue standard deviations.
2. Raw-cosine decomposition - exact per-feature contributions to the cosine similarity
                          in z-scored feature space (sum of contributions = similarity).
3. Encoder attribution - integrated gradients of the learned similarity w.r.t. each
                          input feature of the recommended track.
Plus ranking context: which ranking components (novelty, diversity, genre exploration)
lifted the track into the list.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DNA_FEATURES, SHORT_LABELS

NAMES = {"danceability": "danceability", "energy": "energy", "valence": "valence", "acousticness": "acousticness",
         "instrumentalness": "instrumentalness", "speechiness": "speechiness", "liveness": "liveness",
         "tempo": "tempo", "loudness": "loudness", "duration_ms": "duration", "mode": "mode", "key": "key",
         "time_signature": "time signature", "explicit": "explicit flag"}

CLOSE_Z = 0.35   # within 0.35 catalogue SDs counts as "similar"
FAR_Z = 1.0      # more than 1 SD counts as a notable difference


@dataclass
class Explanation:
    headline: str
    reasons: list[str]
    comparison: pd.DataFrame
    contributions: pd.DataFrame
    attribution: pd.DataFrame | None = None


def _fmt(feature: str, value: float) -> str:
    if feature == "tempo":
        return f"{value:.0f} BPM"
    if feature == "loudness":
        return f"{value:.1f} dB"
    return f"{value:.2f}"


def feature_comparison(tracks: pd.DataFrame, seed_idx: list[int], track_idx: int, catalog_std: pd.Series) -> pd.DataFrame:
    feats = [f for f in DNA_FEATURES if f in tracks.columns]
    seeds = tracks.loc[seed_idx, feats].astype(float)
    t = tracks.loc[track_idx, feats].astype(float)
    prof = seeds.mean()
    z = (t - prof) / catalog_std[feats]
    df = pd.DataFrame(
        {
            "feature": feats,
            "label": [SHORT_LABELS[f] for f in feats],
            "your_profile": prof.values,
            "this_track": t.values,
            "difference": (t - prof).values,
            "z_difference": z.values,
        }
    )
    df["closeness"] = np.where(df.z_difference.abs() <= CLOSE_Z, "similar",
                               np.where(df.z_difference.abs() >= FAR_Z, "different", "somewhat different"))
    return df


def cosine_contributions(raw_track: np.ndarray, raw_profile: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Exact decomposition of cosine(track, profile) in z-scored space into per-feature terms."""
    denom = np.linalg.norm(raw_track) * np.linalg.norm(raw_profile)
    contrib = raw_track * raw_profile / max(denom, 1e-12)
    return pd.DataFrame({"feature": names, "label": [SHORT_LABELS.get(n, n) for n in names], "contribution": contrib})


def group_attribution(attr: np.ndarray, input_names: list[str]) -> pd.DataFrame:
    rows: dict[str, float] = {}
    for a, n in zip(attr, input_names):
        key = "key" if n.startswith("key_") else n
        rows[key] = rows.get(key, 0.0) + float(a)
    df = pd.DataFrame({"feature": list(rows), "attribution": list(rows.values())})
    df["label"] = df.feature.map(lambda f: SHORT_LABELS.get(f, f.replace("_ms", "").replace("_", " ").title()))
    return df.sort_values("attribution", ascending=False)


def build_explanation(
    *,
    tracks: pd.DataFrame,
    seed_idx: list[int],
    track_idx: int,
    catalog_std: pd.Series,
    raw_track: np.ndarray,
    raw_profile: np.ndarray,
    raw_names: list[str],
    relevance: float,
    seed_genres: set[str],
    related_genre: tuple[str, str, float] | None = None,
    novelty_component: float = 0.0,
    diversity_component: float = 0.0,
    lesser_known_threshold: int = 35,
    discovery_mode: bool = False,
    attribution: pd.DataFrame | None = None,
    closest_seed: int | None = None,
) -> Explanation:
    comp = feature_comparison(tracks, seed_idx, track_idx, catalog_std)
    contrib = cosine_contributions(raw_track, raw_profile, raw_names)
    row = tracks.loc[track_idx]
    reasons: list[str] = []

    # 1. strongest shared characteristics (close AND distinctive relative to the catalogue)
    similar = comp[comp.closeness == "similar"].copy()
    if len(similar):
        similar["distinct"] = (similar["your_profile"] - tracks[similar.feature].mean().values).abs() / catalog_std[similar.feature].values
        similar = similar.sort_values("distinct", ascending=False)
        top = similar.head(2)
        parts = [f"{NAMES[r.feature]} ({_fmt(r.feature, r.this_track)} vs your {_fmt(r.feature, r.your_profile)})"
                 for r in top.itertuples() if r.feature != "tempo"]
        if parts:
            reasons.append("Similar " + " and ".join(parts) + " to your selected songs.")

    # 2. tempo
    if "tempo" in comp.feature.values:
        tr = comp.set_index("feature").loc["tempo"]
        seeds_t = tracks.loc[seed_idx, "tempo"].astype(float)
        lo, hi = seeds_t.min() - 5, seeds_t.max() + 5
        if lo <= tr.this_track <= hi:
            reasons.append(
                f"Sits inside your tempo range ({tr.this_track:.0f} BPM; your seeds span "
                f"{seeds_t.min():.0f}-{seeds_t.max():.0f} BPM)."
            )

    # 3. genre
    tgenres = list(row["genre"]) if "genre" in tracks.columns else []
    shared = [g for g in tgenres if g in seed_genres]
    if shared:
        reasons.append(f"Shares the '{shared[0]}' genre label with your seeds.")
    elif related_genre is not None:
        new_g, your_g, sim = related_genre
        reasons.append(
            f"Introduces '{new_g}', a genre the model places close in sound to your '{your_g}' "
            f"(genre similarity {sim:.2f}), while keeping a {relevance:.0%} match to your profile."
        )

    # 4. popularity / discovery
    pop = int(row.get("popularity", -1))
    if pop >= 0 and pop <= lesser_known_threshold and (discovery_mode or novelty_component > 0.05):
        reasons.append(
            f"Lesser-known in this dataset (popularity {pop}/100), surfaced by the discovery setting. "
            "Popularity measures exposure, not quality."
        )

    # 5. diversity
    if diversity_component > 0.05:
        diff = comp.reindex(comp.z_difference.abs().sort_values(ascending=False).index).iloc[0]
        reasons.append(
            f"Adds variety to the list: it differs most in {NAMES[diff.feature]} "
            f"({_fmt(diff.feature, diff.this_track)} vs your {_fmt(diff.feature, diff.your_profile)})."
        )

    # 6. closest seed
    if closest_seed is not None and len(seed_idx) > 1:
        s = tracks.loc[closest_seed]
        reasons.append(f"Closest to your seed '{s['track_name']}' by {s['primary_artist']}.")

    # 7. encoder attribution
    if attribution is not None and len(attribution):
        pos = attribution[attribution.attribution > 0].head(3)
        if len(pos):
            reasons.append(
                "The SoundBridge Encoder's match is driven most by "
                + ", ".join(pos.feature.map(lambda f: NAMES.get(f, f))) + " (integrated-gradients attribution)."
            )

    # main difference, stated honestly
    far = comp[comp.closeness == "different"]
    if len(far):
        f = far.reindex(far.z_difference.abs().sort_values(ascending=False).index).iloc[0]
        direction = "higher" if f.difference > 0 else "lower"
        reasons.append(
            f"Main difference: {direction} {NAMES[f.feature]} ({_fmt(f.feature, f.this_track)} vs your {_fmt(f.feature, f.your_profile)})."
        )

    n_sim = int((comp.closeness == "similar").sum())
    headline = f"{relevance:.0%} sound match - {n_sim} of {len(comp)} audio features closely match your Musical DNA"
    return Explanation(headline, reasons, comp, contrib, attribution)


def short_reason(comp: pd.DataFrame, shared_genre: str | None, lesser_known: bool) -> str:
    """One-line explanation for list views."""
    sim = comp[comp.closeness == "similar"]
    sim = sim[sim.feature != "loudness"]
    feats = " & ".join(sim.feature.map(NAMES).head(2)) if len(sim) else None
    bits = []
    if feats:
        bits.append(f"similar {feats}")
    if shared_genre:
        bits.append(f"shares {shared_genre}")
    if lesser_known:
        bits.append("lesser-known artist")
    return (", ".join(bits) or "close overall sound").capitalize()
