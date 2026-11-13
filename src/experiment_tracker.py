"""Recommendation Experiment Lab: reproducible offline comparisons of strategies.

Protocol
* Seed sets are sampled with a fixed random seed. Each set holds `seeds_per_set` tracks
  from one randomly chosen genre, by different artists, with popularity >= 30 (songs a
  listener is plausibly already familiar with).
* Every strategy produces a top-k list for every seed set.
* List metrics are averaged across seed sets; coverage metrics are computed over the union
  of all lists.

Relevance here is a *proxy* (similarity to the seed profile, genre consistency): no user
judged these lists. Each run is saved with its configuration and a Markdown report.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import path
from .hybrid_recommender import STRATEGIES, RankingWeights, Recommender
from .recommendation_metrics import artist_coverage, catalog_coverage, gini, list_metrics

METRIC_LABELS = {
    "relevance_learned": "Relevance · encoder similarity (proxy)",
    "relevance_raw_features": "Relevance · raw-feature similarity (proxy)",
    "proxy_genre_precision": "Genre-consistency precision@k (proxy)",
    "intra_list_diversity": "Intra-list diversity",
    "artist_diversity": "Artist diversity",
    "genre_unique_ratio": "Genre diversity",
    "genre_entropy": "Genre entropy (bits)",
    "audio_feature_diversity": "Audio-feature diversity",
    "novelty": "Novelty (exposure-based)",
    "mean_popularity": "Mean popularity",
    "lesser_known_share": "Lesser-known artist share",
    "mean_artist_exposure": "Mean artist exposure",
    "repeated_artist_pct": "Repeated-artist %",
    "duplicate_track_pct": "Duplicate-track %",
    "mean_pairwise_similarity": "Mean pairwise similarity",
    "catalog_coverage": "Catalogue coverage",
    "artist_coverage": "Artist coverage",
    "exposure_gini": "Exposure Gini (lower = more even)",
}


def sample_seed_sets(engine, n_sets: int, per_set: int, seed: int = 42, min_popularity: int = 30) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    t = engine.tracks
    eligible = t[t["popularity"] >= min_popularity]
    by_genre = {g: grp for g, grp in eligible.groupby("primary_genre") if grp["primary_artist"].nunique() >= per_set}
    genres = sorted(by_genre)
    sets = []
    while len(sets) < n_sets:
        g = genres[rng.integers(len(genres))]
        grp = by_genre[g].drop_duplicates("primary_artist")
        sets.append(sorted(rng.choice(grp["idx"].to_numpy(), size=per_set, replace=False).tolist()))
    return sets


def run_experiment(
    engine,
    strategies: dict[str, RankingWeights | None] | None = None,
    n_sets: int = 100,
    per_set: int = 3,
    k: int = 10,
    seed: int = 42,
    name: str = "experiment",
    save: bool = True,
    progress=None,
) -> dict:
    rec = Recommender(engine)
    strategies = strategies or {s: None for s in STRATEGIES}
    seed_sets = sample_seed_sets(engine, n_sets, per_set, seed)
    rows, all_recs, rec_rows = [], {s: [] for s in strategies}, []
    t0 = time.time()
    total = len(seed_sets) * len(strategies)
    done = 0
    for si, seeds in enumerate(seed_sets):
        profile = engine.profile(seeds)
        rp = engine.raw_profile(seeds)
        sg = engine.seed_genres(seeds)
        for sname, w in strategies.items():
            base = sname.split("@")[0]
            df = rec.recommend(profile, base, weights=w, k=k)
            ids = df["idx"].tolist()
            all_recs[sname].append(ids)
            rec_rows += [(sname, si, r, i, int(engine.popularity[i])) for r, i in enumerate(ids, 1)]
            m = list_metrics(ids, tracks=engine.tracks, embeddings=engine.embeddings, raw_z=engine.raw_z,
                             profile_embedding=profile.centroid, raw_profile=rp, seed_genres=sg,
                             lesser_known=engine.lesser_known, artist_exposure=engine.artist_exposure)
            m["novelty"] = float(engine.novelty[ids].mean()) if ids else np.nan
            m.update({"strategy": sname, "seed_set": si, "n": len(ids)})
            rows.append(m)
            done += 1
            if progress:
                progress(done / total)
    per_set_df = pd.DataFrame(rows)
    summary = per_set_df.drop(columns=["seed_set", "n"]).groupby("strategy").mean(numeric_only=True)
    n_artists = engine.tracks["primary_artist"].nunique()
    for sname, recs in all_recs.items():
        summary.loc[sname, "catalog_coverage"] = catalog_coverage(recs, len(engine.tracks))
        summary.loc[sname, "artist_coverage"] = artist_coverage(
            [[engine.artists[i] for i in r] for r in recs], n_artists
        )
        counts = pd.Series([engine.artists[i] for r in recs for i in r]).value_counts().to_numpy()
        summary.loc[sname, "exposure_gini"] = gini(counts)
    summary = summary.reindex(list(strategies))
    ci = per_set_df.groupby("strategy").agg(lambda s: 1.96 * s.std(ddof=1) / np.sqrt(len(s)) if s.dtype.kind == "f" else np.nan)
    result = {
        "name": name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - t0, 1),
        "config": {
            "n_sets": n_sets, "per_set": per_set, "k": k, "seed": seed,
            "strategies": {s: (asdict(w) if w is not None else "default") for s, w in strategies.items()},
            "catalog_size": len(engine.tracks),
        },
        "summary": summary,
        "ci95": ci.reindex(list(strategies)),
        "per_set": per_set_df,
        "recommendations": pd.DataFrame(rec_rows, columns=["strategy", "seed_set", "rank", "idx", "popularity"]),
        "seed_sets": seed_sets,
    }
    if save:
        result["dir"] = str(save_experiment(result))
    return result


def save_experiment(result: dict) -> Path:
    stamp = result["timestamp"].replace(" ", "_").replace(":", "")
    d = path("experiments_dir") / f"{stamp}_{result['name']}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({**result["config"], "name": result["name"],
                                               "timestamp": result["timestamp"], "seconds": result["seconds"],
                                               "seed_sets": result["seed_sets"]}, indent=2))
    result["summary"].to_csv(d / "summary.csv")
    result["per_set"].to_csv(d / "per_seed_set.csv", index=False)
    result["recommendations"].to_csv(d / "recommendations.csv", index=False)
    if result.get("ci95") is not None:
        result["ci95"].to_csv(d / "ci95.csv")
    (d / "report.md").write_text(experiment_report(result))
    return d


def experiment_report(result: dict) -> str:
    s = result["summary"]
    ci = result.get("ci95")
    cfg = result["config"]
    cols = [c for c in METRIC_LABELS if c in s.columns]
    lines = [
        f"# Experiment report: {result['name']}",
        "",
        f"* Run: {result['timestamp']} ({result['seconds']} s)",
        f"* Seed sets: {cfg['n_sets']} x {cfg['per_set']} seeds, random seed {cfg['seed']}, top-k = {cfg['k']}",
        f"* Catalogue: {cfg['catalog_size']:,} tracks",
        "",
        "Relevance metrics are proxies (similarity to the seed profile and genre-label "
        "consistency). No listener judged these lists. Values are means over seed sets; "
        "± values are 95% normal-approximation confidence half-widths.",
        "",
        "| Metric | " + " | ".join(STRATEGIES.get(x.split("@")[0], x) + ("" if "@" not in x else f" ({x.split('@')[1]})") for x in s.index) + " |",
        "|---|" + "---|" * len(s.index),
    ]
    for c in cols:
        vals = []
        for strat in s.index:
            v = s.loc[strat, c]
            h = ci.loc[strat, c] if ci is not None and c in ci.columns and strat in ci.index else np.nan
            vals.append(f"{v:.3f}" + (f" ± {h:.3f}" if pd.notna(h) else ""))
        lines.append(f"| {METRIC_LABELS[c]} | " + " | ".join(vals) + " |")
    lines += ["", "## Configuration", "```json", json.dumps(cfg["strategies"], indent=2), "```"]
    return "\n".join(lines)


def list_experiments() -> pd.DataFrame:
    rows = []
    for d in sorted(path("experiments_dir").glob("*/config.json"), reverse=True):
        c = json.loads(d.read_text())
        rows.append({"name": c.get("name"), "timestamp": c.get("timestamp"), "n_sets": c.get("n_sets"),
                     "k": c.get("k"), "strategies": ", ".join(c.get("strategies", {})), "dir": str(d.parent)})
    return pd.DataFrame(rows)


def load_experiment(directory: str | Path) -> dict:
    d = Path(directory)
    return {
        "dir": str(d),
        "config": json.loads((d / "config.json").read_text()),
        "summary": pd.read_csv(d / "summary.csv", index_col=0),
        "ci95": pd.read_csv(d / "ci95.csv", index_col=0) if (d / "ci95.csv").exists() else None,
        "recommendations": pd.read_csv(d / "recommendations.csv") if (d / "recommendations.csv").exists() else None,
        "report": (d / "report.md").read_text() if (d / "report.md").exists() else "",
    }
