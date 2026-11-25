"""Confidence intervals and checks for the headline evaluation numbers.

Usage:  python scripts/evaluation_confidence.py [--boot 1000]

Reads the artifacts written by train_model.py (and, when present, the newest run of each offline
experiment) and writes outputs/reports/evaluation_confidence.json with:

1. held_out_retrieval   - 95% bootstrap intervals for the held-out retrieval metrics of each space
                          and for the encoder-minus-raw difference and ratio. Queries by the same
                          artist are not independent, so whole artists are resampled.
2. credited_artist_check - the same metrics split by whether the held-out artist is credited as a
                          secondary artist on a training track (secondary credits are never used as
                          training labels, but such artists are not completely unseen).
3. normalisation_check  - the feature transforms are fitted on the whole catalogue (no labels). This
                          refits them on training artists only and reports how much the held-out
                          inputs and the (untrained) raw-feature retrieval change.
4. paired_differences   - for the offline experiments, mean differences with 95% intervals computed
                          per seed set (every strategy is scored on the same 200 seed sets, so the
                          paired interval is the right one for "does A differ from B").
Nothing is retrained and no other file is changed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, path  # noqa: E402
from src.feature_engineering import FeaturePipeline, artist_codes, genre_matrix  # noqa: E402
from src.trainer import artist_split, cluster_bootstrap_ci, retrieval_metrics, retrieval_per_query  # noqa: E402

SPACES = {
    "SoundBridge Encoder (learned, 32-d)": "encoder",
    "Raw audio features (z-score, 9-d)": "raw",
    "Encoder inputs without training (15-d)": "inputs",
}
PAIRED = {
    # experiment name -> (reference strategy, metrics)
    "strategy_comparison": ("content", ["relevance_learned", "relevance_raw_features", "proxy_genre_precision",
                                        "lesser_known_share", "exposure_gini"]),
    "novelty_sweep": ("hybrid@novelty=0.0", ["relevance_learned", "relevance_raw_features", "proxy_genre_precision",
                                             "lesser_known_share"]),
}


def _r(x: float) -> float:
    return round(float(x), 4)


def paired_differences(per_set: pd.DataFrame, reference: str, metrics: list[str]) -> dict:
    ref = per_set[per_set["strategy"] == reference].set_index("seed_set")
    out = {}
    for strat, g in per_set.groupby("strategy", sort=False):
        if strat == reference:
            continue
        g = g.set_index("seed_set")
        row = {}
        for m in metrics:
            if m not in g.columns:
                continue
            d = (g[m] - ref[m]).dropna()
            if len(d) < 2:
                continue
            h = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
            row[m] = {"diff": _r(d.mean()), "ci95": [_r(d.mean() - h), _r(d.mean() + h)], "n": int(len(d))}
        out[f"{strat} - {reference}"] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples")
    args = ap.parse_args()
    cfg = load_config()
    seed = cfg["seed"]
    models = path("models_dir")
    tracks = pd.read_parquet(path("processed_dir") / "tracks.parquet")
    pipe = joblib.load(models / "feature_pipeline.joblib")
    spaces = {
        "SoundBridge Encoder (learned, 32-d)": np.load(models / "embeddings.npy"),
        "Raw audio features (z-score, 9-d)": pipe.raw_matrix(tracks),
        "Encoder inputs without training (15-d)": pipe.model_inputs(tracks),
    }
    gm, _ = genre_matrix(tracks)
    codes, _ = artist_codes(tracks)
    train_idx, test_idx = artist_split(codes, cfg["model"]["test_artist_fraction"], seed)

    per_query, point = {}, {}
    for name, sp in spaces.items():
        r = retrieval_per_query(sp, test_idx, codes, gm, k=10, seed=seed)
        ref = retrieval_metrics(sp, test_idx, codes, gm, k=10, seed=seed)
        assert np.isclose(r["same_artist"].mean(), ref["artist_precision@10"]), "per-query results disagree"
        key = SPACES[name]
        per_query[key] = r
        point[key] = ref
    q = per_query["encoder"]["query"]
    assert all(np.array_equal(r["query"], q) for r in per_query.values()), "spaces must share the same queries"
    query_artist = codes[test_idx][q]

    values = {}
    for key, r in per_query.items():
        values[f"{key}_precision"] = r["same_artist"].mean(1)
        values[f"{key}_hit"] = r["same_artist"].any(1).astype(float)
        values[f"{key}_genre"] = r["genre_overlap"].mean(1)

    def stat(m: dict) -> dict:
        out = dict(m)
        out["precision_diff_encoder_minus_raw"] = m["encoder_precision"] - m["raw_precision"]
        out["precision_ratio_encoder_over_raw"] = m["encoder_precision"] / m["raw_precision"]
        out["genre_diff_encoder_minus_raw"] = m["encoder_genre"] - m["raw_genre"]
        return out

    ci = cluster_bootstrap_ci(values, query_artist, statistic=stat, n_boot=args.boot, seed=seed)
    means = stat({n: float(v.mean()) for n, v in values.items()})
    retrieval = {
        "n_queries": int(len(q)),
        "n_query_artists": int(len(np.unique(query_artist))),
        "method": f"percentile bootstrap over query artists, {args.boot} resamples",
        "metrics": {n: {"value": _r(means[n]), "ci95": [_r(a) for a in ci[n]]} for n in means},
    }

    # held-out artists that appear as a secondary credit on a training track
    credited_on_train = {a for lst in tracks["artist_list"].iloc[train_idx] for a in lst}
    held_out_artists = tracks["primary_artist"].iloc[test_idx].unique()
    overlap_artist = np.array([a in credited_on_train for a in held_out_artists])
    q_overlap = tracks["primary_artist"].iloc[test_idx].iloc[q].isin(credited_on_train).to_numpy()
    check = {
        "held_out_artists": int(len(held_out_artists)),
        "held_out_artists_credited_on_a_training_track": int(overlap_artist.sum()),
        "queries_from_those_artists": int(q_overlap.sum()),
    }
    for label, mask in (("not_credited_on_train", ~q_overlap), ("credited_on_train", q_overlap)):
        row = {key: _r(values[f"{key}_precision"][mask].mean()) for key in ("encoder", "raw", "inputs")}
        row["ratio_encoder_over_raw"] = _r(row["encoder"] / row["raw"])
        check[f"artist_precision@10_{label}"] = row

    train_only = FeaturePipeline.fit(tracks.iloc[train_idx], seed=seed)
    x_diff = np.abs(train_only.model_inputs(tracks)[test_idx] - spaces["Encoder inputs without training (15-d)"][test_idx])
    normalisation = {
        "mean_abs_input_change_held_out": _r(x_diff.mean()),
        "raw_artist_precision@10_fit_on_all": _r(point["raw"]["artist_precision@10"]),
        "raw_artist_precision@10_fit_on_train_only": _r(
            retrieval_metrics(train_only.raw_matrix(tracks), test_idx, codes, gm, k=10, seed=seed)["artist_precision@10"]),
    }

    paired = {}
    exp_dir = path("experiments_dir")
    for name, (reference, metrics) in PAIRED.items():
        runs = sorted(exp_dir.glob(f"*_{name}/per_seed_set.csv"))
        if not runs:
            continue
        per_set = pd.read_csv(runs[-1])
        paired[name] = {"source": str(runs[-1].parent.relative_to(path("outputs_dir").parent)),
                        "method": "mean paired difference over seed sets, 95% normal interval",
                        "differences": paired_differences(per_set, reference, metrics)}

    labels_per_track = gm[test_idx].sum(1)
    genre_labels = {"held_out_share_with_several_labels": _r((labels_per_track > 1).mean()),
                    "held_out_mean_labels": _r(labels_per_track.mean())}

    result = {"held_out_retrieval": retrieval, "credited_artist_check": check, "genre_labels": genre_labels,
              "normalisation_check": normalisation, "paired_differences": paired}
    out = path("outputs_dir") / "reports" / "evaluation_confidence.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print("\nWrote", out)


if __name__ == "__main__":
    main()
