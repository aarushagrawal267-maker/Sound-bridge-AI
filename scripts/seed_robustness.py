"""Robustness check: retrain the encoder with different random seeds (and different held-out
artists) and report the evaluation spread. Nothing in models/ is overwritten.

Usage:  python scripts/seed_robustness.py [--seeds 7 123]
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
from src.feature_engineering import artist_codes, genre_matrix  # noqa: E402
from src.model import encode  # noqa: E402
from src.trainer import artist_split, evaluate_heads, retrieval_metrics, train_encoder  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 123])
    args = ap.parse_args()
    m = load_config()["model"]
    t = pd.read_parquet(path("processed_dir") / "tracks.parquet")
    pipe = joblib.load(path("models_dir") / "feature_pipeline.joblib")
    X, R = pipe.model_inputs(t), pipe.raw_matrix(t)
    gm, _ = genre_matrix(t)
    codes, _ = artist_codes(t)
    pop = t["popularity"].to_numpy(dtype=np.float32)
    rows = []
    for seed in args.seeds:
        tr, te = artist_split(codes, m["test_artist_fraction"], seed)
        res = train_encoder(X, gm, pop, codes, tr, hidden=m["hidden_dim"], emb_dim=m["embedding_dim"], dropout=m["dropout"],
                            epochs=m["epochs"], batch_size=m["batch_size"], lr=m["learning_rate"], weight_decay=m["weight_decay"],
                            temperature=m["temperature"], loss_weights=m["loss_weights"], seed=seed, verbose=False)
        E = encode(res.model, X)[0]
        enc = retrieval_metrics(E, te, codes, gm, k=10, seed=seed)
        raw = retrieval_metrics(R, te, codes, gm, k=10, seed=seed)
        heads = evaluate_heads(res.model, X, gm, pop, te)
        rows.append({"seed": seed, "encoder_artist_p@10": enc["artist_precision@10"], "raw_artist_p@10": raw["artist_precision@10"],
                     "encoder_genre_agree@10": enc["genre_agreement@10"], "raw_genre_agree@10": raw["genre_agreement@10"],
                     "genre_top5": heads["genre_top5_accuracy"], "popularity_r2": heads["popularity_r2"]})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    out = path("outputs_dir") / "reports" / "seed_robustness.json"
    out.write_text(json.dumps(rows, indent=2))
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
