"""Step 2: train and evaluate the SoundBridge Encoder, then build all model artifacts.

Usage:  python scripts/train_model.py [--epochs N]

Outputs (models/):        soundbridge_encoder.pt, embeddings.npy, genre_probs.npy,
                          pred_popularity.npy, raw_features.npy, clusters.npy,
                          coords_2d.npy, popularity_xgb.joblib, kmeans.joblib
Outputs (outputs/):       reports/model_evaluation.json, reports/training_history.csv,
                          reports/feature_importance.csv, figures/*.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.cluster import MiniBatchKMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from src.config import load_config, path  # noqa: E402
from src.feature_engineering import artist_codes, genre_matrix  # noqa: E402
from src.model import encode  # noqa: E402
from src.trainer import artist_split, evaluate_heads, retrieval_metrics, train_encoder  # noqa: E402

PURPLE, BLUE, PINK = "#8B5CF6", "#38BDF8", "#F472B6"


def permutation_importance_encoder(model, X, gm, pop, idx, names, seed=0, repeats=3):
    """Drop in genre top-5 accuracy and popularity R^2 when one input is shuffled."""
    rng = np.random.default_rng(seed)
    base = evaluate_heads(model, X, gm, pop, idx)
    rows = []
    groups: dict[str, list[int]] = {}
    for i, n in enumerate(names):
        groups.setdefault("key" if n.startswith("key_") else n, []).append(i)
    for gname, cols in groups.items():
        g_drop, p_drop = [], []
        for _ in range(repeats):
            Xp = X.copy()
            perm = rng.permutation(len(idx))
            Xp[np.ix_(idx, cols)] = X[np.ix_(idx[perm], cols)]
            m = evaluate_heads(model, Xp, gm, pop, idx)
            g_drop.append(base["genre_top5_accuracy"] - m["genre_top5_accuracy"])
            p_drop.append(base["popularity_r2"] - m["popularity_r2"])
        rows.append({"feature": gname, "genre_top5_drop": np.mean(g_drop), "popularity_r2_drop": np.mean(p_drop)})
    return pd.DataFrame(rows).sort_values("genre_top5_drop", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    mcfg = cfg["model"]
    seed = cfg["seed"]
    torch.set_num_threads(max(1, torch.get_num_threads()))

    models_dir, out = path("models_dir"), path("outputs_dir")
    (out / "reports").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    tracks = pd.read_parquet(path("processed_dir") / "tracks.parquet")
    pipe = joblib.load(models_dir / "feature_pipeline.joblib")
    X = pipe.model_inputs(tracks)
    R = pipe.raw_matrix(tracks)
    gm, genres = genre_matrix(tracks)
    codes, _ = artist_codes(tracks)
    pop = tracks["popularity"].to_numpy(dtype=np.float32)
    print(f"tracks={len(tracks)} inputs={X.shape[1]} genres={len(genres)} artists={codes.max() + 1}")

    train_idx, test_idx = artist_split(codes, mcfg["test_artist_fraction"], seed)
    print(f"train tracks={len(train_idx)}  held-out-artist tracks={len(test_idx)}")

    res = train_encoder(
        X, gm, pop, codes, train_idx,
        hidden=mcfg["hidden_dim"], emb_dim=mcfg["embedding_dim"], dropout=mcfg["dropout"],
        epochs=args.epochs or mcfg["epochs"], batch_size=mcfg["batch_size"], lr=mcfg["learning_rate"],
        weight_decay=mcfg["weight_decay"], temperature=mcfg["temperature"],
        loss_weights=mcfg["loss_weights"], seed=seed,
    )
    model = res.model
    hist = pd.DataFrame(res.history)
    hist.to_csv(out / "reports" / "training_history.csv", index=False)

    # ---------------- evaluation on held-out artists ----------------
    E_all, P_all, pop_pred = encode(model, X)
    spaces = {
        "SoundBridge Encoder (learned, 32-d)": E_all,
        "Raw audio features (z-score, 9-d)": R,
        "Encoder inputs without training (15-d)": X,
    }
    retrieval = {name: retrieval_metrics(sp, test_idx, codes, gm, k=10, seed=seed) for name, sp in spaces.items()}
    heads = evaluate_heads(model, X, gm, pop, test_idx)

    # baseline genre classifier: multinomial logistic regression on the same inputs
    primary = tracks["primary_genre"].to_numpy()
    lr = LogisticRegression(max_iter=300, n_jobs=None)
    sub = np.random.default_rng(seed).choice(train_idx, size=min(40000, len(train_idx)), replace=False)
    lr.fit(X[sub], primary[sub])
    proba = lr.predict_proba(X[test_idx])
    cls = list(lr.classes_)
    col_of = [genres.index(c) for c in cls]
    gm_test = gm[test_idx][:, col_of]
    top1 = proba.argmax(1)
    top5 = np.argsort(-proba, 1)[:, :5]
    lr_acc1 = float(gm_test[np.arange(len(test_idx)), top1].mean())
    lr_acc5 = float((np.take_along_axis(gm_test, top5, 1).sum(1) > 0).mean())

    # baseline popularity model: XGBoost on the same inputs (tracks with popularity > 0)
    tr = train_idx[pop[train_idx] > 0]
    te = test_idx[pop[test_idx] > 0]
    xgb = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                       colsample_bytree=0.8, n_jobs=4, random_state=seed)
    xgb.fit(X[tr], pop[tr])
    xgb_pred = xgb.predict(X[te])
    xgb_metrics = {
        "popularity_r2": float(r2_score(pop[te], xgb_pred)),
        "popularity_mae": float(mean_absolute_error(pop[te], xgb_pred)),
    }
    mean_baseline_mae = float(np.abs(pop[te] - pop[tr].mean()).mean())
    joblib.dump(xgb, models_dir / "popularity_xgb.joblib")

    # ---------------- explainability ----------------
    perm_df = permutation_importance_encoder(model, X, gm, pop, test_idx, pipe.input_names, seed=seed)
    try:
        import shap

        sample = X[np.random.default_rng(seed).choice(te, size=min(3000, len(te)), replace=False)]
        sv = shap.TreeExplainer(xgb).shap_values(sample)
        shap_df = pd.DataFrame({"feature": pipe.input_names, "mean_abs_shap": np.abs(sv).mean(0)})
        shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
        plt.figure()
        shap.summary_plot(sv, sample, feature_names=pipe.input_names, show=False, plot_size=(8, 5))
        plt.title("SHAP: XGBoost popularity baseline")
        plt.tight_layout()
        plt.savefig(out / "figures" / "shap_popularity_xgb.png", dpi=140)
        plt.close("all")
    except Exception as exc:  # SHAP is optional
        print("SHAP failed:", exc)
        shap_df = pd.DataFrame(columns=["feature", "mean_abs_shap"])
    imp = perm_df.merge(shap_df, on="feature", how="outer") if len(shap_df) else perm_df
    imp.to_csv(out / "reports" / "feature_importance.csv", index=False)

    # ---------------- catalogue artifacts ----------------
    km = MiniBatchKMeans(n_clusters=40, random_state=seed, n_init=5, batch_size=4096).fit(E_all)
    pca = PCA(n_components=2, random_state=seed).fit(E_all)
    np.save(models_dir / "embeddings.npy", E_all)
    np.save(models_dir / "genre_probs.npy", P_all.astype(np.float16))
    np.save(models_dir / "pred_popularity.npy", pop_pred)
    np.save(models_dir / "raw_features.npy", R)
    np.save(models_dir / "clusters.npy", km.labels_.astype(np.int16))
    np.save(models_dir / "coords_2d.npy", pca.transform(E_all).astype(np.float32))
    np.save(models_dir / "test_idx.npy", test_idx)
    joblib.dump(km, models_dir / "kmeans.joblib")
    model.save(
        models_dir / "soundbridge_encoder.pt",
        extra={"genres": genres, "input_names": pipe.input_names, "trained": time.strftime("%Y-%m-%d %H:%M")},
    )

    evaluation = {
        "dataset": {"tracks": len(tracks), "train_tracks": int(len(train_idx)), "test_tracks": int(len(test_idx)),
                    "test_artists": int(len(np.unique(codes[test_idx]))), "genres": len(genres)},
        "training": {"epochs": len(res.history), "seconds": round(res.seconds, 1),
                     "final_loss": res.history[-1], "config": mcfg},
        "retrieval_held_out_artists": retrieval,
        "encoder_heads": heads,
        "baselines": {
            "logistic_regression_genre": {"genre_top1_accuracy": lr_acc1, "genre_top5_accuracy": lr_acc5},
            "xgboost_popularity": xgb_metrics,
            "mean_predictor_popularity_mae": mean_baseline_mae,
        },
        "pca_explained_variance": [float(v) for v in pca.explained_variance_ratio_],
    }
    (out / "reports" / "model_evaluation.json").write_text(json.dumps(evaluation, indent=2))
    print(json.dumps(evaluation, indent=2))

    # ---------------- figures ----------------
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, col, c in zip(ax, ["contrastive", "genre", "popularity"], [PURPLE, BLUE, PINK]):
        a.plot(hist["epoch"], hist[col], color=c, lw=2)
        a.set_title(f"{col} loss")
        a.set_xlabel("epoch")
    fig.suptitle("SoundBridge Encoder training curves")
    fig.tight_layout()
    fig.savefig(out / "figures" / "training_curves.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
