"""Static report figures (Matplotlib / Seaborn) from the real data and saved experiments.

Usage:  python scripts/make_figures.py      -> outputs/figures/*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SHORT_LABELS, path  # noqa: E402
from src.eda import correlations  # noqa: E402
from src.experiment_tracker import list_experiments, load_experiment  # noqa: E402
from src.visualization import short_strategy  # noqa: E402

BG, FG, GRID = "#12111c", "#ECEAF6", "#2a2838"
VIOLET, BLUE, MUTED = "#9085e9", "#3987e5", "#4a4663"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG,
                     "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG, "axes.edgecolor": GRID,
                     "grid.color": GRID, "font.size": 10})


def save(fig, name):
    out = path("outputs_dir") / "figures" / name
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    t = pd.read_parquet(path("processed_dir") / "tracks.parquet")
    t["genre"] = t["genre"].map(list)

    fig, axes = plt.subplots(2, 4, figsize=(16, 6.5))
    for ax, f in zip(axes.flat, ["popularity", "danceability", "energy", "acousticness", "valence", "tempo",
                                 "instrumentalness", "loudness"]):
        ax.hist(t[f], bins=40, color=VIOLET, edgecolor=BG)
        ax.set_title(SHORT_LABELS.get(f, f.title()))
        ax.grid(axis="y", alpha=.4)
    fig.suptitle(f"Feature distributions — {len(t):,} de-duplicated tracks", color=FG)
    save(fig, "eda_distributions.png")

    fig, ax = plt.subplots(figsize=(8, 6.5))
    c = correlations(t)
    c.index = c.columns = [SHORT_LABELS.get(x, x.title()) for x in c.columns]
    sns.heatmap(c, cmap=sns.diverging_palette(250, 25, s=75, l=55, as_cmap=True), center=0, annot=True, fmt=".2f",
                linewidths=1, linecolor=BG, cbar_kws={"shrink": .7}, ax=ax, annot_kws={"size": 8})
    ax.set_title("Spearman correlations (audio features and popularity)")
    save(fig, "eda_correlations.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, f in zip(axes, ["danceability", "energy"]):
        ax.hexbin(t[f], t["popularity"], gridsize=40, cmap="magma", mincnt=1)
        ax.set_xlabel(f)
        ax.set_ylabel("popularity")
        ax.set_title(f"Popularity vs {f}")
    save(fig, "eda_popularity_vs_features.png")

    a = t.groupby("primary_artist")["popularity"].max().sort_values(ascending=False).to_numpy()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(np.arange(1, len(a) + 1) / len(a), np.cumsum(a) / a.sum(), color=VIOLET, lw=2.5, label="observed")
    ax.plot([0, 1], [0, 1], ls=":", color=FG, alpha=.6, label="equal exposure")
    ax.set_xlabel("share of artists (most exposed first)")
    ax.set_ylabel("share of summed artist exposure")
    ax.legend(frameon=False)
    ax.set_title("Artist exposure concentration (this dataset)")
    save(fig, "artist_exposure_concentration.png")

    exps = list_experiments()
    if not exps.empty:
        sc = exps[exps["name"] == "strategy_comparison"]
        if len(sc):
            s = load_experiment(sc.iloc[0]["dir"])["summary"]
            metrics = [("relevance_learned", "Relevance (encoder, proxy)"), ("relevance_raw_features", "Relevance (raw features, proxy)"),
                       ("proxy_genre_precision", "Genre-consistency precision@10"), ("intra_list_diversity", "Intra-list diversity"),
                       ("lesser_known_share", "Lesser-known artist share"), ("exposure_gini", "Exposure Gini (lower = fairer)")]
            fig, axes = plt.subplots(2, 3, figsize=(15, 7))
            names = [short_strategy(x) for x in s.index]
            for ax, (m, title) in zip(axes.flat, metrics):
                colors = [VIOLET if n == "Hybrid" else MUTED for n in names]
                ax.barh(names[::-1], s[m].to_numpy()[::-1], color=colors[::-1])
                ax.set_title(title)
                ax.grid(axis="x", alpha=.4)
            fig.suptitle("Strategy comparison — 200 seed sets × top-10", color=FG)
            save(fig, "experiment_strategy_comparison.png")
        ns = exps[exps["name"] == "novelty_sweep"]
        if len(ns):
            s = load_experiment(ns.iloc[0]["dir"])["summary"]
            w = [float(i.split("=")[1]) for i in s.index]
            fig, ax1 = plt.subplots(figsize=(7, 4.2))
            ax1.plot(w, s["lesser_known_share"], "o-", color=VIOLET, lw=2, label="lesser-known artist share")
            ax1.plot(w, s["relevance_learned"], "s-", color=BLUE, lw=2, label="relevance (encoder)")
            ax1.plot(w, s["proxy_genre_precision"], "^-", color="#d95926", lw=2, label="genre-consistency precision")
            ax1.set_xlabel("hybrid novelty weight")
            ax1.set_ylim(0, 1)
            ax1.grid(alpha=.4)
            ax1.legend(frameon=False, loc="lower left")
            ax1.set_title("Novelty sweep: discovery vs relevance")
            save(fig, "experiment_novelty_sweep.png")


if __name__ == "__main__":
    main()
