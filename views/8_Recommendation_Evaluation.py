import numpy as np
import pandas as pd
import streamlit as st

from src import ui
from src.experiment_tracker import METRIC_LABELS
from src.feedback import NEGATIVE, POSITIVE, FeedbackStore
from src.hybrid_recommender import STRATEGIES, weights_from_preferences
from src.recommendation_metrics import list_metrics, ndcg_at_k, precision_at_k
from src.visualization import strategy_metric_bars

engine = ui.get_engine()
ui.page_header("Research", "Recommendation Evaluation",
               "Every metric SoundBridge uses, computed live for your own seeds — plus the relevance numbers that come "
               "from real feedback, kept strictly separate from proxies.")

DEFS = pd.DataFrame([
    ("Relevance", "Encoder similarity", "Mean cosine between recommended tracks and your profile in the learned space.", "Proxy"),
    ("Relevance", "Raw-feature similarity", "Same, in z-scored raw audio-feature space — independent of the trained model.", "Proxy"),
    ("Relevance", "Genre-consistency precision@k", "Share of recommendations sharing a genre label with your seeds.", "Proxy"),
    ("Relevance", "Precision@k / NDCG@k (feedback)", "From your likes/dislikes on recommended tracks.", "Real feedback"),
    ("Diversity", "Intra-list diversity", "Mean pairwise cosine distance between recommended tracks (encoder space).", "Computed"),
    ("Diversity", "Artist / genre diversity", "Unique artists / unique primary genres divided by list length.", "Computed"),
    ("Diversity", "Audio-feature diversity", "Mean per-feature standard deviation of z-scored features in the list.", "Computed"),
    ("Discovery", "Novelty", "1 − exposure, mixing artist exposure (70%) and track popularity (30%).", "Computed"),
    ("Discovery", "Lesser-known artist share", "Tracks whose artist's best popularity in the dataset is 1–35.", "Computed"),
    ("Discovery", "Catalogue / artist coverage", "Unique tracks / artists recommended across many lists ÷ catalogue size.", "Experiment Lab"),
    ("Repetition", "Repeated-artist %", "1 − unique artists / list length.", "Computed"),
    ("Repetition", "Duplicate-track %", "Repeated title + artist within the list.", "Computed"),
    ("Repetition", "Mean pairwise similarity", "Average similarity between recommended tracks (higher = more repetitive).", "Computed"),
], columns=["Family", "Metric", "Definition", "Type"])

seeds = st.session_state.seeds
tab_live, tab_feedback, tab_defs, tab_model = st.tabs(["Your lists", "Feedback-based relevance", "Metric definitions", "Model evaluation"])

with tab_live:
    if not seeds:
        ui.empty_state("Add seeds to evaluate", "Metrics are computed for the lists generated from your seeds.")
    else:
        p = st.session_state.prefs
        w = weights_from_preferences(p["familiarity"], p["discovery"], p["exploration"], p["diversity"])
        k = st.select_slider("List length", [5, 10, 15, 20], value=10)
        profile = ui.current_profile(engine)
        rows = {}
        for s in STRATEGIES:
            df = ui.recommend_now(engine, s, w, k=k)
            rows[s] = list_metrics(df["idx"].tolist(), tracks=engine.tracks, embeddings=engine.embeddings, raw_z=engine.raw_z,
                                   profile_embedding=profile.centroid, raw_profile=engine.raw_profile(seeds),
                                   seed_genres=engine.seed_genres(seeds), lesser_known=engine.lesser_known,
                                   artist_exposure=engine.artist_exposure)
            rows[s]["novelty"] = float(engine.novelty[df["idx"]].mean())
        summ = pd.DataFrame(rows).T
        metrics = ["relevance_learned", "relevance_raw_features", "proxy_genre_precision", "intra_list_diversity",
                   "artist_diversity", "novelty", "lesser_known_share", "repeated_artist_pct"]
        cols = st.columns(2, gap="medium")
        for i, m in enumerate(metrics):
            cols[i % 2].plotly_chart(strategy_metric_bars(summ, m, STRATEGIES, highlight="hybrid", title=METRIC_LABELS[m],
                                                          height=320), width="stretch")
        st.caption("Single profile — differences here are anecdotal. See the Experiment Lab for averages over hundreds of seed sets.")
        st.dataframe(summ.rename(columns=METRIC_LABELS).T.round(3), width="stretch")

with tab_feedback:
    fb = FeedbackStore().load()
    session_events = pd.DataFrame(st.session_state.fb.events)
    src = st.radio("Feedback source", ["Stored (consented) feedback", "This session only"], horizontal=True)
    data = fb if src.startswith("Stored") else session_events
    rated = data[data["feedback"].isin(POSITIVE | NEGATIVE)] if len(data) else data
    if len(rated) == 0:
        ui.empty_state("No relevance labels yet", "Like or dislike recommendations on Music Discovery or Hidden Gems. "
                       "Precision and NDCG are only reported once real labels exist.", "📊")
    else:
        rated = rated.assign(rel=rated["feedback"].isin(POSITIVE).astype(int))
        rows = []
        for (sess, strat), g in rated.groupby(["session_id", "strategy"]):
            g = g.sort_values("rank")
            ranks = g[g["rank"] > 0]
            rec = ranks["track_idx"].tolist()
            relevant = ranks.loc[ranks["rel"] == 1, "track_idx"].tolist()
            kk = len(rec)
            rows.append({"strategy": strat, "labels": len(g), "precision@labelled": precision_at_k(rec, relevant, kk) if kk else np.nan,
                         "ndcg@labelled": ndcg_at_k(rec, {i: 1.0 for i in relevant}, kk) if kk and relevant else np.nan})
        res = pd.DataFrame(rows).groupby("strategy").agg(labels=("labels", "sum"), precision=("precision@labelled", "mean"),
                                                         ndcg=("ndcg@labelled", "mean"))
        ui.tiles([(f"{len(rated)}", "Relevance labels", ""), (f"{rated['rel'].mean():.0%}", "Liked share", "of rated tracks"),
                  (f"{rated['session_id'].nunique()}", "Sessions", "")])
        st.dataframe(res.round(3), width="stretch")
        st.caption("Precision is computed over the tracks a person actually rated (unrated tracks are unknown, not irrelevant). "
                   f"With {len(rated)} labels these numbers are descriptive only — no significance is claimed.")

with tab_defs:
    st.dataframe(DEFS, hide_index=True, width="stretch")

with tab_model:
    ev = engine.evaluation
    if ev:
        st.markdown("Held-out-artist evaluation of the SoundBridge Encoder (see Recommendation Engine for charts).")
        st.json({"retrieval": ev["retrieval_held_out_artists"], "heads": ev["encoder_heads"], "baselines": ev["baselines"]})
ui.footer()
