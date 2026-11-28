import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.config import DNA_FEATURES, SHORT_LABELS, path
from src.hybrid_recommender import STRATEGIES, RankingWeights
from src.visualization import ACCENT, CATEGORICAL, MUTED_MARK, TEXT_2, _finish, hbar

engine = ui.get_engine()
ev = engine.evaluation
ui.page_header("Research", "Recommendation Engine",
               "Under the hood of SoundBridge: the neural network that powers every page, how it was trained and "
               "tested, and the ranking strategies built on top of it.")

# ---------------------------------------------------------------- architecture
ui.section("The SoundBridge Encoder", "a multi-task neural network trained from scratch on this dataset (CPU, PyTorch)")
cfg = engine.model.config
st.markdown(f"""
<div class="sb-panel">
<div class="sb-flow">
  <div class="sb-tile"><div class="l">Input</div><div class="v" style="font-size:18px">{cfg['in_dim']} audio features</div>
  <div class="s">quantile-normalised audio + duration, key, mode, metre, explicit</div></div>
  <div class="arrow" aria-hidden="true">→</div>
  <div class="sb-tile"><div class="l">Backbone</div><div class="v" style="font-size:18px">MLP {cfg['hidden']}·{cfg['hidden']}</div>
  <div class="s">LayerNorm · GELU · dropout {cfg['dropout']}</div></div>
  <div class="arrow" aria-hidden="true">→</div>
  <div class="sb-tile"><div class="l">Sound embedding</div><div class="v" style="font-size:18px">{cfg['emb_dim']}-d, unit length</div>
  <div class="s">used for retrieval, profiles, diversity</div></div>
  <div class="arrow" aria-hidden="true">→</div>
  <div class="heads">
    <span class="sb-chip">Contrastive head · same-artist pairs (InfoNCE)</span>
    <span class="sb-chip">Genre head · {cfg['n_genres']} labels (cosine prototypes)</span>
    <span class="sb-chip">Popularity head · exposure regression</span>
  </div>
</div></div>""", unsafe_allow_html=True)
st.caption("Artist, genre and popularity are training targets only — never inputs — so the embedding describes how a "
           "track sounds. Training uses an artist-level split: 15% of artists are held out entirely for evaluation.")

if ev:
    d, tr = ev["dataset"], ev["training"]
    ui.tiles([
        (f"{d['train_tracks']:,}", "Training tracks", ""),
        (f"{d['test_tracks']:,}", "Held-out tracks", f"{d['test_artists']:,} unseen artists"),
        (f"{tr['epochs']}", "Epochs", f"{tr['seconds']:.0f} s on CPU"),
        (f"{ev['encoder_heads']['genre_top5_accuracy']:.0%}", "Genre top-5 accuracy", "held-out artists, 114 labels"),
    ])

    c1, c2 = st.columns([1.1, 1], gap="large")
    with c1:
        ui.section("Held-out evaluation", "unseen artists · 5,000 queries")
        ret = ev["retrieval_held_out_artists"]
        rows = []
        for name, m in ret.items():
            rows += [{"space": name, "metric": "Same-artist precision@10", "value": m["artist_precision@10"]},
                     {"space": name, "metric": "Same-artist hit@10", "value": m["artist_hit@10"]},
                     {"space": name, "metric": "Genre agreement@10", "value": m["genre_agreement@10"]}]
        df = pd.DataFrame(rows)
        fig = go.Figure()
        short = {"SoundBridge Encoder (learned, 32-d)": "SoundBridge Encoder", "Raw audio features (z-score, 9-d)": "Raw features (z-score)",
                 "Encoder inputs without training (15-d)": "Untrained inputs"}
        for color, name in zip([ACCENT, MUTED_MARK, "#3987e5"], ret):
            s = df[df.space == name]
            fig.add_trace(go.Bar(x=s["metric"], y=s["value"], name=short[name], marker=dict(color=color, cornerradius=4),
                                 text=[f"{v:.3f}" for v in s["value"]], textposition="outside", textfont=dict(color=TEXT_2, size=11),
                                 hovertemplate=short[name] + " · %{x}: %{y:.3f}<extra></extra>"))
        chance = next(iter(ret.values()))["genre_agreement_chance@10"]
        fig.add_annotation(x="Genre agreement@10", y=chance, text=f"chance {chance:.3f}", showarrow=False, yshift=10,
                           font=dict(color=TEXT_2, size=10))
        fig.update_layout(barmode="group", yaxis_title="score (higher is better)", bargap=0.3, bargroupgap=0.12)
        st.plotly_chart(_finish(fig, 380), width="stretch")
    with c2:
        ui.section("Heads vs. classical baselines", "same inputs, same held-out tracks")
        h, b = ev["encoder_heads"], ev["baselines"]
        tab = pd.DataFrame([
            {"Task": "Genre top-1 accuracy", "SoundBridge Encoder": h["genre_top1_accuracy"], "Baseline": b["logistic_regression_genre"]["genre_top1_accuracy"], "Baseline model": "LogReg"},
            {"Task": "Genre top-5 accuracy", "SoundBridge Encoder": h["genre_top5_accuracy"], "Baseline": b["logistic_regression_genre"]["genre_top5_accuracy"], "Baseline model": "LogReg"},
            {"Task": "Popularity R²", "SoundBridge Encoder": h["popularity_r2"], "Baseline": b["xgboost_popularity"]["popularity_r2"], "Baseline model": "XGBoost"},
            {"Task": "Popularity MAE", "SoundBridge Encoder": h["popularity_mae"], "Baseline": b["xgboost_popularity"]["popularity_mae"], "Baseline model": "XGBoost"},
        ])
        st.dataframe(tab, hide_index=True, width="stretch",
                     column_config={"SoundBridge Encoder": st.column_config.NumberColumn(format="%.3f"),
                                    "Baseline": st.column_config.NumberColumn(format="%.3f")})
        st.markdown(f"""
* The encoder roughly **doubles same-artist precision** over raw audio-feature similarity on artists it never saw.
* Its genre head beats a logistic-regression baseline on top-1 and top-5 accuracy.
* **Popularity is barely predictable from sound** (R² {h['popularity_r2']:.2f}; XGBoost {b['xgboost_popularity']['popularity_r2']:.2f};
  predicting the mean gives MAE {b['mean_predictor_popularity_mae']:.1f}). That supports treating popularity as exposure,
  not as a property of the music.
""")

    hist_path = path("outputs_dir") / "reports" / "training_history.csv"
    if hist_path.exists():
        ui.section("Training curves")
        hist = pd.read_csv(hist_path)
        fig = go.Figure()
        for color, col in zip(CATEGORICAL, ["contrastive", "genre", "popularity"]):
            y = hist[col] / hist[col].iloc[0]
            fig.add_trace(go.Scatter(x=hist["epoch"], y=y, mode="lines", name=f"{col} loss", line=dict(color=color, width=2),
                                     customdata=hist[col], hovertemplate=col + " epoch %{x}: %{customdata:.4f}<extra></extra>"))
        fig.update_layout(xaxis_title="epoch", yaxis_title="loss relative to epoch 1")
        st.plotly_chart(_finish(fig, 300), width="stretch")

    imp_path = path("outputs_dir") / "reports" / "feature_importance.csv"
    if imp_path.exists():
        ui.section("What the model relies on", "permutation importance (encoder) · SHAP (XGBoost popularity baseline)")
        imp = pd.read_csv(imp_path)
        c3, c4 = st.columns(2, gap="large")
        pi = imp.dropna(subset=["genre_top5_drop"]).sort_values("genre_top5_drop", ascending=False)
        lab = lambda f: SHORT_LABELS.get(f, f.replace("_ms", "").replace("_", " "))  # noqa: E731
        c3.plotly_chart(hbar([lab(f) for f in pi["feature"]], pi["genre_top5_drop"], "Encoder genre head: accuracy drop when shuffled",
                             height=420, fmt=".3f"), width="stretch")
        if "mean_abs_shap" in imp.columns:
            sh = imp.dropna(subset=["mean_abs_shap"]).sort_values("mean_abs_shap", ascending=False)
            c4.plotly_chart(hbar([lab(f) for f in sh["feature"]], sh["mean_abs_shap"], "XGBoost popularity: mean |SHAP| (points)",
                                 height=420, fmt=".2f"), width="stretch")

# ---------------------------------------------------------------- design a sound
ui.section("Design a sound — ask the model", "set audio features; the trained network embeds them live")
feats = [f for f in DNA_FEATURES if f in engine.tracks.columns]
med = engine.tracks[feats + ["duration_ms"]].median()
cols = st.columns(3)
vals = {}
for i, f in enumerate(feats):
    lo, hi = (40.0, 210.0) if f == "tempo" else (-30.0, 0.0) if f == "loudness" else (0.0, 1.0)
    vals[f] = cols[i % 3].slider(SHORT_LABELS[f], lo, hi, float(np.clip(med[f], lo, hi)), key=f"design_{f}")
probe = pd.DataFrame([{**vals, "duration_ms": med["duration_ms"], "mode": 1, "key": 0, "time_signature": 4, "explicit": 0}])
X = engine.pipe.model_inputs(probe)
emb, probs, pop = engine.encode_inputs(X)
top = np.argsort(-probs[0])[:6]
idx, sims = engine.space.neighbours(emb[0], 6)
c5, c6 = st.columns([1, 1.4], gap="large")
with c5:
    st.plotly_chart(hbar([engine.genres[j] for j in top], probs[0][top], "Genre head: what this sound resembles", height=300, fmt=".0%"),
                    width="stretch", config={"displayModeBar": False})
    st.metric("Popularity head prediction", f"{pop[0]:.0f} / 100",
              help="Expected popularity for this sound in this dataset. R² is low, so treat it as a weak signal.")
with c6:
    st.caption("Nearest real tracks in the encoder's sound space")
    for i, s in zip(idx[0], sims[0]):
        st.markdown(ui.track_card(engine, engine.tracks.loc[int(i)], match=float(s)), unsafe_allow_html=True)

# ---------------------------------------------------------------- strategy comparison
seeds = st.session_state.seeds
ui.section("Compare strategies on your seeds", "same profile, different ranking objectives")
if not seeds:
    ui.empty_state("Add seed songs to compare strategies", "Strategies differ only once there is a profile to rank for.")
else:
    a, b = st.columns(2, gap="large")
    s1 = a.selectbox("Strategy A", list(STRATEGIES), format_func=STRATEGIES.get, index=1)
    s2 = b.selectbox("Strategy B", list(STRATEGIES), format_func=STRATEGIES.get, index=6)
    with st.expander("Hybrid weights (used when a strategy is Hybrid)"):
        w = RankingWeights(
            relevance=st.slider("Relevance weight", 0.0, 1.0, 0.7, 0.05),
            novelty=st.slider("Novelty weight", 0.0, 1.0, 0.15, 0.05),
            diversity=st.slider("Diversity weight", 0.0, 1.0, 0.15, 0.05),
            genre=st.slider("Genre weight", 0.0, 1.0, 0.45, 0.05),
            repetition=st.slider("Repetition penalty", 0.0, 1.0, 0.2, 0.05),
            explore=st.slider("Genre exploration", 0.0, 1.0, 0.1, 0.05),
            max_per_artist=st.select_slider("Max tracks per artist", [1, 2, 3, 5, 10], value=2),
        )
    ra = ui.recommend_now(engine, s1, w, k=10)
    rb = ui.recommend_now(engine, s2, w, k=10)
    overlap = len(set(ra["idx"]) & set(rb["idx"]))
    ui.note(f"Overlap between the two lists: <b>{overlap} of 10</b> tracks.")
    for col, r, s in [(a, ra, s1), (b, rb, s2)]:
        with col:
            st.markdown(f"**{STRATEGIES[s]}** · mean popularity {r['popularity'].mean():.0f} · "
                        f"{r['primary_artist'].nunique()} artists · mean match {r['relevance'].mean():.2f}")
            for _, row in r.iterrows():
                st.markdown(ui.track_card(engine, row, rank=int(row["rank"]), match=float(row["relevance"]),
                                          discovery=float(row["discovery_score"])), unsafe_allow_html=True)
ui.footer()
