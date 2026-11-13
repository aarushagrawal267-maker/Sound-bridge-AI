import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.hidden_gems import LEVEL_LABELS, find_hidden_gems
from src.visualization import MUTED_MARK, PROFILE, TRACK, _finish

engine = ui.get_engine()
ui.page_header("Hidden Gems", "Lesser-known artists, matched to you",
               "Tracks that sound like your Musical DNA, by artists with low exposure <i>in this dataset</i>. "
               "Every gem must first clear a strict relevance floor — obscurity alone never qualifies a track.")

seeds = st.session_state.seeds
if not seeds:
    ui.empty_state("Pick some songs first", "Hidden Gems are matched to your Musical DNA.", "💎")
    st.stop()

ui.note("Low popularity does not mean high quality, and it does not tell us whether an artist is independent or "
        "emerging — the dataset has no such metadata. Spotify popularity reflects recent plays at collection time, "
        "so we treat it only as an <b>exposure proxy</b>, and we label results neutrally as <b>lesser-known in this dataset</b>.")

cfg = engine.cfg["hidden_gems"]["levels"]
level = st.segmented_control("Discovery level", list(LEVEL_LABELS), format_func=LEVEL_LABELS.get, default="balanced",
                             key="gem_level") or "balanced"
c = cfg[level]
st.caption(f"{LEVEL_LABELS[level]}: match in the top {c['top_percent']}% of the catalogue · artist's best track "
           f"popularity ≤ {c['max_artist_exposure']} · novelty weight {c['novelty_weight']} · one track per artist")

profile = ui.current_profile(engine)
res = find_hidden_gems(engine, profile, level, k=12)
ui.tiles([
    (f"{res.n_relevant:,}", "Tracks above relevance floor", f"cosine ≥ {res.relevance_floor:.2f} in encoder space"),
    (f"{res.n_lesser_known:,}", "…of which lesser-known", "eligible candidates"),
    (f"{res.n_excluded_irrelevant:,}", "Lesser-known but excluded", "did not match your sound closely enough"),
    (f"{len(res.tracks)}", "Gems shown", "after diversity re-ranking"),
])

if res.tracks.empty:
    ui.empty_state("No gems at this level", "Try the Deep level, or add more seed songs.")
    st.stop()

tab_cards, tab_why = st.tabs(["Gems", "How they were selected"])
with tab_cards:
    cols = st.columns(2, gap="medium")
    for n, (_, r) in enumerate(res.tracks.iterrows()):
        seed = engine.tracks.loc[int(r["best_seed"])]
        reason = (f"<b>Top {100 - r['catalog_percentile']:.2f}%</b> sound match in the catalogue · closest to your "
                  f"seed “{seed['track_name']}” · artist's best popularity here: {int(r['artist_exposure'])}")
        if r["exposure_gap"] > 0.15:
            reason += f" · sounds like music that is usually more exposed (model predicts {r['predicted_popularity']:.0f})"
        with cols[n % 2]:
            st.markdown(ui.track_card(engine, r, rank=int(r["rank"]), reason=reason, match=float(r["relevance"]),
                                      discovery=float(r["novelty"])), unsafe_allow_html=True)
            ui.track_actions(engine, r, key=f"gem_{r['idx']}", on_why=ui.go_why)

with tab_why:
    rel_all = engine.content.score_all(profile)
    rng = np.random.default_rng(0)
    top = np.argsort(-rel_all)[:6000]
    smp = rng.choice(top, 2500, replace=False)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=engine.artist_exposure[smp], y=rel_all[smp], mode="markers", name="Close-matching catalogue tracks",
                               marker=dict(size=5, color=MUTED_MARK), hoverinfo="skip"))
    g = res.tracks
    fig.add_trace(go.Scatter(x=g["artist_exposure"], y=g["relevance"], mode="markers", name="Selected gems",
                             marker=dict(size=12, color=TRACK, line=dict(color="#12111c", width=2)),
                             text=g["track_name"] + " — " + g["primary_artist"], hovertemplate="%{text}<extra></extra>"))
    fig.add_hline(y=res.relevance_floor, line=dict(color=PROFILE, dash="dash"),
                  annotation_text="relevance floor", annotation_font_color=PROFILE)
    fig.add_vline(x=c["max_artist_exposure"], line=dict(color=PROFILE, dash="dot"),
                  annotation_text="exposure limit", annotation_font_color=PROFILE)
    fig.update_layout(xaxis_title="artist exposure (best track popularity in dataset)",
                      yaxis_title="match to your profile (encoder cosine)")
    st.plotly_chart(_finish(fig, 440, "Gems live in the top-left: high match, low exposure"), width="stretch")
    st.markdown("""
1. **Score** every one of the catalogue's tracks against your profile with the SoundBridge Encoder.
2. **Relevance floor** — keep only the top X% closest tracks (level-dependent).
3. **Exposure filter** — keep tracks whose artist's best track popularity in this dataset is at or under the limit.
   Tracks with popularity 0 are excluded: in this dataset 0 usually marks a duplicate release, not obscurity.
4. **Rank** by `(1 − w) × match + w × novelty + 0.1 × exposure gap`, where the exposure gap is how much more popular
   the encoder's popularity head expects a track to be from its sound.
5. **Diversify** with one track per artist and a mild similarity penalty.
""")
ui.footer()
