import streamlit as st

from src import ui
from src.hidden_gems import find_hidden_gems
from src.hybrid_recommender import weights_from_preferences
from src.musical_dna import feature_summary
from src.visualization import radar

engine = ui.get_engine()
ev = engine.evaluation
ret = ev.get("retrieval_held_out_artists", {})
enc = ret.get("SoundBridge Encoder (learned, 32-d)", {})
raw = ret.get("Raw audio features (z-score, 9-d)", {})

st.markdown(
    f"""<div class="sb-hero"><div>
  <span class="eyebrow">● AI music discovery · {len(engine):,} real tracks</span>
  <h1>Discover beyond<br><span class="sb-grad">your algorithm.</span></h1>
  <p class="lead">SoundBridge AI learns the sound of the songs you love with its own neural network, then bridges you
  to music you haven't heard — including lesser-known artists — and tells you exactly why each song was chosen.</p>
  </div>{ui.hero_eq()}
</div>""",
    unsafe_allow_html=True,
)

lift = (enc.get("artist_precision@10", 0) / raw["artist_precision@10"]) if raw.get("artist_precision@10") else None
ui.tiles([
    (f"{len(engine):,}", "Tracks", "after de-duplication"),
    (f"{engine.tracks['primary_artist'].nunique():,}", "Artists", "primary credited artists"),
    (f"{len(engine.genres)}", "Genre labels", "from the source dataset"),
    (f"{lift:.1f}×" if lift else "—", "Encoder vs raw features", "same-artist precision@10 on unseen artists"),
])

seeds = st.session_state.seeds

# ---------------------------------------------------------------- search
ui.section("Start with songs you love", "search the catalogue or try a ready-made taste")
q = st.text_input("Search tracks or artists", placeholder="e.g. Radiohead, Daft Punk, Bon Iver…", label_visibility="collapsed")
if q:
    res = engine.search(q, limit=6)
    if res.empty:
        st.caption("No matches in this dataset — it is a sample of Spotify, not the full catalogue.")
    for _, r in res.iterrows():
        c1, c2 = st.columns([8, 1.3], vertical_alignment="center")
        c1.markdown(ui.track_card(engine, r), unsafe_allow_html=True)
        if r["idx"] in seeds:
            c2.button("Added", key=f"h_add_{r['idx']}", disabled=True, icon=":material/check:", width="stretch")
        elif c2.button("Add", key=f"h_add_{r['idx']}", icon=":material/add:", type="primary", width="stretch"):
            ui.add_seed(int(r["idx"]))
            st.rerun()

cols = st.columns(4)
for col, (key, label) in zip(cols, [("indie", "Indie & alt-rock"), ("electronic", "Electronic"),
                                     ("acoustic", "Acoustic & folk"), ("hiphop", "Hip-hop")]):
    if col.button(label, key=f"demo_{key}", icon=":material/auto_awesome:", width="stretch"):
        found = ui.demo_seeds(engine, key)
        for i in found:
            ui.add_seed(i)
        st.rerun()

if not seeds:
    ui.section("How SoundBridge works")
    c = st.columns(3)
    steps = [
        ("1", "Pick seed songs", "Choose a few tracks you already enjoy. Their audio features become your Musical DNA."),
        ("2", "The encoder listens", "A neural network trained on 79k tracks maps every song into a 32-dimensional sound space."),
        ("3", "Bridge outward", "Hybrid ranking balances match, novelty, diversity and lesser-known artists — with reasons."),
    ]
    for col, (n, t, b) in zip(c, steps):
        col.markdown(f'<div class="sb-panel"><div class="sb-step"><div class="n">{n}</div><div><h4>{t}</h4><p>{b}</p></div></div></div>',
                     unsafe_allow_html=True)
    ui.footer()
    st.stop()

# ---------------------------------------------------------------- profile + recommendations
profile = ui.current_profile(engine)
left, right = st.columns([1, 1.35], gap="large")
with left:
    ui.section("Your Musical DNA", f"{len(seeds)} seed songs")
    summ = feature_summary(engine.tracks, seeds)
    fig = radar(dict(zip(summ.feature, summ.display_mean)), baseline=dict(zip(summ.feature, summ.display_catalog)), height=360)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    if st.button("Explore your full Musical DNA", icon=":material/genetics:", width="stretch"):
        st.switch_page("views/2_Musical_DNA.py")
with right:
    ui.section("Recommended discoveries", "hybrid ranking · your current preferences")
    p = st.session_state.prefs
    w = weights_from_preferences(p["familiarity"], p["discovery"], p["exploration"], p["diversity"])
    recs = ui.recommend_now(engine, "hybrid", w, k=5)
    for _, r in recs.iterrows():
        st.markdown(ui.track_card(engine, r, rank=int(r["rank"]), reason=ui.short_reason_for(engine, r, seeds),
                                  match=float(r["relevance"]), discovery=float(r["discovery_score"])),
                    unsafe_allow_html=True)
    if st.button("Open the full discovery experience", icon=":material/travel_explore:", type="primary", width="stretch"):
        st.switch_page("views/1_Music_Discovery.py")

ui.section("Hidden artist spotlight", "relevant tracks by artists with low exposure in this dataset")
gems = find_hidden_gems(engine, profile, "balanced", k=3)
if gems.tracks.empty:
    st.caption("No lesser-known tracks passed the relevance floor for this profile — try the Deep level on Hidden Gems.")
else:
    cols = st.columns(3)
    for col, (_, r) in zip(cols, gems.tracks.iterrows()):
        col.markdown(ui.track_card(engine, r, reason=f"Top {100 - r['catalog_percentile']:.2f}% match · artist's best popularity {int(r['artist_exposure'])}",
                                   match=float(r["relevance"])), unsafe_allow_html=True)
c1, c2 = st.columns(2)
if c1.button("See all Hidden Gems", icon=":material/diamond:", width="stretch"):
    st.switch_page("views/3_Hidden_Gems.py")
if c2.button("Open the Recommendation Engine", icon=":material/neurology:", width="stretch"):
    st.switch_page("views/6_Recommendation_Engine.py")
ui.footer()
