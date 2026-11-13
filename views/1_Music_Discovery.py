import streamlit as st

from src import ui
from src.feedback import adjust_weights
from src.hybrid_recommender import weights_from_preferences
from src.visualization import sound_map

engine = ui.get_engine()
ui.page_header("Your music discovery", "Music Discovery",
               "Search the catalogue, add songs you already enjoy, tune how adventurous you feel — the ranking "
               "recalculates instantly and every pick comes with its reasons.")

seeds = st.session_state.seeds
left, right = st.columns([1.25, 1], gap="large")

# ---------------------------------------------------------------- search + seeds
with left:
    ui.section("1 · Find songs you enjoy")
    c1, c2 = st.columns([2, 1])
    q = c1.text_input("Search", placeholder="Track or artist name…", label_visibility="collapsed", key="disc_q")
    genre = c2.selectbox("Genre", ["All genres"] + engine.genres, label_visibility="collapsed", key="disc_genre")
    if q or genre != "All genres":
        res = engine.search(q, limit=8, genre=genre)
        if res.empty:
            st.caption("No matches in this dataset sample.")
        for _, r in res.iterrows():
            a, b = st.columns([8, 1.4], vertical_alignment="center")
            a.markdown(ui.track_card(engine, r), unsafe_allow_html=True)
            if r["idx"] in seeds:
                if b.button("Remove", key=f"d_rm_{r['idx']}", icon=":material/close:", width="stretch"):
                    ui.remove_seed(int(r["idx"]))
                    st.rerun()
            elif b.button("Add", key=f"d_add_{r['idx']}", icon=":material/add:", type="primary", width="stretch"):
                ui.add_seed(int(r["idx"]))
                st.rerun()
    else:
        st.caption("Type a track or artist, or pick a genre to browse its most popular tracks.")

    ui.section("2 · Your seed songs", f"{len(seeds)} selected")
    if not seeds:
        ui.empty_state("No seeds yet", "Add at least one song. Three to five varied favourites give the richest profile.")
    for i in list(seeds):
        a, b = st.columns([8, 1.4], vertical_alignment="center")
        a.markdown(ui.seed_pill(engine, i), unsafe_allow_html=True)
        if b.button("", key=f"d_seed_rm_{i}", icon=":material/close:", help="Remove seed", width="stretch"):
            ui.remove_seed(i)
            st.rerun()

# ---------------------------------------------------------------- preferences
with right:
    ui.section("3 · Discovery preferences", "re-ranks live")
    p = st.session_state.prefs
    for name in p:
        st.session_state.setdefault(f"pref_{name}", p[name])
    with st.container(border=True):
        p["familiarity"] = st.slider("Musical familiarity", 0.0, 1.0, step=0.05, key="pref_familiarity",
                                     help="How strongly to favour the closest sound match to your seeds.")
        p["discovery"] = st.slider("Lesser-known artist discovery", 0.0, 1.0, step=0.05, key="pref_discovery",
                                   help="Weight on artists with lower exposure in this dataset (popularity is an exposure proxy, not quality).")
        p["exploration"] = st.slider("Genre exploration", 0.0, 1.0, step=0.05, key="pref_exploration",
                                     help="0 keeps to your seeds' genres; 1 favours genres the model finds sonically related but new to you.")
        p["diversity"] = st.slider("Recommendation diversity", 0.0, 1.0, step=0.05, key="pref_diversity",
                                   help="Penalises tracks that sound like ones already in the list and repeated artists.")
        k = st.select_slider("How many recommendations", [6, 9, 12, 15, 20], value=12)
    st.toggle("Store my feedback on this computer (consent)", key="consent",
              help="Feedback always adapts this session. With consent it is also saved to data/feedback/ for the "
                   "Feedback Dashboard. Nothing leaves this machine.")

if not seeds:
    ui.footer()
    st.stop()

# ---------------------------------------------------------------- recommendations
base = weights_from_preferences(p["familiarity"], p["discovery"], p["exploration"], p["diversity"])
w = adjust_weights(base, st.session_state.fb)
recs = ui.recommend_now(engine, "hybrid", w, k=k)
st.session_state.last_recs = recs

ui.section("Your recommendations", "SoundBridge hybrid ranking")
fb = st.session_state.fb
if fb.events:
    msg = [f"profile adapted from {len(fb.liked)} liked / {len(fb.disliked)} disliked tracks"]
    if fb.penalised_genres:
        msg.append("down-weighting " + ", ".join(fb.penalised_genres))
    if w.as_dict() != base.as_dict():
        msg.append(f"weights adjusted (novelty {base.novelty}→{w.novelty}, diversity {base.diversity}→{w.diversity}, "
                   f"relevance {base.relevance}→{w.relevance})")
    ui.note("Feedback active: " + "; ".join(msg) + ".")

tab_list, tab_map, tab_table = st.tabs(["Cards", "Sound map", "Table & weights"])
with tab_list:
    cols = st.columns(2, gap="medium")
    for n, (_, r) in enumerate(recs.iterrows()):
        with cols[n % 2]:
            st.markdown(ui.track_card(engine, r, rank=int(r["rank"]), reason=ui.short_reason_for(engine, r, seeds),
                                      match=float(r["relevance"]), discovery=float(r["discovery_score"])),
                        unsafe_allow_html=True)
            ui.track_actions(engine, r, key=f"disc_{r['idx']}", weights=w.as_dict(), on_why=ui.go_why)
with tab_map:
    labels = engine.tracks["track_name"] + " — " + engine.tracks["primary_artist"]
    st.plotly_chart(sound_map(engine.coords, labels, recs["idx"].to_numpy(), seeds,
                              title="Where your recommendations sit in the encoder's sound space (PCA projection)"),
                    width="stretch")
    st.caption("Grey points: a random sample of the catalogue. The 2-D projection keeps only part of the 32-D "
               "embedding's structure, so nearby points are similar but distant ones are not necessarily dissimilar.")
with tab_table:
    show = recs[["rank", "track_name", "artists", "album_name", "genre", "popularity", "relevance", "raw_similarity",
                 "novelty", "discovery_score", "diversity", "genre_score", "score"]]
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={
                     "relevance": st.column_config.ProgressColumn("Match (encoder)", min_value=0, max_value=1, format="%.2f"),
                     "discovery_score": st.column_config.ProgressColumn("Discovery", min_value=0, max_value=1, format="%.2f"),
                     "popularity": st.column_config.NumberColumn("Popularity"),
                     "raw_similarity": st.column_config.NumberColumn("Raw-feature cosine", format="%.2f"),
                 })
    st.json({"weights_used": w.as_dict(),
             "discovery_score": "0.7 × novelty (exposure-based) + 0.3 × (1 − genre affinity)",
             "match": "0.5 × cos(track, profile centroid) + 0.5 × max cos(track, any seed), encoder space"})
ui.footer()
