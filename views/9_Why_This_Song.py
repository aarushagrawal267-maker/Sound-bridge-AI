import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from src import ui
from src.explainability import build_explanation
from src.feature_engineering import display_scale
from src.feedback import StudyStore
from src.spotify_integration import SpotifyClient, credentials_available, embed_url, track_url
from src.visualization import attribution_bars, comparison_bars, hbar, radar

engine = ui.get_engine()
ui.page_header("Explainable recommendations", "Why This Song?",
               "Every recommendation is explained from real numbers: how its audio features compare with your Musical "
               "DNA, what drives the neural network's match, and which ranking objective lifted it into your list.")

seeds = st.session_state.seeds
if not seeds:
    ui.empty_state("No profile to explain against", "Add seed songs first — explanations compare a track with your Musical DNA.", "search")
    st.stop()

# choose a track: last recommendations, or search
options = []
recs = st.session_state.get("last_recs")
if recs is not None and len(recs):
    options = [int(i) for i in recs["idx"]]
if st.session_state.why_track is not None and st.session_state.why_track not in options:
    options = [st.session_state.why_track] + options
c1, c2 = st.columns([2, 1])
q = c2.text_input("…or search any track", placeholder="track or artist")
if q:
    options = [int(i) for i in engine.search(q, 10)["idx"]] + options
if not options:
    profile = ui.current_profile(engine)
    options = [int(i) for i in ui.recommend_now(engine, "hybrid", None, k=10)["idx"]]
default = options.index(st.session_state.why_track) if st.session_state.why_track in options else 0
idx = c1.selectbox("Track to explain", options, index=default, format_func=engine.label)
st.session_state.why_track = idx

profile = ui.current_profile(engine)
row = engine.tracks.loc[idx]
rel = float(0.5 * engine.embeddings[idx] @ profile.centroid + 0.5 * (profile.seed_embeddings @ engine.embeddings[idx]).max())
closest = profile.seed_idx[int(np.argmax(engine.embeddings[profile.seed_idx] @ engine.embeddings[idx]))]
sg = engine.seed_genres(seeds)
rec_row = recs[recs["idx"] == idx].iloc[0] if recs is not None and (recs["idx"] == idx).any() else None
attr = engine.attribution(idx, profile.centroid)
exp = build_explanation(
    tracks=engine.tracks, seed_idx=seeds, track_idx=idx, catalog_std=engine.catalog_std,
    raw_track=engine.raw_z[idx], raw_profile=engine.raw_profile(seeds), raw_names=engine.dna_features,
    relevance=rel, seed_genres=sg, related_genre=engine.related_genre_for(idx, sg),
    novelty_component=float(rec_row["novelty"]) * 0.3 if rec_row is not None and "score" in rec_row else 0.0,
    diversity_component=float(rec_row.get("diversity", 0) or 0) * 0.2 if rec_row is not None else 0.0,
    lesser_known_threshold=engine.lesser_known,
    discovery_mode=bool(engine.exposure_known[idx] and engine.artist_exposure[idx] <= engine.lesser_known),
    attribution=attr, closest_seed=closest,
)

left, right = st.columns([1.05, 1], gap="large")
with left:
    art = None
    if credentials_available():
        art = SpotifyClient().album_art(str(row["track_id"]))
    st.markdown(ui.track_card(engine, row, match=rel, discovery=float(engine.novelty[idx]), art_url=art), unsafe_allow_html=True)
    st.markdown(f"### {exp.headline}")
    st.markdown('<ul class="sb-reasons">' + "".join(f"<li>{r}</li>" for r in exp.reasons) + "</ul>", unsafe_allow_html=True)
    if rec_row is not None and "score" in rec_row and not np.isnan(rec_row.get("score", np.nan)):
        st.caption(f"Ranking context: rank #{int(rec_row['rank'])} by {rec_row['strategy']} · "
                   f"match {rec_row['relevance']:.2f} · novelty {rec_row['novelty']:.2f} · "
                   f"diversity gain {rec_row.get('diversity', 0):.2f} · genre term {rec_row.get('genre_score', 0):.2f}")
    st.toggle("Load Spotify's official player for this track", key="embeds",
              help="Loads Spotify's embeddable widget (artwork and any preview Spotify offers). Contacts Spotify only when on.")
    if st.session_state.embeds:
        components.iframe(embed_url(str(row["track_id"])), height=152)
    st.link_button("Open in Spotify", track_url(str(row["track_id"])), icon=":material/open_in_new:")
with right:
    feats = engine.dna_features
    prof_disp = display_scale(engine.tracks.loc[seeds], feats).mean()
    tr_disp = display_scale(engine.tracks.loc[[idx]], feats).iloc[0]
    st.plotly_chart(radar(prof_disp.to_dict(), compare=tr_disp.to_dict(), compare_name=str(row["track_name"])[:28], height=420),
                    width="stretch", config={"displayModeBar": False})

t1, t2, t3, t4 = st.tabs(["Feature comparison", "Similarity breakdown", "Neural network attribution", "Model's view"])
with t1:
    st.plotly_chart(comparison_bars(exp.comparison), width="stretch")
    st.caption("Green band: within 0.35 standard deviations of your profile (counted as 'similar'). Values in "
               "catalogue SDs so tempo and loudness are comparable with 0–1 features.")
    st.dataframe(exp.comparison[["label", "your_profile", "this_track", "difference", "z_difference", "closeness"]],
                 hide_index=True, width="stretch")
with t2:
    contrib = exp.contributions.sort_values("contribution", ascending=False)
    st.plotly_chart(hbar(contrib["label"], contrib["contribution"], "Per-feature terms of the raw-feature cosine similarity",
                         height=360, fmt=".3f"), width="stretch")
    st.caption(f"Exact decomposition: the terms sum to the cosine similarity in z-scored audio-feature space "
               f"({contrib['contribution'].sum():.3f}). Positive terms pull the track towards your profile.")
with t3:
    st.plotly_chart(attribution_bars(attr), width="stretch")
    st.caption("Integrated gradients of the encoder's similarity (track embedding · your profile embedding) with respect "
               "to each input feature, measured from the catalogue's median track. Positive bars increase the match.")
with t4:
    pg = engine.predicted_genres(idx, 5)
    c3, c4 = st.columns(2)
    c3.plotly_chart(hbar([g for g, _ in pg], [p for _, p in pg], "Genre head: predicted labels", height=280, fmt=".0%"),
                    width="stretch")
    c4.metric("Actual popularity", int(row["popularity"]))
    c4.metric("Predicted from sound (popularity head)", f"{engine.pred_popularity[idx]:.0f}",
              help="Low R²: popularity depends mostly on factors other than sound.")
    c4.metric("Artist exposure in dataset", int(engine.artist_exposure[idx]))
    c4.caption("Dataset labels: " + ", ".join(row["genre"]))

ui.section("Was this explanation useful?", "your rating helps evaluate the explanations")
with st.form("expl_rating", border=True):
    rating = st.slider("Usefulness (1 = not useful, 5 = very useful)", 1, 5, 4)
    understood = st.radio("Did you understand why this song was recommended?", ["Yes", "Partly", "No"], horizontal=True)
    if st.form_submit_button("Submit rating", type="primary"):
        if st.session_state.consent:
            StudyStore().append({"type": "explanation_rating", "session_id": st.session_state.session_id, "track_idx": idx,
                                 "usefulness": rating, "understood": understood})
            st.success("Thanks — rating stored locally.")
        else:
            st.info("Rating not stored: switch on feedback consent on Music Discovery or the Feedback Dashboard first.")
ui.footer()
