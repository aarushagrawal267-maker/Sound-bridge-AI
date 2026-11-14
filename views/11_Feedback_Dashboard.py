import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.feedback import FEEDBACK_TYPES, NEGATIVE, POSITIVE, FeedbackStore, PersonalPreferenceModel, StudyStore, adjust_weights, feedback_summary
from src.hybrid_recommender import STRATEGIES, weights_from_preferences
from src.visualization import MUTED_MARK, TRACK, _finish, hbar

engine = ui.get_engine()
ui.page_header("Research", "Feedback Dashboard",
               "What listeners actually told SoundBridge, and how that feedback changes the recommendations. "
               "Everything here comes from feedback recorded on this computer — nothing is simulated.")

st.toggle("Store my feedback on this computer (consent)", key="consent",
          help="When on, feedback events are appended to data/feedback/feedback.jsonl. You can delete them below.")

store = FeedbackStore()
data = store.load()
summ = feedback_summary(data)
fb = st.session_state.fb

ui.section("This session")
ui.tiles([(f"{len(fb.events)}", "Feedback events", ""), (f"{len(fb.liked)}", "Liked / more like this", ""),
          (f"{len(fb.disliked)}", "Disliked / not my genre", ""), (f"{len(fb.known_artists)}", "Known artists flagged", ""),
          (f"{fb.variety_requests}", "Variety requests", "")])
p = st.session_state.prefs
base = weights_from_preferences(p["familiarity"], p["discovery"], p["exploration"], p["diversity"])
adj = adjust_weights(base, fb)
c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("**How feedback changed your ranking weights**")
    wdf = pd.DataFrame({"weight": ["relevance", "novelty", "diversity", "max per artist"],
                        "from preferences": [base.relevance, base.novelty, base.diversity, base.max_per_artist],
                        "after feedback": [adj.relevance, adj.novelty, adj.diversity, adj.max_per_artist]})
    st.dataframe(wdf, hide_index=True, width="stretch")
    if fb.penalised_genres:
        st.caption("Down-weighted genres: " + ", ".join(fb.penalised_genres))
with c2:
    pm = PersonalPreferenceModel(engine.cfg["feedback"]["min_labels_for_personal_model"])
    ready = pm.fit(engine.embeddings, fb.liked, fb.disliked)
    st.markdown("**Personal preference layer**")
    st.progress(min(pm.n_labels / pm.min_labels, 1.0),
                text=f"{pm.n_labels} / {pm.min_labels} like-dislike labels" + (" — active" if ready else ""))
    st.caption("A logistic-regression layer on the encoder's embeddings, trained only on your own labels. It switches on "
               "once there are enough labels of both kinds and then nudges candidate scores.")

ui.section("Stored feedback", f"consented events · stored in {store.backend}")
if summ["n_events"] == 0:
    ui.empty_state("No stored feedback yet", "Turn on consent and rate some recommendations. This dashboard will then "
                   "analyse the real events — it never shows invented data.", "🗂️")
else:
    ui.tiles([(f"{summ['n_events']}", "Events", ""), (f"{summ['n_sessions']}", "Sessions", "anonymous session ids"),
              (f"{summ.get('like_rate', float('nan')):.0%}" if "like_rate" in summ else "—", "Like rate", "likes ÷ (likes + dislikes)")])
    c3, c4 = st.columns(2, gap="large")
    counts = pd.Series(summ["counts"]).rename(index=FEEDBACK_TYPES).sort_values(ascending=False)
    c3.plotly_chart(hbar(counts.index, counts.values, "Feedback types", height=320), width="stretch")
    if "like_rate_by_strategy" in summ:
        lr = summ["like_rate_by_strategy"]
        lr["label"] = lr["strategy"].map(lambda s: STRATEGIES.get(str(s).split("@")[0].split(":")[0], s))
        c4.plotly_chart(hbar(lr["label"] + " (n=" + lr["n"].astype(str) + ")", lr["like_rate"], "Like rate by strategy",
                             height=320, fmt=".0%"), width="stretch")
    rated = data[data["feedback"].isin(POSITIVE | NEGATIVE)].copy()
    if len(rated):
        rated["outcome"] = rated["feedback"].isin(POSITIVE).map({True: "Liked", False: "Disliked"})
        fig = go.Figure()
        for o, color in [("Liked", TRACK), ("Disliked", MUTED_MARK)]:
            fig.add_trace(go.Box(y=rated.loc[rated.outcome == o, "popularity"], name=o, marker_color=color, boxpoints="all",
                                 jitter=0.4, pointpos=0, line=dict(color=color)))
        fig.update_layout(yaxis=dict(title="track popularity", range=[-2, 102]), showlegend=False)
        st.plotly_chart(_finish(fig, 320, "Discovery satisfaction: popularity of liked vs disliked tracks"), width="stretch")
        lk = rated[(rated["popularity"] > 0) & (rated["popularity"] <= engine.lesser_known)]
        if len(lk):
            st.caption(f"Lesser-known tracks rated: {len(lk)} · liked: {lk['feedback'].isin(POSITIVE).mean():.0%}")
    if not ui.admin_only("Viewing, exporting and deleting raw events"):
        st.stop()
    with st.expander("Raw events"):
        st.dataframe(data, width="stretch")
    c5, c6 = st.columns(2)
    c5.download_button("Export feedback (CSV)", data.to_csv(index=False), "soundbridge_feedback.csv", icon=":material/download:",
                       width="stretch")
    if c6.button("Delete all stored feedback", icon=":material/delete:", width="stretch"):
        store.clear()
        StudyStore().clear()
        st.success("Stored feedback deleted.")
        st.rerun()
st.caption("SoundBridge does not learn from a population of users: it adapts to the person using it, from the labels shown here.")
ui.footer()
