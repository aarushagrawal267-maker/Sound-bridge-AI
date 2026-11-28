"""Pilot user study: a blind A/B comparison of a baseline and the SoundBridge hybrid list."""
import time

import numpy as np
import pandas as pd
import streamlit as st

from src import ui
from src.config import path
from src.feedback import StudyStore
from src.hybrid_recommender import weights_from_preferences
from src.visualization import hbar

engine = ui.get_engine()
ui.page_header("Real-world validation", "Pilot Study",
               "A structured, consented user test: participants pick songs they enjoy, rate two anonymised "
               "recommendation lists, flag unfamiliar artists and repetition, and say which list they prefer.")

CONFIGS = {"content": "Baseline: content-based (encoder, relevance only)", "hybrid": "SoundBridge hybrid (balanced)"}
store = StudyStore()
tab_run, tab_report, tab_protocol = st.tabs(["Take part", "Results report", "Protocol"])

with tab_protocol:
    st.markdown("""
**Design** — within-subjects, blind A/B. Each participant sees two lists of 8 tracks generated from their own seeds:
a relevance-only content-based list and the SoundBridge hybrid list. Which one is shown as "List A" is randomised per
participant, and the configuration names are hidden until the study is submitted.

**Measures** (per list): perceived relevance (1–5), perceived variety (1–5), number of artists the participant did not
know, whether the list felt repetitive, usefulness of explanations (1–5, shown after rating); then an overall preference.

**Target** — 10–20 consenting participants. **Analysis** — means with bootstrap 95% intervals and a paired comparison.
With a sample this small the results are descriptive; no statistical significance is claimed.

**Data** — stored locally in `data/feedback/study_responses.jsonl`, anonymous (random id, no names or contact details).
""")

with tab_run:
    seeds = st.session_state.seeds
    if not st.session_state.get("study_consent"):
        st.markdown("#### Consent")
        st.markdown("Your answers are stored on this computer only, without your name. You can stop at any time; "
                    "nothing is stored until you submit.")
        if st.checkbox("I agree to take part and for my anonymous answers to be stored locally."):
            st.session_state.study_consent = True
            st.rerun()
        st.stop()
    if len(seeds) < 2:
        ui.empty_state("Step 1 — choose at least two songs you enjoy",
                       "Add seeds on Music Discovery (search for songs you genuinely like), then return here.", "music_note")
        if st.button("Go to Music Discovery", type="primary"):
            st.switch_page("views/1_Music_Discovery.py")
        st.stop()

    if "study_order" not in st.session_state:
        st.session_state.study_order = list(np.random.default_rng().permutation(list(CONFIGS)))
        st.session_state.study_id = st.session_state.session_id + f"-{int(time.time()) % 100000}"
    order = st.session_state.study_order
    lists = {}
    w = weights_from_preferences(0.6, 0.4, 0.1, 0.5)
    for cfg in CONFIGS:
        lists[cfg] = ui.recommend_now(engine, cfg, w, k=8)

    st.markdown("#### Step 2 — rate two lists")
    st.caption("Listen via the Spotify links if you like. Tick each artist you did not know before.")
    answers = {}
    with st.form("study_form"):
        cols = st.columns(2, gap="large")
        for col, cfg, label in zip(cols, order, ["List A", "List B"]):
            with col:
                st.markdown(f"### {label}")
                for _, r in lists[cfg].iterrows():
                    st.markdown(ui.track_card(engine, r), unsafe_allow_html=True)
                    st.checkbox("I didn't know this artist", key=f"unk_{label}_{r['idx']}")
                answers[cfg] = {
                    "label": label,
                    "relevance": st.slider(f"{label}: how well does it match your taste?", 1, 5, 3, key=f"rel_{label}"),
                    "variety": st.slider(f"{label}: how varied is it?", 1, 5, 3, key=f"var_{label}"),
                    "repetitive": st.radio(f"{label}: did it feel repetitive?", ["No", "Somewhat", "Yes"], horizontal=True, key=f"rep_{label}"),
                    "unknown_artists_key": label,
                }
        pref = st.radio("Overall, which list would you rather listen to?", ["List A", "List B", "No preference"], horizontal=True)
        expl = st.slider("After seeing explanations on 'Why This Song?', how useful were they? (1–5; leave at 3 if not viewed)", 1, 5, 3)
        comment = st.text_area("Anything else? (optional, please don't include personal details)")
        submitted = st.form_submit_button("Submit my answers", type="primary")
    if submitted:
        for cfg, a in answers.items():
            label = a["label"]
            a["unknown_artists"] = sum(bool(st.session_state.get(f"unk_{label}_{i}")) for i in lists[cfg]["idx"])
            a.pop("unknown_artists_key")
        preferred = {"List A": order[0], "List B": order[1]}.get(pref, "none")
        store.append({"type": "pilot_response", "participant": st.session_state.study_id, "timestamp": time.time(),
                      "n_seeds": len(seeds), "order": order, "lists": answers, "preferred": preferred,
                      "explanation_usefulness": expl, "comment": comment[:500]})
        st.success(f"Thanks for taking part. List A was **{CONFIGS[order[0]]}** and List B was **{CONFIGS[order[1]]}**.")
        del st.session_state["study_order"]

with tab_report:
    data = store.load()
    resp = data[data.get("type", pd.Series(dtype=str)) == "pilot_response"] if len(data) else pd.DataFrame()
    n = len(resp)
    ui.tiles([(f"{n}", "Participants", "target 10–20"), ("Descriptive", "Analysis", "no significance claimed")])
    if n == 0:
        ui.empty_state("No pilot responses recorded yet",
                       "The study has not been run. Results will appear here as real participants submit — nothing is "
                       "pre-filled or simulated.", "science")
    else:
        rows = []
        for _, r in resp.iterrows():
            for cfg, a in r["lists"].items():
                rows.append({"participant": r["participant"], "config": cfg, "relevance": a["relevance"], "variety": a["variety"],
                             "unknown_artists": a["unknown_artists"], "repetitive": a["repetitive"] == "Yes"})
        long = pd.DataFrame(rows)
        rng = np.random.default_rng(0)

        def boot(x):
            x = np.asarray(x, dtype=float)
            if len(x) < 2:
                return (np.nan, np.nan)
            b = [rng.choice(x, len(x)).mean() for _ in range(2000)]
            return np.percentile(b, [2.5, 97.5])

        table = []
        for cfg, g in long.groupby("config"):
            for m in ["relevance", "variety", "unknown_artists", "repetitive"]:
                lo, hi = boot(g[m])
                table.append({"configuration": CONFIGS.get(cfg, cfg), "measure": m, "mean": g[m].mean(), "ci95_low": lo, "ci95_high": hi, "n": len(g)})
        tdf = pd.DataFrame(table)
        st.dataframe(tdf.round(2), hide_index=True, width="stretch")
        prefs = resp["preferred"].map(lambda c: CONFIGS.get(c, "No preference")).value_counts()
        st.plotly_chart(hbar(prefs.index, prefs.values, "Preferred list", height=240), width="stretch")
        st.metric("Mean explanation usefulness (1–5)", f"{resp['explanation_usefulness'].mean():.2f}")
        report = ["# SoundBridge AI pilot study report", "", f"Participants: {n} (target 10–20).",
                  "Design: blind within-subjects A/B, randomised order.", "",
                  "Results are descriptive; the sample is too small for significance testing.", "",
                  "```", tdf.round(2).to_csv(index=False), "```", "", "Preferred list:",
                  *[f"* {k}: {v}" for k, v in prefs.items()], "",
                  f"Mean explanation usefulness: {resp['explanation_usefulness'].mean():.2f} / 5"]
        text = "\n".join(report)
        (path("outputs_dir") / "reports" / "pilot_study_report.md").write_text(text)
        st.download_button("Download pilot report", text, "pilot_study_report.md", icon=":material/download:")
ui.footer()
