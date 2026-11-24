"""SoundBridge AI — Discover Beyond Your Algorithm.

Launch:  streamlit run app.py
"""
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
ICON = ROOT / "assets" / "icon.png"

st.set_page_config(
    page_title="SoundBridge AI",
    page_icon=str(ICON) if ICON.exists() else "🎧",
    layout="wide",
    initial_sidebar_state="auto",
)

from src.engine import ArtifactsMissing  # noqa: E402
from src.ui import get_engine, setup_page, sidebar  # noqa: E402

P = "views/"
pages = {
    "Discover": [
        st.Page(P + "0_Home.py", title="Home", icon=":material/home:", default=True),
        st.Page(P + "1_Music_Discovery.py", title="Music Discovery", icon=":material/travel_explore:"),
        st.Page(P + "2_Musical_DNA.py", title="Musical DNA", icon=":material/genetics:"),
        st.Page(P + "3_Hidden_Gems.py", title="Hidden Gems", icon=":material/diamond:"),
        st.Page(P + "10_Listening_Preferences.py", title="Listening Preferences", icon=":material/headphones:"),
        st.Page(P + "9_Why_This_Song.py", title="Why This Song?", icon=":material/psychology:"),
    ],
    "Explore the catalogue": [
        st.Page(P + "4_Genre_Explorer.py", title="Genre Explorer", icon=":material/category:"),
        st.Page(P + "5_Artist_Opportunity_Explorer.py", title="Artist Opportunity Explorer", icon=":material/mic_external_on:"),
        st.Page(P + "13_Dataset_Explorer.py", title="Dataset Explorer", icon=":material/dataset:"),
    ],
    "Research": [
        st.Page(P + "6_Recommendation_Engine.py", title="Recommendation Engine", icon=":material/neurology:"),
        st.Page(P + "7_Experiment_Lab.py", title="Experiment Lab", icon=":material/science:"),
        st.Page(P + "8_Recommendation_Evaluation.py", title="Recommendation Evaluation", icon=":material/query_stats:"),
        st.Page(P + "14_Pilot_Study.py", title="Pilot Study", icon=":material/groups:"),
        st.Page(P + "11_Feedback_Dashboard.py", title="Feedback Dashboard", icon=":material/insights:"),
    ],
    "About": [
        st.Page(P + "12_About.py", title="About SoundBridge AI", icon=":material/info:"),
    ],
}

nav = st.navigation(pages, expanded=True)
setup_page()

try:
    engine = get_engine()
except ArtifactsMissing as exc:
    import subprocess
    import sys

    st.markdown(
        '<div class="sb-hero"><div><span class="eyebrow">● First-time setup</span><h1>Let\'s build your<br>'
        '<span class="sb-grad">SoundBridge.</span></h1><p class="lead">SoundBridge needs to download the public Spotify '
        "tracks dataset, clean it, load its trained neural network and index the catalogue. This runs automatically, "
        "once, and takes a few minutes. (Run <code>python scripts/setup_all.py</code> to retrain the network from scratch.)</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.caption(str(exc))
    if st.session_state.get("_auto_build_ran"):
        # a build already finished in this session but the artifacts are still unusable: do not loop
        st.error("Automatic setup finished but the model artifacts are still not usable. Run the command "
                 "below in a terminal and check its output.")
        st.code("python scripts/setup_all.py", language="bash")
        st.stop()
    import time

    lock = ROOT / "data" / "processed" / ".building"
    if lock.exists() and time.time() - lock.stat().st_mtime < 1200:
        # another visitor already started the build: wait for it
        with st.spinner("SoundBridge is being set up by another visitor — this page refreshes when ready…"):
            while lock.exists() and time.time() - lock.stat().st_mtime < 1200:
                time.sleep(5)
        get_engine.clear()
        st.rerun()
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.touch()
    try:
        with st.status("Building SoundBridge AI (first start only, a few minutes)…", expanded=True) as status:
            line_box = st.empty()
            proc = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "setup_all.py"), "--reuse-weights"], cwd=ROOT,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                if line.strip():
                    line_box.code(line.strip())
                    lock.touch()
            ok = proc.wait() == 0
    finally:
        lock.unlink(missing_ok=True)
    if ok:
        st.session_state._auto_build_ran = True
        status.update(label="SoundBridge is ready", state="complete")
        get_engine.clear()
        st.rerun()
    status.update(label="Setup failed — see the last message above", state="error")
    st.code("python scripts/setup_all.py", language="bash")
    st.stop()

demo = st.query_params.get("demo")
if demo and not st.session_state.seeds and not st.session_state.get("_demo_loaded"):
    from src.ui import demo_seeds

    st.session_state.seeds = demo_seeds(engine, demo)
    st.session_state._demo_loaded = True

sidebar(engine)
nav.run()
