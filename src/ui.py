"""SoundBridge design system for Streamlit: CSS, generative artwork, cards and shared state."""
from __future__ import annotations

import hashlib
import html
from typing import Callable

import numpy as np
import pandas as pd
import streamlit as st

from .visualization import gauge_ring

BRAND = "#8b5cf6"
BLUE = "#38bdf8"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{
  --bg:#09080f; --surface:#12111c; --surface-2:#191726; --line:rgba(255,255,255,.08); --line-2:rgba(255,255,255,.14);
  --text:#ECEAF6; --text-2:#B9B5CF; --muted:#7D7896; --violet:#8b5cf6; --violet-2:#a78bfa; --blue:#38bdf8; --aqua:#199e70;
  --pink:#e879f9; --grad:linear-gradient(120deg,#a78bfa 0%,#8b5cf6 30%,#38bdf8 75%,#67e8f9 100%);
}
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: var(--bg); }
.stApp::before{ content:""; position:fixed; inset:-20%; z-index:0; pointer-events:none;
  background:
    radial-gradient(40% 35% at 15% 10%, rgba(139,92,246,.20), transparent 70%),
    radial-gradient(35% 30% at 90% 15%, rgba(56,189,248,.14), transparent 70%),
    radial-gradient(45% 40% at 70% 95%, rgba(232,121,249,.08), transparent 70%);
  animation: sbDrift 28s ease-in-out infinite alternate; }
@keyframes sbDrift { 0%{transform:translate3d(0,0,0) scale(1)} 100%{transform:translate3d(-3%,2%,0) scale(1.06)} }
[data-testid="stAppViewContainer"] > .main, [data-testid="stMain"] { position:relative; z-index:1; }
[data-testid="stHeader"]{ background: transparent; }
[data-testid="stMainBlockContainer"], .block-container { padding-top: 2.2rem; max-width: 1280px; }
h1,h2,h3,h4 { font-family:'Space Grotesk', Inter, sans-serif !important; letter-spacing:-.02em; color:var(--text); }
p, li { color: var(--text-2); }
a { color: var(--blue); }
hr { border-color: var(--line) !important; }

/* sidebar */
[data-testid="stSidebar"]{ background: linear-gradient(180deg, rgba(20,18,32,.96), rgba(10,9,16,.98)); border-right:1px solid var(--line); }
[data-testid="stSidebar"] [data-testid="stSidebarNavItems"] a, [data-testid="stSidebarNav"] a { border-radius:10px; transition: background .2s ease; }
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover { background: rgba(139,92,246,.12); }
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] { background: linear-gradient(90deg, rgba(139,92,246,.28), rgba(56,189,248,.10)); }
[data-testid="stSidebarNavSeparator"]{ border-color: var(--line); }
[data-testid="stNavSectionHeader"], [data-testid="stSidebarNav"] header { color: var(--muted) !important; text-transform: uppercase; letter-spacing:.14em; font-size:.68rem !important; }

/* widgets */
.stButton > button, .stDownloadButton > button, [data-testid="stPopover"] button, .stFormSubmitButton > button {
  border-radius: 12px; border:1px solid var(--line-2); background: rgba(255,255,255,.03); color: var(--text);
  transition: all .18s ease; }
.stButton > button:hover, [data-testid="stPopover"] button:hover { border-color: rgba(139,92,246,.7); background: rgba(139,92,246,.12); transform: translateY(-1px); }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] { background: linear-gradient(120deg,#7c3aed,#6d5bf0 45%,#2563eb); border:0; color:white; font-weight:600;
  box-shadow: 0 8px 24px -8px rgba(124,58,237,.65); }
.stButton > button[kind="primary"]:hover { filter: brightness(1.12); box-shadow: 0 10px 30px -8px rgba(124,58,237,.85); }
.stButton > button[kind="tertiary"] { border: 0; background: transparent; color: var(--text-2); padding: 0 .4rem; }
.stButton > button[kind="tertiary"]:hover { color: var(--text); background: rgba(255,255,255,.05); }
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] { background: var(--surface) !important; border-radius: 12px !important; border-color: var(--line-2) !important; }
[data-testid="stSlider"] [role="slider"] { box-shadow: 0 0 0 6px rgba(139,92,246,.18); }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: .25rem; border-bottom: 1px solid var(--line); }
[data-testid="stTabs"] [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: .55rem 1rem; color: var(--text-2); }
[data-testid="stTabs"] [aria-selected="true"] { color: var(--text); background: rgba(139,92,246,.10); }
[data-testid="stExpander"] { border:1px solid var(--line) !important; border-radius: 14px !important; background: rgba(18,17,28,.6); }
[data-testid="stMetric"] { background: linear-gradient(160deg, rgba(25,23,38,.9), rgba(15,14,24,.9)); border:1px solid var(--line); border-radius: 16px; padding: 1rem 1.1rem; }
[data-testid="stMetricValue"] { font-family:'Space Grotesk',sans-serif; color: var(--text); }
[data-testid="stMetricLabel"] p { color: var(--muted); text-transform: uppercase; letter-spacing:.08em; font-size:.72rem; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border-radius: 14px; overflow: hidden; border:1px solid var(--line); }
[data-testid="stPlotlyChart"] { background: linear-gradient(160deg, rgba(22,20,34,.72), rgba(13,12,20,.72)); border:1px solid var(--line); border-radius: 18px; overflow: hidden; }
[data-testid="stAlert"] { border-radius: 14px; }
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) { border-radius: 18px; }

/* hero */
.sb-hero { position:relative; display:grid; grid-template-columns: minmax(0,1fr) 230px; gap: 2rem; align-items:center; padding: 2.6rem 2.4rem 2.2rem; border-radius: 28px; overflow:hidden; border:1px solid var(--line);
  background: radial-gradient(120% 140% at 0% 0%, rgba(139,92,246,.30), transparent 55%),
              radial-gradient(90% 120% at 100% 0%, rgba(56,189,248,.20), transparent 60%),
              linear-gradient(160deg, #15122a, #0b0a14 70%); margin-bottom: 1.6rem; }
.sb-hero .eyebrow { display:inline-flex; gap:.5rem; align-items:center; font-size:.74rem; font-weight:600; letter-spacing:.16em; text-transform:uppercase;
  color: var(--violet-2); background: rgba(139,92,246,.12); border:1px solid rgba(139,92,246,.35); padding:.35rem .75rem; border-radius:999px; }
.sb-hero h1 { font-size: clamp(2.2rem, 5vw, 4.1rem); line-height:1.02; margin:.9rem 0 .6rem; font-weight:700; }
.sb-grad { background: var(--grad); -webkit-background-clip:text; background-clip:text; color: transparent; }
.sb-hero p.lead { font-size: 1.08rem; max-width: 560px; color: var(--text-2); position:relative; z-index:1; }
.sb-eq { display:flex; gap:6px; align-items:center; justify-content:flex-end; height: 170px; opacity:.9; }
.sb-eq span { width: 8px; border-radius: 6px; background: var(--grad); animation: sbEq 1.25s ease-in-out infinite; transform-origin: center; }
@keyframes sbEq { 0%,100%{ transform: scaleY(.3)} 50%{ transform: scaleY(1)} }
@media (max-width: 1000px){ .sb-eq{ display:none; } .sb-hero{ grid-template-columns: 1fr; padding: 1.8rem 1.4rem; } }
@media (max-width: 640px){ .sb-hero{ padding: 1.4rem 1.1rem; border-radius: 20px; } .sb-hero h1{ font-size: 2.1rem; } .sb-tiles{ grid-template-columns: repeat(2, minmax(0,1fr)); } .sb-tile .v{ font-size:1.35rem; } [data-testid="stMainBlockContainer"], .block-container{ padding-left: 1rem; padding-right: 1rem; } }

/* page header */
.sb-page { margin: .2rem 0 1.4rem; }
.sb-page .kicker { color: var(--violet-2); font-size:.74rem; letter-spacing:.18em; text-transform:uppercase; font-weight:600; }
.sb-page h1 { font-size: clamp(1.9rem, 3.4vw, 2.8rem); margin:.25rem 0 .35rem; }
.sb-page p { max-width: 820px; font-size: 1.0rem; }

/* section */
.sb-section { display:flex; flex-wrap:wrap; align-items:baseline; column-gap:.8rem; row-gap:.2rem; margin: 1.8rem 0 .8rem; }
.sb-section h3 { flex: 0 1 auto; }
.sb-section h3 { margin:0; font-size:1.3rem; }
.sb-section .hint { color: var(--muted); font-size:.85rem; }

/* panels & tiles */
.sb-panel { border:1px solid var(--line); border-radius: 20px; padding: 1.2rem 1.3rem; background: linear-gradient(160deg, rgba(25,23,38,.85), rgba(13,12,20,.85)); }
.sb-panel h4 { margin: 0 0 .4rem; font-size: 1.02rem; }
.sb-tiles { display:grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr)); gap: .8rem; margin: .4rem 0 1rem; }
.sb-tile { position:relative; border:1px solid var(--line); border-radius: 18px; padding: 1rem 1.1rem; overflow:hidden;
  background: linear-gradient(160deg, rgba(27,24,44,.92), rgba(14,13,22,.92)); }
.sb-tile::after{ content:""; position:absolute; inset:auto -30% -60% auto; width:140px; height:140px; border-radius:50%; background: radial-gradient(circle, rgba(139,92,246,.22), transparent 70%); }
.sb-tile .v { font-family:'Space Grotesk',sans-serif; font-size: 1.75rem; font-weight:700; color: var(--text); line-height:1.1; }
.sb-tile .l { color: var(--muted); font-size:.72rem; letter-spacing:.1em; text-transform:uppercase; margin-top:.25rem; }
.sb-tile .s { color: var(--text-2); font-size:.8rem; margin-top:.35rem; }

/* chips */
.sb-chip { display:inline-block; padding:.18rem .6rem; margin: 0 .3rem .3rem 0; border-radius:999px; font-size:.72rem; font-weight:500;
  color: var(--text-2); background: rgba(255,255,255,.05); border:1px solid var(--line); }
.sb-chip.v { color:#d8ccff; background: rgba(139,92,246,.14); border-color: rgba(139,92,246,.35); }
.sb-chip.b { color:#bdeaff; background: rgba(56,189,248,.10); border-color: rgba(56,189,248,.32); }
.sb-chip.a { color:#a7f0d3; background: rgba(25,158,112,.14); border-color: rgba(25,158,112,.4); }

/* track card */
.sb-tc { container-type: inline-size; margin-bottom: .1rem; }
.sb-track { display:grid; grid-template-columns: 84px 1fr auto; gap: 1rem; align-items:center; padding: .85rem; border-radius: 18px;
  border:1px solid var(--line); background: linear-gradient(160deg, rgba(27,24,44,.88), rgba(14,13,22,.88));
  transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease; animation: sbIn .45s ease both; }
.sb-track:hover { transform: translateY(-2px); border-color: rgba(139,92,246,.55); box-shadow: 0 18px 40px -22px rgba(139,92,246,.7); }
@keyframes sbIn { from { opacity:0; transform: translateY(6px);} to {opacity:1; transform:none;} }
.sb-art { position:relative; width:84px; height:84px; border-radius: 14px; overflow:hidden; box-shadow: 0 10px 24px -12px rgba(0,0,0,.8); }
.sb-art svg, .sb-art img { width:100%; height:100%; display:block; object-fit:cover; }
.sb-rank { position:absolute; left:6px; top:6px; font-size:.68rem; font-weight:700; color:#fff; background: rgba(0,0,0,.45); padding: .05rem .4rem; border-radius: 8px; backdrop-filter: blur(4px); }
.sb-title { font-family:'Space Grotesk',sans-serif; font-weight:600; font-size: 1.04rem; color: var(--text); line-height:1.25; overflow:hidden; text-overflow:ellipsis; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; }
.sb-artist { color: var(--text-2); font-size:.86rem; margin:.15rem 0 .35rem; }
.sb-artist .album { color: var(--muted); }
.sb-reason { color: var(--text-2); font-size: .8rem; margin-top:.15rem; }
.sb-reason b { color: var(--text); font-weight:600; }
.sb-scores { display:flex; gap:.6rem; align-items:center; }
.sb-score { display:flex; flex-direction:column; align-items:center; gap:.1rem; }
.sb-score small { color: var(--muted); font-size:.62rem; letter-spacing:.08em; text-transform:uppercase; }
.sb-pop { width: 64px; }
.sb-pop .bar { height:6px; border-radius:6px; background: rgba(255,255,255,.08); overflow:hidden; }
.sb-pop .bar i { display:block; height:100%; border-radius:6px; background: linear-gradient(90deg,#38bdf8,#8b5cf6); }
.sb-pop small { display:block; text-align:center; color: var(--muted); font-size:.62rem; letter-spacing:.08em; text-transform:uppercase; margin-top:.3rem; }
@container (max-width: 520px){ .sb-track{ grid-template-columns: 64px 1fr; } .sb-art{ width:64px; height:64px; } .sb-scores{ grid-column: 1 / -1; justify-content:flex-start; } }

/* seed pill in sidebar */
.sb-seed { display:flex; gap:.6rem; align-items:center; padding:.45rem; border-radius: 12px; border:1px solid var(--line); background: rgba(255,255,255,.03); margin-bottom:.4rem; }
.sb-seed .art { width:38px; height:38px; border-radius:9px; overflow:hidden; flex: none; }
.sb-seed .art svg { width:100%; height:100%; display:block; }
.sb-seed .t { font-size:.8rem; color: var(--text); line-height:1.2; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width: 170px; }
.sb-seed .a { font-size:.72rem; color: var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width: 170px; }
.sb-brand { display:flex; align-items:center; gap:.65rem; padding: .3rem 0 1rem; }
.sb-brand .name { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size: 1.15rem; color: var(--text); }
.sb-brand .tag { font-size:.68rem; color: var(--muted); letter-spacing:.06em; }

.sb-note { border-left: 3px solid var(--violet); padding: .55rem .9rem; background: rgba(139,92,246,.08); border-radius: 0 12px 12px 0; color: var(--text-2); font-size: .86rem; margin: .5rem 0 1rem; }
.sb-reasons { list-style:none; padding:0; margin:0; }
.sb-reasons li { position:relative; padding: .55rem .8rem .55rem 2.1rem; margin-bottom:.45rem; border-radius: 12px; border:1px solid var(--line); background: rgba(255,255,255,.025); color: var(--text-2); }
.sb-reasons li::before { content:""; position:absolute; left:.85rem; top: 1.0rem; width:8px; height:8px; border-radius:50%; background: var(--grad); }
.sb-footer { margin-top: 3rem; padding-top: 1.2rem; border-top:1px solid var(--line); color: var(--muted); font-size:.78rem; }
.sb-empty { text-align:center; padding: 2.4rem 1rem; border:1px dashed var(--line-2); border-radius: 20px; color: var(--text-2); background: rgba(255,255,255,.015); }
.sb-empty h4 { margin: .6rem 0 .3rem; }
.sb-step { display:flex; gap:.8rem; align-items:flex-start; }
.sb-step .n { flex:none; width: 30px; height:30px; border-radius:10px; display:grid; place-items:center; font-weight:700; color:white; background: linear-gradient(135deg,#7c3aed,#2563eb); }
</style>
"""

LOGO = """<svg width="34" height="34" viewBox="0 0 34 34" aria-hidden="true"><defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="#a78bfa"/><stop offset="1" stop-color="#38bdf8"/></linearGradient></defs>
<rect width="34" height="34" rx="10" fill="#151226" stroke="url(#lg)" stroke-opacity=".5"/>
<path d="M6 19 Q11 9 17 17 T28 15" fill="none" stroke="url(#lg)" stroke-width="2.6" stroke-linecap="round"/>
<g fill="url(#lg)"><rect x="8" y="22" width="2.6" height="5" rx="1.3"/><rect x="13" y="20" width="2.6" height="7" rx="1.3"/>
<rect x="18" y="23" width="2.6" height="4" rx="1.3"/><rect x="23" y="21" width="2.6" height="6" rx="1.3"/></g></svg>"""


# ------------------------------------------------------------------ resources & state
@st.cache_resource(show_spinner="Loading the SoundBridge Encoder and track catalogue…")
def get_engine():
    from .engine import SoundBridgeEngine

    return SoundBridgeEngine()


def init_state() -> None:
    from .feedback import SessionFeedback, new_session_id

    ss = st.session_state
    ss.setdefault("seeds", [])
    ss.setdefault("fb", SessionFeedback())
    ss.setdefault("consent", False)
    ss.setdefault("session_id", new_session_id())
    ss.setdefault("prefs", {"familiarity": 0.6, "discovery": 0.4, "exploration": 0.1, "diversity": 0.5})
    ss.setdefault("why_track", None)
    ss.setdefault("embeds", False)
    ss.setdefault("feedback_given", {})


def feedback_state():
    from .engine import FeedbackState

    fb = st.session_state.fb
    return FeedbackState(liked=list(fb.liked), disliked=list(fb.disliked), penalised_genres=list(fb.penalised_genres))


def current_profile(engine):
    seeds = st.session_state.seeds
    if not seeds:
        return None
    return engine.profile(seeds, feedback_state())


def add_seed(idx: int) -> None:
    if idx not in st.session_state.seeds:
        st.session_state.seeds.append(int(idx))


def remove_seed(idx: int) -> None:
    if idx in st.session_state.seeds:
        st.session_state.seeds.remove(idx)


def demo_seeds(engine, which: str = "indie") -> list[int]:
    """Example seed sets built by searching the real catalogue (no hard-coded indices)."""
    presets = {
        "indie": [("Mr. Brightside", "The Killers"), ("Do I Wanna Know?", "Arctic Monkeys"), ("Creep", "Radiohead")],
        "electronic": [("Strobe", "deadmau5"), ("Midnight City", "M83"), ("Get Lucky", "Daft Punk")],
        "acoustic": [("Holocene", "Bon Iver"), ("Skinny Love", "Bon Iver"), ("The Night We Met", "Lord Huron")],
        "hiphop": [("HUMBLE.", "Kendrick Lamar"), ("SICKO MODE", "Travis Scott"), ("Lose Yourself", "Eminem")],
    }
    out = []
    for title, artist in presets.get(which, []):
        t = engine.tracks
        m = t[(t["track_name"].str.lower().str.startswith(title.lower())) & (t["artists"].str.contains(artist, case=False, regex=False))]
        if len(m):
            out.append(int(m.sort_values("popularity", ascending=False).iloc[0]["idx"]))
    return out


# ------------------------------------------------------------------ artwork
def artwork_svg(row: pd.Series | dict, size: int = 96, uid: str | None = None) -> str:
    """Generative cover derived *only* from the track's audio features (not official artwork)."""
    tid = str(row.get("track_id", row.get("idx", "x")))
    h = int(hashlib.md5(tid.encode()).hexdigest(), 16)
    rng = np.random.default_rng(h % (2**32))
    v = float(row.get("valence", .5)); e = float(row.get("energy", .5)); d = float(row.get("danceability", .5))
    a = float(row.get("acousticness", .5)); ins = float(row.get("instrumentalness", 0)); tempo = float(row.get("tempo", 120))
    # low valence -> cool blues/violets, high valence -> warm magentas/ambers; acoustic -> teal shift
    h1 = (195 + 150 * v - 80 * a + (h % 61) - 30) % 360
    h2 = (h1 + 35 + 70 * e) % 360
    l1, l2 = 14 + 16 * e, 30 + 24 * v
    gid = (uid or tid)[-10:].replace("-", "_")
    rings = 2 + int(np.clip(tempo, 60, 200) // 40)
    cx, cy = 30 + 40 * d, 30 + 40 * (1 - a)
    parts = [
        f'<svg viewBox="0 0 100 100" width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg" role="img">'
        f'<title>Generated from audio features — not official artwork</title>'
        f'<defs><linearGradient id="g{gid}" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="hsl({h1:.0f},70%,{l1:.0f}%)"/><stop offset="1" stop-color="hsl({h2:.0f},75%,{l2:.0f}%)"/></linearGradient>'
        f'<radialGradient id="r{gid}"><stop offset="0" stop-color="hsl({h2:.0f},90%,75%)" stop-opacity=".75"/>'
        f'<stop offset="1" stop-color="hsl({h2:.0f},90%,60%)" stop-opacity="0"/></radialGradient></defs>'
        f'<rect width="100" height="100" fill="url(#g{gid})"/>'
        f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{26 + 18 * e:.0f}" fill="url(#r{gid})"/>'
    ]
    for i in range(rings):
        r = 10 + i * (60 / rings) + rng.uniform(-2, 2)
        parts.append(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{r:.1f}" fill="none" stroke="white" stroke-opacity="{.10 + .05 * (i % 2):.2f}" stroke-width="{0.6 + ins:.1f}"/>')
    feats = [row.get(k, .5) for k in ("danceability", "energy", "valence", "acousticness", "instrumentalness", "speechiness", "liveness")]
    for i, f in enumerate(feats):
        hh = 6 + 26 * float(f)
        parts.append(f'<rect x="{12 + i * 11}" y="{92 - hh:.1f}" width="6" height="{hh:.1f}" rx="3" fill="white" fill-opacity=".55"/>')
    parts.append("</svg>")
    return "".join(parts)


# ------------------------------------------------------------------ components
def setup_page() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()


def page_header(kicker: str, title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="sb-page"><div class="kicker">{html.escape(kicker)}</div><h1>{title}</h1>'
        f'<p>{subtitle}</p></div>', unsafe_allow_html=True)


def section(title: str, hint: str = "") -> None:
    st.markdown(f'<div class="sb-section"><h3>{title}</h3><span class="hint">{hint}</span></div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="sb-note">{text}</div>', unsafe_allow_html=True)


def tiles(items: list[tuple[str, str, str]]) -> None:
    """items: (value, label, sub)"""
    inner = "".join(f'<div class="sb-tile"><div class="v">{v}</div><div class="l">{l}</div>'
                    + (f'<div class="s">{s}</div>' if s else "") + "</div>" for v, l, s in items)
    st.markdown(f'<div class="sb-tiles">{inner}</div>', unsafe_allow_html=True)


def empty_state(title: str, body: str, icon: str = "🎧") -> None:
    st.markdown(f'<div class="sb-empty"><div style="font-size:2rem">{icon}</div><h4>{title}</h4><p>{body}</p></div>',
                unsafe_allow_html=True)


def chips(values, kind: str = "") -> str:
    return "".join(f'<span class="sb-chip {kind}">{html.escape(str(v))}</span>' for v in values if v)


def track_card(engine, row: pd.Series, rank: int | None = None, reason: str | None = None,
               match: float | None = None, discovery: float | None = None, art_url: str | None = None) -> str:
    t = engine.tracks.loc[int(row["idx"])]
    art = (f'<img src="{html.escape(art_url)}" alt="Album artwork from Spotify"/>' if art_url
           else artwork_svg(t, uid=f"{t['track_id']}{rank or ''}"))
    genres = list(t["genre"])[:3]
    lk = engine.exposure_known[int(row["idx"])] and engine.artist_exposure[int(row["idx"])] <= engine.lesser_known
    chip_html = chips(genres, "v") + (chips(["Lesser-known in dataset"], "a") if lk else "")
    rank_html = f'<span class="sb-rank">#{rank}</span>' if rank else ""
    scores = ""
    if match is not None:
        scores += f'<div class="sb-score">{gauge_ring(match, "match", "#8b5cf6")}<small>match</small></div>'
    if discovery is not None:
        scores += f'<div class="sb-score">{gauge_ring(discovery, "discovery", "#38bdf8")}<small>discovery</small></div>'
    pop = int(t["popularity"])
    scores += (f'<div class="sb-pop" title="Spotify popularity at collection time: an exposure proxy, not quality">'
               f'<div class="bar"><i style="width:{pop}%"></i></div><small>pop {pop}</small></div>')
    reason_html = f'<div class="sb-reason">{reason}</div>' if reason else ""
    return (
        f'<div class="sb-tc"><div class="sb-track"><div class="sb-art">{art}{rank_html}</div>'
        f'<div class="sb-meta"><div class="sb-title">{html.escape(str(t["track_name"]))}</div>'
        f'<div class="sb-artist">{html.escape(t["artists"].replace(";", ", "))} · <span class="album">{html.escape(str(t["album_name"]))}</span></div>'
        f'<div>{chip_html}</div>{reason_html}</div>'
        f'<div class="sb-scores">{scores}</div></div></div>'
    )


def seed_pill(engine, idx: int) -> str:
    t = engine.tracks.loc[idx]
    return (f'<div class="sb-seed"><div class="art">{artwork_svg(t, 38, uid=f"s{idx}")}</div><div>'
            f'<div class="t">{html.escape(str(t["track_name"]))}</div><div class="a">{html.escape(t["primary_artist"])}</div></div></div>')


def sidebar(engine) -> None:
    with st.sidebar:
        st.markdown(f'<div class="sb-brand">{LOGO}<div><div class="name">SoundBridge AI</div>'
                    f'<div class="tag">Discover beyond your algorithm</div></div></div>', unsafe_allow_html=True)
        seeds = st.session_state.seeds
        st.caption(f"YOUR SEEDS · {len(seeds)}")
        if seeds:
            for i in seeds[:6]:
                st.markdown(seed_pill(engine, i), unsafe_allow_html=True)
            if len(seeds) > 6:
                st.caption(f"+ {len(seeds) - 6} more")
            if st.button("Clear seeds", key="sb_clear", icon=":material/delete_sweep:", width="stretch"):
                st.session_state.seeds = []
                st.rerun()
        else:
            st.markdown('<div class="sb-note" style="font-size:.78rem">Add songs you enjoy on <b>Music Discovery</b> '
                        'to build your Musical DNA.</div>', unsafe_allow_html=True)
        fb = st.session_state.fb
        if fb.events:
            st.caption(f"Feedback this session: {len(fb.events)} · liked {len(fb.liked)} · disliked {len(fb.disliked)}")


def footer() -> None:
    st.markdown(
        '<div class="sb-footer">SoundBridge AI is an independent research project. Data: Spotify Tracks Dataset '
        '(public Kaggle release, audio features from Spotify\'s Web API at collection time). Popularity is an exposure '
        'proxy, not a measure of quality. Not affiliated with or endorsed by Spotify. Cover images are generated from '
        'audio features unless Spotify artwork is explicitly enabled.</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ feedback widgets
def record_feedback(engine, row: pd.Series, kind: str, weights: dict | None = None) -> None:
    from .feedback import FeedbackEvent, FeedbackStore, apply_event

    idx = int(row["idx"])
    t = engine.tracks.loc[idx]
    ev = FeedbackEvent(
        session_id=st.session_state.session_id, track_idx=idx, track_id=str(t["track_id"]), feedback=kind,
        strategy=str(row.get("strategy", "")), rank=int(row.get("rank", -1)),
        relevance=float(row.get("relevance", float("nan"))), popularity=int(t["popularity"]),
        genre=str(t.get("primary_genre", "")), artist=str(t["primary_artist"]), weights=weights or {},
    )
    apply_event(st.session_state.fb, ev, primary_genre=str(t.get("primary_genre", "")))
    st.session_state.feedback_given.setdefault(idx, []).append(kind)
    if st.session_state.consent:
        FeedbackStore().record(ev)


def track_actions(engine, row: pd.Series, key: str, weights: dict | None = None,
                  on_why: Callable[[int], None] | None = None) -> None:
    from .feedback import FEEDBACK_TYPES

    idx = int(row["idx"])
    given = st.session_state.feedback_given.get(idx, [])
    from .spotify_integration import track_url

    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
        if st.button("Like", key=f"like_{key}", icon=":material/thumb_up:", help="Like",
                     type="primary" if "like" in given else "secondary"):
            record_feedback(engine, row, "like", weights)
            st.toast("Liked — your profile moves toward this sound", icon="💜")
            st.rerun()
        if st.button("", key=f"dis_{key}", icon=":material/thumb_down:", help="Dislike",
                     type="primary" if "dislike" in given else "secondary"):
            record_feedback(engine, row, "dislike", weights)
            st.toast("Noted — similar tracks will be ranked lower", icon="🫡")
            st.rerun()
        with st.popover("", icon=":material/more_horiz:", help="More feedback options"):
            for k in ("known_artist", "not_my_genre", "more_like_this", "more_variety"):
                if st.button(FEEDBACK_TYPES[k], key=f"{k}_{key}", width="stretch",
                             type="primary" if k in given else "secondary"):
                    record_feedback(engine, row, k, weights)
                    st.toast(f"Feedback recorded: {FEEDBACK_TYPES[k]}", icon="✅")
                    st.rerun()
        if on_why is not None and st.button("Why?", key=f"why_{key}", icon=":material/psychology:", type="tertiary"):
            on_why(idx)
        st.link_button("Spotify", track_url(engine.tracks.at[idx, "track_id"]), icon=":material/open_in_new:",
                       type="tertiary")


def go_why(idx: int) -> None:
    st.session_state.why_track = int(idx)
    st.switch_page("views/9_Why_This_Song.py")


def hero_eq(n: int = 16) -> str:
    rng = np.random.default_rng(7)
    bars = "".join(
        f'<span style="height:{rng.integers(30, 100)}%;animation-delay:{-rng.uniform(0, 1.2):.2f}s;animation-duration:{rng.uniform(.9, 1.6):.2f}s"></span>'
        for _ in range(n))
    return f'<div class="sb-eq">{bars}</div>'


# ------------------------------------------------------------------ cached computations
@st.cache_data(max_entries=512, show_spinner=False)
def cached_recommend(_engine, seeds: tuple, liked: tuple, disliked: tuple, penalised: tuple, strategy: str,
                     weights: tuple, k: int, personal: bool = False) -> pd.DataFrame:
    from .engine import FeedbackState
    from .feedback import PersonalPreferenceModel
    from .hybrid_recommender import RankingWeights, Recommender

    profile = _engine.profile(list(seeds), FeedbackState(list(liked), list(disliked), list(penalised)))
    w = RankingWeights(**dict(weights)) if weights else None
    scores = None
    if personal:
        pm = PersonalPreferenceModel(_engine.cfg["feedback"]["min_labels_for_personal_model"])
        if pm.fit(_engine.embeddings, list(liked), list(disliked)):
            cands = _engine.content.candidates(profile, n=_engine.cfg["recommendation"]["candidate_pool"])
            scores = pm.predict(_engine.embeddings, cands.idx)
    return Recommender(_engine).recommend(profile, strategy, weights=w, k=k, personal_scores=scores)


def recommend_now(engine, strategy: str = "hybrid", weights=None, k: int = 12) -> pd.DataFrame:
    fb = st.session_state.fb
    wt = tuple(sorted(weights.as_dict().items())) if weights is not None else ()
    return cached_recommend(engine, tuple(st.session_state.seeds), tuple(fb.liked), tuple(fb.disliked),
                            tuple(fb.penalised_genres), strategy, wt, k, personal=True)


def short_reason_for(engine, row: pd.Series, seeds: list[int]) -> str:
    from .explainability import feature_comparison, short_reason

    comp = feature_comparison(engine.tracks, seeds, int(row["idx"]), engine.catalog_std)
    sg = engine.seed_genres(seeds)
    tg = engine.tracks.at[int(row["idx"]), "genre"]
    shared = next((g for g in tg if g in sg), None)
    lk = bool(engine.exposure_known[int(row["idx"])] and engine.artist_exposure[int(row["idx"])] <= engine.lesser_known)
    text = short_reason(comp, shared, lk)
    if not shared and len(tg):
        rel = engine.related_genre_for(int(row["idx"]), sg)
        if rel:
            text += f" · new genre: {rel[0]} (close to your {rel[1]})"
    return text
