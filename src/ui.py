"""SoundBridge design system for Streamlit: CSS, generative artwork, cards and shared state."""
from __future__ import annotations

import hashlib
import html
from typing import Callable

import numpy as np
import pandas as pd
import streamlit as st

ACCENT = "#e0a63c"   # amber: the level-meter needle; used sparingly

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root{
  --bg:#141414; --surface:#1b1b1c; --surface-2:#232324; --surface-3:#2b2b2c;
  --line:rgba(236,236,234,.08); --line-2:rgba(236,236,234,.14);
  --text:#ecebe8; --text-2:#b4b2ac; --muted:#878580;
  --accent:#e0a63c; --accent-ink:#1a1408; --accent-soft:rgba(224,166,60,.12);
  --mono:'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
  --ease:cubic-bezier(.2,.7,.2,1);
}
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
.stApp { background: var(--bg); }
[data-testid="stHeader"]{ background: transparent; }
[data-testid="stMainBlockContainer"], .block-container { padding-top: 32px; max-width: 1440px; }
h1,h2,h3,h4 { font-family: 'IBM Plex Sans', system-ui, sans-serif !important; font-weight: 500 !important; color: var(--text); }
h1 { letter-spacing: -.015em; }
p, li { color: var(--text-2); }
a { color: var(--text); text-decoration-color: var(--line-2); }
a:hover { text-decoration-color: var(--accent); }
hr { border-color: var(--line) !important; }
code { font-family: var(--mono); }
.material-symbols-rounded, .sb-ico { font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal; line-height: 1;
  letter-spacing: normal; text-transform: none; display: inline-block; white-space: nowrap; -webkit-font-feature-settings: 'liga'; font-feature-settings: 'liga'; }

/* sidebar */
[data-testid="stSidebar"]{ background: #181818; border-right:1px solid var(--line); }
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a { border-radius: 8px; transition: background-color .15s var(--ease), color .15s var(--ease); }
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover { background: var(--surface-2); }
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] { background: var(--surface-3); box-shadow: inset 0 0 0 1px var(--line); }
[data-testid="stSidebarNavSeparator"]{ border-color: var(--line); }
[data-testid="stNavSectionHeader"], [data-testid="stSidebarNav"] header { color: var(--muted) !important; text-transform: none; letter-spacing: 0; font-size: 12px !important; font-weight: 500; }

/* widgets */
.stButton > button, .stDownloadButton > button, [data-testid="stPopover"] button, .stFormSubmitButton > button, .stLinkButton > a {
  border-radius: 8px; border:1px solid var(--line-2); background: var(--surface); color: var(--text);
  transition: background-color .15s var(--ease), border-color .15s var(--ease), color .15s var(--ease), transform .1s var(--ease); }
.stButton > button:hover, [data-testid="stPopover"] button:hover, .stDownloadButton > button:hover { border-color: rgba(236,236,234,.24); background: var(--surface-2); color: var(--text); }
.stButton > button:active { transform: scale(.98); }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] { background: var(--accent); border:1px solid var(--accent); color: var(--accent-ink); font-weight: 500; }
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover { background: #e8b453; border-color: #e8b453; color: var(--accent-ink); }
.stButton > button[kind="primary"] p, .stFormSubmitButton > button[kind="primary"] p { color: var(--accent-ink); }
.stButton > button[kind="tertiary"], .stLinkButton > a[kind="tertiary"] { border: 0; background: transparent; color: var(--text-2); padding: 0 8px; }
.stButton > button[kind="tertiary"]:hover { color: var(--text); background: var(--surface-2); }
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] { background: var(--surface) !important; border-radius: 8px !important; border-color: var(--line-2) !important; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
[data-testid="stTabs"] [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 8px 16px; color: var(--text-2); }
[data-testid="stTabs"] [aria-selected="true"] { color: var(--text); }
[data-testid="stExpander"] { border:1px solid var(--line) !important; border-radius: 12px !important; background: var(--surface); }
[data-testid="stMetric"] { background: var(--surface); border:1px solid var(--line); border-radius: 12px; padding: 16px 20px; }
[data-testid="stMetricValue"] { color: var(--text); font-weight: 500; font-variant-numeric: tabular-nums; }
[data-testid="stMetricLabel"] p { color: var(--muted); font-size: 13px; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border-radius: 12px; overflow: hidden; border:1px solid var(--line); }
[data-testid="stPlotlyChart"] { background: var(--surface); border:1px solid var(--line); border-radius: 12px; overflow: hidden; }
[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) { border-radius: 12px; }

/* hero: text left, sleeves right */
.sb-hero { display:grid; grid-template-columns: minmax(0,1.25fr) minmax(0,1fr); gap: 48px; align-items:center; padding: 40px;
  border-radius: 16px; border:1px solid var(--line); background: var(--surface); margin-bottom: 24px; }
.sb-hero.solo { grid-template-columns: minmax(0,1fr); }
.sb-hero .meta { font-family: var(--mono); font-size: 13px; color: var(--muted); font-variant-numeric: tabular-nums; }
.sb-hero h1 { font-size: clamp(36px, 4.4vw, 60px); line-height: 1.05; margin: 12px 0 16px; padding: 0; text-wrap: balance; }
.sb-hero p.lead { font-size: 17px; line-height: 1.55; max-width: 560px; color: var(--text-2); margin: 0; }

/* record crate: generated sleeves, front one carries a liner note */
.sb-crate { position: relative; height: 288px; }
.sb-sleeve { position:absolute; width: 208px; height: 208px; border-radius: 4px; overflow:hidden;
  box-shadow: 0 1px 0 rgba(255,255,255,.06) inset, 0 12px 28px -12px rgba(0,0,0,.7); }
.sb-sleeve svg { width:100%; height:100%; display:block; }
.sb-sleeve.s0 { left: 0; top: 0; } .sb-sleeve.s1 { left: 64px; top: 20px; } .sb-sleeve.s2 { left: 128px; top: 40px; }
.sb-liner { position:absolute; left: 128px; top: 256px; width: 208px; font-family: var(--mono); font-size: 12px; color: var(--muted); line-height: 1.4; }
.sb-liner b { color: var(--text-2); font-weight: 500; }
@media (max-width: 1000px){ .sb-hero{ grid-template-columns: 1fr; gap: 32px; padding: 32px 24px; } .sb-crate{ height: 248px; }
  .sb-sleeve{ width: 168px; height: 168px; } .sb-sleeve.s1{ left: 48px; top: 16px; } .sb-sleeve.s2{ left: 96px; top: 32px; }
  .sb-liner{ left: 96px; top: 208px; width: 200px; } }
@media (max-width: 640px){ .sb-hero{ padding: 24px 20px; } .sb-hero h1{ font-size: 32px; } .sb-tiles{ grid-template-columns: repeat(2, minmax(0,1fr)); }
  .sb-tile .v{ font-size: 22px; } [data-testid="stMainBlockContainer"], .block-container{ padding-left: 16px; padding-right: 16px; } }

/* page header */
.sb-page { margin: 4px 0 24px; }
.sb-page .kicker { color: var(--muted); font-size: 13px; font-family: var(--mono); }
.sb-page h1 { font-size: clamp(30px, 3.2vw, 40px); line-height: 1.1; margin: 8px 0 8px; padding: 0; }
.sb-page p { max-width: 760px; font-size: 16px; line-height: 1.55; margin: 0; }

/* section */
.sb-section { display:flex; flex-wrap:wrap; align-items:baseline; column-gap: 12px; row-gap: 4px; margin: 32px 0 12px; }
.sb-section h3 { flex: 0 1 auto; margin:0; padding: 0; font-size: 20px; }
.sb-section .hint { color: var(--muted); font-size: 13px; }

/* panels & tiles */
.sb-panel { border:1px solid var(--line); border-radius: 12px; padding: 20px 24px; background: var(--surface); }
.sb-panel h4 { margin: 0 0 4px; padding: 0; font-size: 16px; }
.sb-panel p { margin: 0; font-size: 14px; line-height: 1.5; }
.sb-tiles { display:grid; grid-template-columns: repeat(auto-fit, minmax(168px,1fr)); gap: 12px; margin: 4px 0 16px; }
.sb-tile { border:1px solid var(--line); border-radius: 12px; padding: 16px 20px; background: var(--surface); }
.sb-tile .v { font-size: 26px; font-weight: 500; color: var(--text); line-height: 1.15; font-variant-numeric: tabular-nums; }
.sb-tile .l { color: var(--text-2); font-size: 13px; margin-top: 4px; }
.sb-tile .s { color: var(--muted); font-size: 12px; margin-top: 4px; line-height: 1.4; }

/* chips: quiet text tags */
.sb-chip { display:inline-block; padding: 2px 8px; margin: 0 4px 4px 0; border-radius: 4px; font-size: 12px; font-weight: 400;
  color: var(--text-2); background: var(--surface-2); border:1px solid var(--line); }
.sb-chip.a { color: var(--accent); border-color: rgba(224,166,60,.32); background: transparent; }

/* track card */
.sb-tc { container-type: inline-size; margin-bottom: 4px; }
.sb-track { display:grid; grid-template-columns: 80px minmax(0,1fr) auto; gap: 16px; align-items:center; padding: 12px; border-radius: 12px;
  border:1px solid var(--line); background: var(--surface); transition: border-color .15s var(--ease), background-color .15s var(--ease); }
.sb-track:hover { border-color: var(--line-2); background: #1f1f20; }
.sb-art { position:relative; width:80px; height:80px; border-radius: 4px; overflow:hidden; }
.sb-art svg, .sb-art img { width:100%; height:100%; display:block; object-fit:cover; }
.sb-rank { position:absolute; left:4px; top:4px; font-family: var(--mono); font-size: 11px; font-weight: 500; color:#fff; background: rgba(0,0,0,.55); padding: 0 4px; border-radius: 2px; }
.sb-title { font-weight: 500; font-size: 15px; color: var(--text); line-height: 1.3; overflow:hidden; text-overflow:ellipsis; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; }
.sb-artist { color: var(--text-2); font-size: 13px; margin: 2px 0 8px; }
.sb-artist .album { color: var(--muted); }
.sb-reason { color: var(--text-2); font-size: 13px; line-height: 1.45; margin-top: 4px; }
.sb-reason b { color: var(--text); font-weight: 500; }

/* signature: seven-bar audio fingerprint, mono readout */
.sb-fp { display:flex; flex-direction:column; align-items:flex-end; gap: 8px; font-family: var(--mono); font-variant-numeric: tabular-nums; }
.sb-fp .bars { display:grid; grid-template-columns: repeat(7, 8px); column-gap: 4px; align-items:end; height: 28px; }
.sb-fp .bars i { display:block; width: 8px; background: var(--text-2); border-radius: 1px; }
.sb-fp .keys { display:grid; grid-template-columns: repeat(7, 8px); column-gap: 4px; font-size: 10px; color: var(--muted); text-align:center; line-height: 1; }
.sb-fp .nums { font-size: 11px; color: var(--muted); white-space: nowrap; line-height: 1.5; text-align:right; }
.sb-fp .nums b { color: var(--accent); font-weight: 500; }
.sb-fp .nums span { color: var(--text-2); }
@container (max-width: 480px){ .sb-track{ grid-template-columns: 64px minmax(0,1fr); } .sb-art{ width:64px; height:64px; }
  .sb-fp{ grid-column: 1 / -1; flex-direction: row; align-items:center; justify-content: space-between; } }

/* seed row in sidebar */
.sb-seed { display:flex; gap: 12px; align-items:center; padding: 8px; border-radius: 8px; border:1px solid var(--line); background: var(--surface); margin-bottom: 4px; }
.sb-seed .art { width: 36px; height: 36px; border-radius: 2px; overflow:hidden; flex: none; }
.sb-seed .art svg { width:100%; height:100%; display:block; }
.sb-seed .t { font-size: 13px; color: var(--text); line-height: 1.3; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width: 170px; }
.sb-seed .a { font-size: 12px; color: var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width: 170px; }
.sb-brand { display:flex; align-items:center; gap: 12px; padding: 4px 0 16px; }
.sb-brand .name { font-weight: 600; font-size: 17px; color: var(--text); }
.sb-brand .tag { font-size: 12px; color: var(--muted); }

.sb-note { border-left: 2px solid var(--accent); padding: 8px 16px; background: var(--surface); border-radius: 0 8px 8px 0; color: var(--text-2); font-size: 14px; line-height: 1.5; margin: 8px 0 16px; }
.sb-reasons { list-style:none; padding:0; margin:0; }
.sb-reasons li { position:relative; padding: 10px 16px 10px 32px; margin-bottom: 8px; border-radius: 8px; border:1px solid var(--line); background: var(--surface); color: var(--text-2); font-size: 14px; line-height: 1.5; }
.sb-reasons li::before { content:""; position:absolute; left: 16px; top: 19px; width: 6px; height: 6px; border-radius: 1px; background: var(--muted); }
.sb-footer { margin-top: 48px; padding-top: 16px; border-top:1px solid var(--line); color: var(--muted); font-size: 12px; line-height: 1.5; }
.sb-empty { text-align:center; padding: 40px 16px; border:1px dashed var(--line-2); border-radius: 12px; color: var(--text-2); background: transparent; }
.sb-empty .sb-ico { font-size: 28px; color: var(--muted); }
.sb-empty h4 { margin: 12px 0 4px; padding: 0; font-size: 16px; }
.sb-empty p { margin: 0; font-size: 14px; }
.sb-step { display:flex; gap: 12px; align-items:flex-start; }
.sb-step .n { flex:none; width: 28px; height: 28px; border-radius: 6px; display:grid; place-items:center; font-family: var(--mono); font-size: 13px; color: var(--text-2);
  background: var(--surface-2); border: 1px solid var(--line); }
</style>
"""

LOGO = """<svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true"><g fill="#e0a63c">
<rect x="3" y="14" width="3" height="10" rx="1"/><rect x="8.5" y="6" width="3" height="18" rx="1"/>
<rect x="14" y="10" width="3" height="14" rx="1"/><rect x="19.5" y="16" width="3" height="8" rx="1"/></g>
<rect x="2" y="25.5" width="22" height="1" fill="#878580"/></svg>"""


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


def empty_state(title: str, body: str, icon: str = "headphones") -> None:
    """icon: a Material Symbols name (e.g. "headphones", "search")."""
    st.markdown(f'<div class="sb-empty"><span class="sb-ico" aria-hidden="true">{html.escape(icon)}</span>'
                f'<h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def chips(values, kind: str = "") -> str:
    return "".join(f'<span class="sb-chip {kind}">{html.escape(str(v))}</span>' for v in values if v)


FINGERPRINT = [("danceability", "D"), ("energy", "E"), ("valence", "V"), ("acousticness", "A"),
               ("instrumentalness", "I"), ("speechiness", "S"), ("liveness", "L")]


def fingerprint(t: pd.Series | dict) -> str:
    """The seven-bar audio-feature fingerprint: a small mono readout shown beside every track."""
    vals = [float(np.clip(np.nan_to_num(float(t.get(k, 0))), 0, 1)) for k, _ in FINGERPRINT]
    tip = " · ".join(f"{k} {v:.2f}" for (k, _), v in zip(FINGERPRINT, vals))
    bars = "".join(f'<i style="height:{max(2, round(28 * v))}px"></i>' for v in vals)
    keys = "".join(f"<span>{c}</span>" for _, c in FINGERPRINT)
    return f'<div class="bars" title="{html.escape(tip)}" aria-label="Audio fingerprint: {html.escape(tip)}">{bars}</div><div class="keys" aria-hidden="true">{keys}</div>'


def track_card(engine, row: pd.Series, rank: int | None = None, reason: str | None = None,
               match: float | None = None, discovery: float | None = None, art_url: str | None = None) -> str:
    t = engine.tracks.loc[int(row["idx"])]
    art = (f'<img src="{html.escape(art_url)}" alt="Album artwork from Spotify"/>' if art_url
           else artwork_svg(t, uid=f"{t['track_id']}{rank or ''}"))
    genres = list(t["genre"])[:3]
    lk = engine.exposure_known[int(row["idx"])] and engine.artist_exposure[int(row["idx"])] <= engine.lesser_known
    chip_html = chips(genres) + (chips(["Lesser-known in dataset"], "a") if lk else "")
    rank_html = f'<span class="sb-rank">{rank:02d}</span>' if rank else ""
    nums = []
    if match is not None:
        nums.append(f'match <b>{float(np.clip(match, 0, 1)) * 100:.0f}</b>')
    if discovery is not None:
        nums.append(f'disc <span>{float(np.clip(discovery, 0, 1)) * 100:.0f}</span>')
    pop = int(t["popularity"])
    nums.append(f'<abbr title="Spotify popularity at collection time: an exposure proxy, not quality" '
                f'style="text-decoration:none">pop <span>{pop}</span></abbr>')
    reason_html = f'<div class="sb-reason">{reason}</div>' if reason else ""
    return (
        f'<div class="sb-tc"><div class="sb-track"><div class="sb-art">{art}{rank_html}</div>'
        f'<div class="sb-meta"><div class="sb-title">{html.escape(str(t["track_name"]))}</div>'
        f'<div class="sb-artist">{html.escape(t["artists"].replace(";", ", "))} · <span class="album">{html.escape(str(t["album_name"]))}</span></div>'
        f'<div>{chip_html}</div>{reason_html}</div>'
        f'<div class="sb-fp">{fingerprint(t)}<div class="nums">{" · ".join(nums)}</div></div></div></div>'
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
        st.caption(f"Your seeds · {len(seeds)}")
        if seeds:
            for i in seeds[:6]:
                st.markdown(seed_pill(engine, i), unsafe_allow_html=True)
            if len(seeds) > 6:
                st.caption(f"+ {len(seeds) - 6} more")
            if st.button("Clear seeds", key="sb_clear", icon=":material/delete_sweep:", width="stretch"):
                st.session_state.seeds = []
                st.rerun()
        else:
            st.markdown('<div class="sb-note" style="font-size:13px">Add songs you enjoy on <b>Music Discovery</b> '
                        'to build your Musical DNA.</div>', unsafe_allow_html=True)
        admin_login()
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
            st.toast("Liked — your profile moves toward this sound", icon=":material/thumb_up:")
            st.rerun()
        if st.button("", key=f"dis_{key}", icon=":material/thumb_down:", help="Dislike",
                     type="primary" if "dislike" in given else "secondary"):
            record_feedback(engine, row, "dislike", weights)
            st.toast("Noted — similar tracks will be ranked lower", icon=":material/thumb_down:")
            st.rerun()
        with st.popover("", icon=":material/more_horiz:", help="More feedback options"):
            for k in ("known_artist", "not_my_genre", "more_like_this", "more_variety"):
                if st.button(FEEDBACK_TYPES[k], key=f"{k}_{key}", width="stretch",
                             type="primary" if k in given else "secondary"):
                    record_feedback(engine, row, k, weights)
                    st.toast(f"Feedback recorded: {FEEDBACK_TYPES[k]}", icon=":material/check:")
                    st.rerun()
        if on_why is not None and st.button("Why?", key=f"why_{key}", icon=":material/psychology:", type="tertiary"):
            on_why(idx)
        st.link_button("Spotify", track_url(engine.tracks.at[idx, "track_id"]), icon=":material/open_in_new:",
                       type="tertiary")


def go_why(idx: int) -> None:
    st.session_state.why_track = int(idx)
    st.switch_page("views/9_Why_This_Song.py")


def hero_crate(engine, which: str = "indie") -> str:
    """Three covers generated from real tracks' audio features, stacked like sleeves in a listening-booth crate."""
    idx = demo_seeds(engine, which)[:3]
    if len(idx) < 3:
        idx = engine.tracks.sort_values("popularity", ascending=False)["idx"].head(3).astype(int).tolist()
    sleeves = "".join(f'<div class="sb-sleeve s{n}">{artwork_svg(engine.tracks.loc[i], 208, uid=f"hero{n}")}</div>'
                      for n, i in enumerate(idx))
    front = engine.tracks.loc[idx[-1]]
    liner = (f'<div class="sb-liner"><b>{html.escape(str(front["track_name"]))}</b><br>'
             f'{html.escape(str(front["primary_artist"]))} · cover drawn from its audio features</div>')
    return f'<div class="sb-crate" aria-hidden="true">{sleeves}{liner}</div>'


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


# ------------------------------------------------------------------ researcher (admin) access
def is_admin() -> bool:
    """True locally (no admin password configured) or after the researcher logs in."""
    from .storage import admin_password

    return admin_password() is None or bool(st.session_state.get("admin_ok"))


def admin_login() -> None:
    from .storage import admin_password

    pw = admin_password()
    if pw is None:
        return
    with st.sidebar.expander("Researcher login", icon=":material/lock:", expanded=False):
        if st.session_state.get("admin_ok"):
            st.caption("Logged in as researcher.")
            if st.button("Log out", key="admin_logout"):
                st.session_state.admin_ok = False
                st.rerun()
        else:
            entered = st.text_input("Password", type="password", key="admin_pw")
            if entered:
                import hmac

                st.session_state.admin_ok = hmac.compare_digest(entered, pw)
                if not st.session_state.admin_ok:
                    st.error("Wrong password")
                else:
                    st.rerun()


def admin_only(what: str) -> bool:
    if is_admin():
        return True
    st.info(f"{what} is limited to the researcher on this deployment. Log in from the sidebar.", icon=":material/lock:")
    return False
