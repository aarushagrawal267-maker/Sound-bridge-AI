import streamlit as st

from src import ui
from src.config import SHORT_LABELS
from src.listening_modes import MODES, atypicality, recommend_for_mode


@st.cache_data(show_spinner=False)
def _atyp(_engine):
    return atypicality(_engine)


engine = ui.get_engine()
ui.page_header("What should I listen to today?", "Listening Preferences",
               "Pick a listening preference. SoundBridge translates it into audio-feature targets, then uses the "
               "encoder to find tracks that share that sound — personalised to your Musical DNA if you have one.")
ui.note("These are <b>approximate descriptors</b> from audio analysis, not mood classifications. “Relaxed” means low "
        "estimated energy and softer dynamics; whether a song relaxes <i>you</i> is up to you.")

keys = list(MODES)
mode = st.segmented_control("Preference", keys, format_func=lambda k: MODES[k].title, default="energetic", key="mode_pick") or "energetic"
m = MODES[mode]
seeds = st.session_state.seeds
c1, c2 = st.columns([2, 1])
c1.markdown(f"#### {m.title}\n{m.description}")
personal = c2.toggle("Personalise with my Musical DNA", value=bool(seeds), disabled=not seeds)
genre_pick = None
if mode == "genre_exploration" and not (personal and seeds):
    genre_pick = st.selectbox("Start from a genre", engine.genres, index=engine.genres.index("indie") if "indie" in engine.genres else 0)

if m.constraints:
    parts = []
    for f, (lo, hi) in m.constraints.items():
        unit = " BPM" if f == "tempo" else " dB" if f == "loudness" else ""
        parts.append(f"{SHORT_LABELS[f].lower()} {lo:g}–{hi:g}{unit}")
    st.caption("Constraints: " + " · ".join(parts) + " · target: " +
               ", ".join(f"{SHORT_LABELS[f].lower()} ≈ {v:g}" for f, v in m.target.items()) + " (scaled 0–1)")

profile = ui.current_profile(engine) if (personal and seeds) else None
df, info = recommend_for_mode(engine, mode, profile=profile, k=12, explore_genre=genre_pick, atyp=_atyp(engine))
extra = ""
if "genres" in info:
    extra = "genres explored: " + ", ".join(info["genres"])
if "atypicality_threshold" in info:
    extra = f"top 15% most atypical for their genre (distance ≥ {info['atypicality_threshold']:.2f})"
ui.tiles([(f"{info['n_matching']:,}", "Tracks matching", extra or "after constraints"),
          ("Yes" if info.get("personalised") else "No", "Personalised", "blended with your profile" if info.get("personalised") else "")])

if df.empty:
    ui.empty_state("Nothing matched", "Try another preference.")
else:
    st.session_state.last_recs = df
    cols = st.columns(2, gap="medium")
    for n, (_, r) in enumerate(df.iterrows()):
        t = engine.tracks.loc[int(r["idx"])]
        reason = ", ".join(f"{SHORT_LABELS[f].lower()} {t[f]:.0f}{' BPM' if f == 'tempo' else ''}" if f == "tempo"
                           else f"{SHORT_LABELS[f].lower()} {t[f]:.2f}" for f in (list(m.target) or ["energy", "valence"])[:3])
        with cols[n % 2]:
            st.markdown(ui.track_card(engine, r, rank=int(r["rank"]), reason=reason, match=float(min(r["relevance"], 1.0)),
                                      discovery=float(r["discovery_score"])), unsafe_allow_html=True)
            ui.track_actions(engine, r, key=f"mode_{mode}_{r['idx']}", on_why=ui.go_why if seeds else None)
ui.footer()
