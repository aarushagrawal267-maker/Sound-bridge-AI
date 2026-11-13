import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.config import DNA_FEATURES, SHORT_LABELS
from src.feature_engineering import display_scale
from src.musical_dna import build_profile, describe_profile, feature_summary, taste_cohesion
from src.visualization import CATEGORICAL, SEQ, TRACK, _finish, feature_bars, hbar, radar

engine = ui.get_engine()
ui.page_header("Musical DNA", "Your Musical DNA",
               "A profile of the sound of the songs you picked — what the audio features say, and how the "
               "SoundBridge Encoder hears your taste.")
seeds = st.session_state.seeds
if not seeds:
    ui.empty_state("Your DNA is empty", "Add seed songs on Music Discovery (or try a ready-made taste on Home).", "🧬")
    if st.button("Go to Music Discovery", type="primary", icon=":material/travel_explore:"):
        st.switch_page("views/1_Music_Discovery.py")
    st.stop()

ui.note("Audio features are computational estimates from Spotify's audio analysis, not measurements of how a song "
        "feels. <b>Valence</b> estimates musical positiveness; it is not a reading of anyone's emotions.")

# interactive subset: watch the profile change as songs are toggled
labels = {i: engine.label(i) for i in seeds}
active = st.multiselect("Songs included in this profile — remove or re-add to watch your DNA change",
                        options=seeds, default=seeds, format_func=lambda i: labels[i])
if not active:
    st.warning("Select at least one song.")
    st.stop()

summ = feature_summary(engine.tracks, active)
desc = describe_profile(engine.tracks, active)
prof = build_profile(engine.embeddings, active)
coh = taste_cohesion(prof)

t_lo, t_hi = desc.get("tempo_range", (np.nan, np.nan))
ui.tiles([
    (f"{len(active)}", "Songs in profile", ""),
    (f"{desc.get('tempo_mean', 0):.0f} BPM", "Average tempo", f"range {t_lo:.0f}–{t_hi:.0f} BPM"),
    (desc.get("energy_mood", "—").split(",")[0], "Energy character", desc.get("energy_mood", "")),
    (f"{coh:.2f}", "Taste cohesion", "mean encoder similarity between your songs (1 = very consistent)"),
    (f"{desc.get('major_share', 0):.0%}", "Major mode", f"most common key: {desc.get('common_key') or '—'}"),
])

c1, c2 = st.columns([1.1, 1], gap="large")
with c1:
    ui.section("Radar profile", "your seeds vs. the catalogue average")
    st.plotly_chart(radar(dict(zip(summ.feature, summ.display_mean)),
                          baseline=dict(zip(summ.feature, summ.display_catalog)), height=430),
                    width="stretch", config={"displayModeBar": False})
with c2:
    ui.section("Audio feature bars", "hover for exact values")
    st.plotly_chart(feature_bars(summ, height=430), width="stretch", config={"displayModeBar": False})

ui.section("Musical characteristic summary")
traits = "".join(f"<li>{t[0].upper() + t[1:]}</li>" for t in desc["traits"])
gp = engine.genre_model.genres
probs = engine.genre_probs[active].mean(0)
top = np.argsort(-probs)[:5]
heard = ", ".join(f"<b>{gp[j]}</b> {probs[j]:.0%}" for j in top)
st.markdown(f'<ul class="sb-reasons">{traits}<li>The encoder\'s genre head hears your profile as: {heard} '
            f'(averaged genre probabilities — a model estimate, not a label).</li></ul>', unsafe_allow_html=True)

c3, c4 = st.columns(2, gap="large")
with c3:
    ui.section("Genre distribution", "labels of your seed songs")
    g = desc.get("genres", pd.Series(dtype=int))
    if len(g):
        st.plotly_chart(hbar(g.index, g.values, height=max(220, 34 * len(g) + 80), xtitle="seed songs carrying this label"),
                        width="stretch", config={"displayModeBar": False})
with c4:
    ui.section("Energy & mood map", "energy vs valence across the catalogue")
    sample = engine.tracks.sample(12000, random_state=0)
    fig = go.Figure(go.Histogram2d(x=sample["valence"], y=sample["energy"], nbinsx=30, nbinsy=30, colorscale=SEQ,
                                   showscale=False, hoverinfo="skip", opacity=0.55, xgap=1, ygap=1))
    s = engine.tracks.loc[active]
    fig.add_trace(go.Scatter(x=s["valence"], y=s["energy"], mode="markers", name="Your songs",
                             marker=dict(size=20, symbol="star", color=TRACK, line=dict(color="#ffffff", width=1.5)),
                             text=[labels[i] for i in active], hovertemplate="%{text}<br>valence %{x:.2f} · energy %{y:.2f}<extra></extra>"))
    for x, y, t in [(0.02, 1.0, "energetic · darker"), (0.98, 1.0, "energetic · brighter"),
                    (0.02, 0.0, "calm · darker"), (0.98, 0.0, "calm · brighter")]:
        fig.add_annotation(x=x, y=y, text=t, showarrow=False, font=dict(color="#ECEAF6", size=11),
                           bgcolor="rgba(18,17,28,.75)", xanchor="left" if x < .5 else "right",
                           yanchor="top" if y > .5 else "bottom")
    fig.update_layout(xaxis=dict(title="valence (musical positiveness estimate)", range=[0, 1]),
                      yaxis=dict(title="energy", range=[0, 1]), showlegend=False)
    st.plotly_chart(_finish(fig, 380), width="stretch", config={"displayModeBar": False})

ui.section("How your profile evolves", "feature means as each song is added, in order")
feats = [f for f in DNA_FEATURES if f in engine.tracks.columns]
disp = display_scale(engine.tracks.loc[active], feats)
cum = disp.expanding().mean().reset_index(drop=True)
fig = go.Figure()
focus = ["energy", "danceability", "acousticness", "valence", "instrumentalness", "speechiness"]
for color, f in zip(CATEGORICAL, focus):
    fig.add_trace(go.Scatter(x=list(range(1, len(cum) + 1)), y=cum[f], mode="lines+markers", name=SHORT_LABELS[f],
                             line=dict(color=color, width=2), marker=dict(size=8),
                             hovertemplate=SHORT_LABELS[f] + " after %{x} songs: %{y:.2f}<extra></extra>"))
fig.update_layout(xaxis=dict(title="songs included", dtick=1), yaxis=dict(title="profile mean (0–1)", range=[0, 1]))
st.plotly_chart(_finish(fig, 340), width="stretch", config={"displayModeBar": False})

with st.expander("Profile table (exact values)"):
    st.dataframe(summ[["label", "mean", "std", "min", "max", "catalog_mean", "catalog_percentile"]].rename(columns={
        "label": "Feature", "mean": "Your mean", "std": "Spread (SD)", "catalog_mean": "Catalogue mean",
        "catalog_percentile": "Percentile vs catalogue"}), hide_index=True, width="stretch")
ui.footer()
