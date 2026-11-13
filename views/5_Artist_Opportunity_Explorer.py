import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.eda import artist_stats, genre_stats, popularity_concentration
from src.visualization import ACCENT, MUTED, _finish, histogram


@st.cache_data(show_spinner=False)
def _artists(_engine):
    a = artist_stats(_engine.tracks)
    a["exposure"] = a["max_popularity"]
    return a


@st.cache_data(show_spinner=False)
def _genres(_engine):
    return genre_stats(_engine.tracks)


engine = ui.get_engine()
ui.page_header("Explore the catalogue", "Artist Opportunity Explorer",
               "Who is in this dataset, how exposure is distributed across artists and genres, and which lesser-known "
               "artists fit your taste.")
ui.note("This analysis describes <b>this dataset's sample</b> of roughly 1,000 tracks per genre label collected through "
        "Spotify search. It cannot measure the full Spotify catalogue or the music industry's exposure distribution.")

arts = _artists(engine)
conc = popularity_concentration(engine.tracks)
ui.tiles([
    (f"{len(arts):,}", "Artists", "primary credited"),
    (f"{(arts['tracks'] == 1).mean():.0%}", "One-track artists", "appear with a single track"),
    (f"{conc['top10pct_share']:.0%}", "Top-10% share", "of summed artist exposure"),
    (f"{conc['artist_gini']:.2f}", "Artist Gini", "0 = equal exposure"),
    (f"{conc['artists_under_20']:,}", "Artists with exposure 1–20", "best track popularity"),
])

# ---------------------------------------------------------------- filters
with st.container(border=True):
    f1, f2, f3, f4 = st.columns(4)
    genres = f1.multiselect("Genre", engine.genres, placeholder="All genres")
    exp_range = f2.slider("Artist exposure (best track popularity)", 0, 100, (1, 100))
    energy = f3.slider("Mean energy", 0.0, 1.0, (0.0, 1.0), 0.05)
    acoustic = f4.slider("Mean acousticness", 0.0, 1.0, (0.0, 1.0), 0.05)
filt = arts[(arts["exposure"].between(*exp_range)) & (arts["energy"].between(*energy)) & (arts["acousticness"].between(*acoustic))]
if genres:
    filt = filt[filt["genres"].map(lambda s: any(g in s.split(", ") for g in genres))]
st.caption(f"{len(filt):,} artists match the filters.")

c1, c2 = st.columns(2, gap="large")
with c1:
    ui.section("Exposure distribution", "best track popularity per artist")
    st.plotly_chart(histogram(filt["exposure"], "", nbins=50, xtitle="artist exposure (0–100)"), width="stretch")
with c2:
    ui.section("Concentration curve", "share of total exposure held by the top x% of artists")
    v = np.sort(filt["exposure"].to_numpy())[::-1]
    if v.sum() > 0:
        cum = np.cumsum(v) / v.sum()
        x = np.arange(1, len(v) + 1) / len(v)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=cum, mode="lines", line=dict(color=ACCENT, width=2.5), name="Observed",
                                 hovertemplate="top %{x:.0%} of artists hold %{y:.0%}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, dash="dot", width=1.5), name="Equal exposure"))
        fig.update_layout(xaxis=dict(title="share of artists (most exposed first)", tickformat=".0%"),
                          yaxis=dict(title="share of summed exposure", tickformat=".0%"))
        st.plotly_chart(_finish(fig, 300), width="stretch")

ui.section("Artists", "sortable — click a column header")
show = filt.reset_index()[["primary_artist", "tracks", "exposure", "mean_popularity", "top_genre", "genres", "energy",
                           "danceability", "acousticness", "valence"]].sort_values("tracks", ascending=False).head(500)
st.dataframe(show, hide_index=True, width="stretch", height=360, column_config={
    "primary_artist": "Artist", "exposure": st.column_config.ProgressColumn("Exposure", min_value=0, max_value=100, format="%d"),
    "mean_popularity": st.column_config.NumberColumn("Mean popularity", format="%.0f"),
    "energy": st.column_config.NumberColumn(format="%.2f"), "danceability": st.column_config.NumberColumn(format="%.2f"),
    "acousticness": st.column_config.NumberColumn(format="%.2f"), "valence": st.column_config.NumberColumn(format="%.2f")})

ui.section("Genre representation & exposure", "which labels give lesser-known artists more room?")
gs = _genres(engine)
gs = gs.assign(artists_per_100=gs["artists"] / gs["tracks"] * 100)
fig = go.Figure(go.Scatter(
    x=gs["popularity"], y=gs["lesser_known_share"], mode="markers", text=gs.index,
    marker=dict(size=np.sqrt(gs["artists"]) * 1.2, color=ACCENT, opacity=0.75, line=dict(color="#12111c", width=1)),
    customdata=np.c_[gs["artists"], gs["tracks"]],
    hovertemplate="%{text}<br>mean popularity %{x:.0f}<br>lesser-known share %{y:.0%}<br>%{customdata[0]:,} artists · %{customdata[1]:,} tracks<extra></extra>"))
for g in list(gs["popularity"].nlargest(4).index) + list(gs["popularity"].nsmallest(4).index):
    fig.add_annotation(x=gs.loc[g, "popularity"], y=gs.loc[g, "lesser_known_share"], text=g, showarrow=False, yshift=14,
                       font=dict(size=10, color="#B9B5CF"))
fig.update_layout(xaxis_title="mean track popularity in genre", yaxis=dict(title="share of tracks with popularity 1–35", tickformat=".0%"))
st.plotly_chart(_finish(fig, 460, "Bubble size = number of artists"), width="stretch")
low_rep = gs.nsmallest(8, "artists")[["tracks", "artists", "popularity", "lesser_known_share"]]
st.caption("Fewest distinct artists per label (a few artists dominate these labels in this sample): "
           + ", ".join(f"{g} ({int(r.artists)} artists)" for g, r in low_rep.iterrows()))

seeds = st.session_state.seeds
if seeds:
    ui.section("Lesser-known artists that match your taste", "artist sound centroid vs your Musical DNA")
    prof = ui.current_profile(engine)
    max_exp = st.slider("Maximum artist exposure", 5, 60, 35, 5)
    t = engine.tracks
    ok = engine.exposure_known & (engine.artist_exposure <= max_exp) & ~np.isin(engine.artists, engine.artists[seeds])
    idx = np.where(ok)[0]
    df = pd.DataFrame({"artist": engine.artists[idx], "sim": engine.embeddings[idx] @ prof.centroid, "idx": idx})
    agg = df.groupby("artist").agg(match=("sim", "mean"), best=("sim", "max"), tracks=("idx", "size"),
                                   best_idx=("idx", lambda s: s.iloc[np.argmax(df.loc[s.index, "sim"])]))
    agg = agg[agg["tracks"] >= 2].sort_values("match", ascending=False).head(8)
    cols = st.columns(2)
    for n, (artist, r) in enumerate(agg.iterrows()):
        row = t.loc[int(r["best_idx"])]
        with cols[n % 2]:
            st.markdown(ui.track_card(engine, row, reason=f"<b>{artist}</b>: {int(r['tracks'])} tracks here · mean artist "
                                      f"match {r['match']:.0%} · exposure {int(engine.artist_exposure[int(r['best_idx'])])}",
                                      match=float(r["best"])), unsafe_allow_html=True)
    st.caption("Only artists with at least two tracks are ranked, so the artist-level match is not based on one song.")
ui.footer()
