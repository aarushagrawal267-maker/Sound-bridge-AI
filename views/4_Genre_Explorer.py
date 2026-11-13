import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

from src import ui
from src.config import DNA_FEATURES
from src.eda import genre_stats
from src.feature_engineering import display_scale
from src.visualization import ACCENT, MUTED_MARK, TRACK, _finish, hbar, heatmap, radar


@st.cache_data(show_spinner=False)
def _genre_stats(_engine):
    return genre_stats(_engine.tracks)


@st.cache_data(show_spinner=False)
def _confusions(_engine, n: int = 15) -> pd.DataFrame:
    """Most-confused genre pairs from the encoder's genre head on held-out-artist tracks."""
    from src.config import path

    test = np.load(path("models_dir") / "test_idx.npy")
    probs = _engine.genre_probs[test]
    pred = probs.argmax(1)
    gm = _engine.genre_multi_hot[test]
    wrong = gm[np.arange(len(test)), pred] == 0
    true_primary = _engine.tracks.loc[test, "primary_genre"].to_numpy()
    df = pd.DataFrame({"true": true_primary[wrong], "predicted": np.array(_engine.genres)[pred[wrong]]})
    df["pair"] = [" ↔ ".join(sorted(p)) for p in zip(df["true"], df["predicted"])]
    out = df.groupby("pair").size().sort_values(ascending=False).head(n).rename("confusions").reset_index()
    return out


engine = ui.get_engine()
gm = engine.genre_model
ui.page_header("Explore the catalogue", "Genre Explorer",
               "How the 114 genre labels sound, which ones the SoundBridge Encoder hears as neighbours, and where your "
               "taste could travel next.")
ui.note("Genre labels come from the search genre used to collect each track; a track can carry several. They are useful "
        "signals, not ground truth — which is exactly why the encoder learns relationships from sound.")

gs = _genre_stats(engine)
feats = [f for f in DNA_FEATURES if f in engine.tracks.columns]

genre = st.selectbox("Choose a genre", gm.genres, index=gm.genres.index("indie") if "indie" in gm.genres else 0)
row = gs.loc[genre]
ui.tiles([
    (f"{int(row['tracks']):,}", "Tracks", "with this label"),
    (f"{int(row['artists']):,}", "Artists", ""),
    (f"{row['popularity']:.0f}", "Mean popularity", f"median {row['median_popularity']:.0f}"),
    (f"{row['lesser_known_share']:.0%}", "Lesser-known share", "popularity 1–35"),
    (f"{row['tempo']:.0f} BPM", "Mean tempo", ""),
])

c1, c2 = st.columns([1, 1], gap="large")
with c1:
    ui.section(f"The sound of {genre}", "genre mean vs catalogue")
    mask = engine.tracks["genre"].map(lambda g: genre in g)
    gdisp = display_scale(engine.tracks.loc[mask], feats).mean()
    cdisp = display_scale(engine.tracks.sample(20000, random_state=0), feats).mean()
    st.plotly_chart(radar(gdisp.to_dict(), baseline=cdisp.to_dict(), profile_name=genre, height=400),
                    width="stretch", config={"displayModeBar": False})
with c2:
    ui.section("Sonic neighbours", "cosine similarity of genre centroids in embedding space")
    rel = gm.related_genres(genre, 10)
    st.plotly_chart(hbar(rel["genre"], rel["similarity"], height=400, fmt=".2f", xtitle="centroid similarity"),
                    width="stretch", config={"displayModeBar": False})

ui.section("Genre map", "every genre placed by the encoder — close points sound alike")
pca = PCA(2, random_state=0).fit_transform(gm.centroids)
sizes = np.sqrt(gm.counts / gm.counts.max()) * 26 + 6
selected = set([genre] + rel["genre"].tolist())
# label the selected neighbourhood plus genres that are well separated from already-labelled ones
span = np.ptp(pca, axis=0).max()
labelled = [i for i, g in enumerate(gm.genres) if g in selected]
for i in np.argsort(-gm.counts):
    if gm.genres[i] in selected:
        continue
    if all(np.linalg.norm(pca[i] - pca[j]) > span * 0.07 for j in labelled):
        labelled.append(i)
text = [g if i in labelled else "" for i, g in enumerate(gm.genres)]
fig = go.Figure(go.Scatter(
    x=pca[:, 0], y=pca[:, 1], mode="markers+text", text=text, hovertext=gm.genres,
    textposition="top center", textfont=dict(size=10, color=[("#ECEAF6" if g in selected else "#7D7896") for g in gm.genres]),
    marker=dict(size=sizes, color=[(TRACK if g == genre else ACCENT if g in selected else MUTED_MARK) for g in gm.genres],
                line=dict(color="#12111c", width=1.5), opacity=0.9),
    customdata=gm.counts, hovertemplate="%{hovertext}<br>%{customdata:,} tracks<extra></extra>"))
fig.update_layout(xaxis=dict(showticklabels=False, title="genre-centroid PC 1"), yaxis=dict(showticklabels=False, title="genre-centroid PC 2"))
st.plotly_chart(_finish(fig, 620), width="stretch")

c3, c4 = st.columns([1.1, 1], gap="large")
with c3:
    ui.section("Compare genres feature by feature")
    pick = st.multiselect("Genres", gm.genres, default=[genre] + rel["genre"].tolist()[:4], max_selections=10)
    if pick:
        m = gs.loc[pick, feats]
        z = (m - gs[feats].mean()) / gs[feats].std()
        st.plotly_chart(heatmap(z, "Genre feature means (z-score across genres)", height=120 + 36 * len(pick)),
                        width="stretch")
with c4:
    ui.section("Sonically interchangeable genres", "most frequent confusions of the encoder's genre head")
    conf = _confusions(engine)
    st.plotly_chart(hbar(conf["pair"], conf["confusions"], height=520, xtitle="held-out tracks misclassified across the pair"),
                    width="stretch", config={"displayModeBar": False})
    st.caption("Confusions between labels like these suggest the labels overlap in sound, which is useful for exploration.")

seeds = st.session_state.seeds
if seeds:
    ui.section("Genre exploration opportunities for you", "labels you have not picked, closest to your genre mix")
    prof = ui.current_profile(engine)
    opp = gm.opportunities(prof.genre_vector, 8)
    cols = st.columns(4)
    for i, r in enumerate(opp.itertuples()):
        cols[i % 4].markdown(f'<div class="sb-tile"><div class="v" style="font-size:1.2rem">{r.genre}</div>'
                             f'<div class="l">relatedness {r.relatedness:.2f}</div></div>', unsafe_allow_html=True)
    st.caption("Use the Genre exploration slider on Music Discovery, or the Genre exploration mode on Listening Preferences.")

ui.section(f"Most popular in {genre}", "add any of them as a seed")
top = engine.search("", limit=6, genre=genre)
cols = st.columns(2)
for n, (_, r) in enumerate(top.iterrows()):
    with cols[n % 2]:
        a, b = st.columns([6, 1.3], vertical_alignment="center")
        a.markdown(ui.track_card(engine, r), unsafe_allow_html=True)
        if b.button("Add", key=f"ge_add_{r['idx']}", icon=":material/add:", disabled=r["idx"] in seeds, width="stretch"):
            ui.add_seed(int(r["idx"]))
            st.rerun()
ui.footer()
