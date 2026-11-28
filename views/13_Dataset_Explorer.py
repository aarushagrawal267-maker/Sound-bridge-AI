import json

import pandas as pd
import streamlit as st

from src import ui
from src.config import SHORT_LABELS, path
from src.data_loader import load_tracks
from src.eda import artist_stats, correlations, genre_stats, insights, overview, popular_vs_lesser
from src.visualization import CATEGORICAL, box_by_group, heatmap, hbar, histogram, scatter_density


@st.cache_data(show_spinner="Profiling the raw dataset…")
def _raw_overview():
    try:
        raw = load_tracks()
    except FileNotFoundError:
        return None, None
    miss = raw.isna().sum()
    return {"rows": len(raw), "cols": list(raw.columns), "missing": miss[miss > 0].to_dict(),
            "dup_ids": int(raw["track_id"].duplicated().sum()) if "track_id" in raw else 0}, raw[["popularity"]].describe()


@st.cache_data(show_spinner=False)
def _eda(_engine):
    t = _engine.tracks
    return overview(None, t), insights(t), genre_stats(t), artist_stats(t), correlations(t), popular_vs_lesser(t)


engine = ui.get_engine()
t = engine.tracks
ui.page_header("Explore the catalogue", "Dataset Explorer",
               "The real data behind SoundBridge: what it contains, how it was cleaned, how its audio features are "
               "distributed and where its sampling biases lie. Every insight below is computed live from the data.")

ov, ins, gs, arts, corr, pvl = _eda(engine)
raw_ov, _ = _raw_overview()
val_path = path("outputs_dir") / "reports" / "data_validation.json"
val = json.loads(val_path.read_text()) if val_path.exists() else {}

ui.tiles([
    (f"{val.get('raw_rows', raw_ov['rows'] if raw_ov else 0):,}", "Raw rows", "Kaggle Spotify Tracks Dataset"),
    (f"{ov['tracks']:,}", "Clean tracks", "unique songs after de-duplication"),
    (f"{ov['artists']:,}", "Primary artists", f"{ov['all_credited_artists']:,} credited in total"),
    (f"{ov['genres']}", "Genre labels", ""),
    (f"{ov['avg_popularity']:.1f}", "Average popularity", f"median {ov['median_popularity']:.0f}"),
    (f"{val.get('duplicate_track_id_rows', 0):,}", "Duplicate-ID rows", "same track under several genres"),
])

ui.section("Insights from the data", "generated dynamically")
st.markdown('<ul class="sb-reasons">' + "".join(f"<li>{i}</li>" for i in ins) + "</ul>", unsafe_allow_html=True)

tab_q, tab_dist, tab_rel, tab_genre, tab_bias = st.tabs(["Quality & cleaning", "Distributions", "Relationships", "Genres & artists", "Popular vs lesser-known"])

with tab_q:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Cleaning steps**")
        log_path = path("outputs_dir") / "reports" / "cleaning_log.csv"
        if log_path.exists():
            st.dataframe(pd.read_csv(log_path), hide_index=True, width="stretch")
        st.markdown("**Validation notes**")
        for n in val.get("notes", []):
            st.markdown(f"- {n}")
    with c2:
        st.markdown("**Missing values in the raw file**")
        miss = val.get("missing", raw_ov["missing"] if raw_ov else {})
        st.dataframe(pd.DataFrame({"column": list(miss), "missing": list(miss.values())}), hide_index=True, width="stretch")
        st.markdown("**Out-of-range values (excluded, never imputed)**")
        oor = val.get("out_of_range", {})
        st.dataframe(pd.DataFrame({"column": list(oor), "rows": list(oor.values())}), hide_index=True, width="stretch")
        st.markdown("**Schema detected by the flexible loader**")
        sch = val.get("schema", {})
        st.markdown(" ".join(('<span class="sb-chip">' + k + '</span>' if v else '<span class="sb-chip" style="opacity:.5">' + k + ' ✕</span>') for k, v in sch.items()),
                    unsafe_allow_html=True)
        st.caption("✕ = field not present in this dataset (e.g. artist IDs, album IDs, release dates, image URLs). "
                   "Absent fields are left absent — never fabricated.")

with tab_dist:
    c1, c2 = st.columns(2, gap="medium")
    c1.plotly_chart(histogram(t["popularity"], "Popularity distribution", 50, xtitle="popularity (0–100)"), width="stretch")
    c2.plotly_chart(histogram(t["danceability"], "Danceability distribution", 40, CATEGORICAL[0]), width="stretch")
    c1.plotly_chart(histogram(t["energy"], "Energy distribution", 40, CATEGORICAL[1]), width="stretch")
    c2.plotly_chart(histogram(t["acousticness"], "Acousticness distribution", 40, CATEGORICAL[2]), width="stretch")
    c1.plotly_chart(histogram(t["valence"], "Valence distribution", 40, CATEGORICAL[3], xtitle="valence (musical positiveness estimate)"),
                    width="stretch")
    c2.plotly_chart(histogram(t["tempo"], "Tempo distribution", 60, CATEGORICAL[4], xtitle="tempo (BPM)"), width="stretch")
    pick = st.selectbox("Any other feature", ["instrumentalness", "speechiness", "liveness", "loudness", "duration_ms"])
    st.plotly_chart(histogram(t[pick] / (60000 if pick == "duration_ms" else 1), pick.replace("_ms", " (minutes)"), 60),
                    width="stretch")

with tab_rel:
    c1, c2 = st.columns(2, gap="medium")
    c1.plotly_chart(scatter_density(t, "danceability", "popularity", "Popularity vs danceability"), width="stretch")
    c2.plotly_chart(scatter_density(t, "energy", "popularity", "Popularity vs energy"), width="stretch")
    st.plotly_chart(heatmap(corr, "Spearman correlations between audio features and popularity", height=520), width="stretch")

with tab_genre:
    c1, c2 = st.columns(2, gap="large")
    top_g = gs.sort_values("tracks", ascending=False)
    c1.plotly_chart(hbar(top_g.index[:25], top_g["tracks"][:25], "Genre distribution (25 largest labels)", height=720,
                         xtitle="tracks"), width="stretch")
    top_a = arts.head(25)
    c2.plotly_chart(hbar(top_a.index, top_a["tracks"], "Artist representation (25 most frequent artists)", height=720,
                         xtitle="tracks in dataset"), width="stretch")
    feat = st.selectbox("Feature by genre", ["energy", "danceability", "acousticness", "valence", "tempo", "popularity"])
    chosen = st.multiselect("Genres", gs.index.tolist(), default=["classical", "edm", "hip-hop", "jazz", "metal", "pop", "acoustic"])
    ex = t[["genre", feat]].explode("genre")
    ex = ex[ex["genre"].isin(chosen)]
    st.plotly_chart(box_by_group(ex, "genre", feat, f"{SHORT_LABELS.get(feat, feat)} by genre", order=chosen), width="stretch")

with tab_bias:
    st.markdown("**How do popular tracks (≥ 60) differ from lesser-known ones (1–25)?**")
    st.dataframe(pvl.round(3), hide_index=True, width="stretch")
    st.caption("Effect size in pooled standard deviations. With tens of thousands of tracks, even tiny differences have "
               "p ≈ 0, so judge the effect size, not the p-value.")
    st.markdown("""
**Sampling biases to keep in mind**
* Roughly 1,000 tracks were collected per genre label, so genres are far more evenly represented than in real listening.
* Tracks come from Spotify search results, which favour items Spotify already surfaces; truly obscure music is under-sampled.
* Popularity is a snapshot from the collection date and is recency-weighted.
* Genre labels reflect the search term used, not musicological classification; many labels are languages or regions.
* Popularity 0 often marks duplicate releases and is treated as missing exposure information.
""")
ui.footer()
