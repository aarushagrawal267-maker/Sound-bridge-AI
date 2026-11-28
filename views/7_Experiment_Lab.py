import json

import pandas as pd
import streamlit as st

from src import ui
from src.experiment_tracker import METRIC_LABELS, experiment_report, list_experiments, load_experiment, run_experiment
from src.hybrid_recommender import STRATEGIES, RankingWeights, preset_weights
from src.visualization import popularity_violin, strategy_metric_bars, tradeoff_scatter

engine = ui.get_engine()
ui.page_header("Research", "Recommendation Experiment Lab",
               "Compare recommendation strategies on hundreds of reproducible seed sets, test your own weightings, "
               "and export a report. This is where SoundBridge's claims are measured rather than asserted.")
ui.note("No listener rated these lists, so relevance here is a <b>proxy</b>: similarity to the seed profile (in the "
        "encoder space and, independently, in raw audio-feature space) and genre-label consistency. Real relevance comes "
        "from the Pilot Study and Feedback Dashboard.")

GROUPS = {
    "Relevance (proxy)": ["relevance_learned", "relevance_raw_features", "proxy_genre_precision"],
    "Diversity": ["intra_list_diversity", "artist_diversity", "genre_unique_ratio", "audio_feature_diversity"],
    "Discovery": ["novelty", "lesser_known_share", "mean_popularity", "catalog_coverage", "artist_coverage", "exposure_gini"],
    "Repetition": ["repeated_artist_pct", "duplicate_track_pct", "mean_pairwise_similarity"],
}


def show_results(summary: pd.DataFrame, ci: pd.DataFrame | None, recs: pd.DataFrame | None, key: str) -> None:
    labels = {**STRATEGIES}
    t1, t2, t3, t4 = st.tabs(["Trade-offs", "All metrics", "Popularity distribution", "Table"])
    with t1:
        c1, c2 = st.columns(2, gap="large")
        c1.plotly_chart(tradeoff_scatter(summary, "novelty", "relevance_learned", labels, "novelty (exposure-based)",
                                         "relevance (encoder similarity)", "Relevance vs novelty"), width="stretch", key=f"{key}_t1")
        c2.plotly_chart(tradeoff_scatter(summary, "intra_list_diversity", "relevance_raw_features", labels,
                                         "intra-list diversity", "relevance (raw-feature similarity)", "Relevance vs diversity"),
                        width="stretch", key=f"{key}_t2")
    with t2:
        for group, metrics in GROUPS.items():
            st.markdown(f"#### {group}")
            cols = st.columns(2, gap="medium")
            for i, m in enumerate([m for m in metrics if m in summary.columns]):
                fmt = ".1f" if m == "mean_popularity" else ".3f"
                cols[i % 2].plotly_chart(strategy_metric_bars(summary, m, labels, highlight="hybrid", title=METRIC_LABELS[m],
                                                              ci=ci, height=90 + 34 * len(summary), fmt=fmt),
                                         width="stretch", key=f"{key}_{m}")
    with t3:
        if recs is not None and len(recs):
            st.plotly_chart(popularity_violin(recs, "strategy", labels, "Popularity of recommended tracks"),
                            width="stretch", key=f"{key}_violin")
        else:
            st.caption("Per-track data not available for this run.")
    with t4:
        st.dataframe(summary.rename(columns=METRIC_LABELS).T, width="stretch")


tab_saved, tab_run = st.tabs(["Saved experiments", "Run a new experiment"])

with tab_saved:
    exps = list_experiments()
    if exps.empty:
        ui.empty_state("No experiments yet", "Run one in the next tab, or run `python scripts/run_experiments.py`.")
    else:
        default_i = int(exps.index[exps["name"] == "strategy_comparison"][0]) if (exps["name"] == "strategy_comparison").any() else 0
        choice = st.selectbox("Experiment", exps.index, index=default_i, format_func=lambda i: f"{exps.loc[i, 'name']} · {exps.loc[i, 'timestamp']} · "
                                                                             f"{exps.loc[i, 'n_sets']} seed sets")
        exp = load_experiment(exps.loc[choice, "dir"])
        cfg = exp["config"]
        ui.tiles([(f"{cfg['n_sets']}", "Seed sets", f"{cfg['per_set']} seeds each"), (f"{cfg['k']}", "List length", ""),
                  (f"{cfg['seed']}", "Random seed", "reproducible"), (f"{len(exp['summary'])}", "Configurations", "")])
        show_results(exp["summary"], exp["ci95"], exp["recommendations"], key="saved")
        c1, c2 = st.columns(2)
        c1.download_button("Download report (Markdown)", exp["report"], file_name=f"{cfg['name']}_report.md",
                           icon=":material/download:", width="stretch")
        c2.download_button("Download configuration (JSON)", json.dumps(cfg, indent=2), file_name=f"{cfg['name']}_config.json",
                           icon=":material/data_object:", width="stretch")
        with st.expander("Report"):
            st.markdown(exp["report"])

with tab_run:
  if ui.admin_only("Running new experiments"):
    st.markdown("Pick strategies, add custom hybrid weightings, and run. Every run is saved to `outputs/experiments/`.")
    strategies = st.multiselect("Strategies", list(STRATEGIES), default=["content_raw", "content", "popularity_ranked", "diversity_mmr", "hybrid"],
                                format_func=STRATEGIES.get)
    presets = st.multiselect("Hybrid presets (from config.yaml)", list(engine.cfg["hybrid_presets"]), default=[])
    with st.expander("Add a custom hybrid weighting"):
        use_custom = st.checkbox("Include custom weighting")
        c = st.columns(5)
        custom = RankingWeights(
            relevance=c[0].number_input("Relevance", 0.0, 2.0, 0.6, 0.05),
            novelty=c[1].number_input("Novelty", 0.0, 2.0, 0.3, 0.05),
            diversity=c[2].number_input("Diversity", 0.0, 2.0, 0.2, 0.05),
            genre=c[3].number_input("Genre", 0.0, 2.0, 0.45, 0.05),
            repetition=c[4].number_input("Repetition penalty", 0.0, 2.0, 0.2, 0.05),
        )
    c1, c2, c3, c4 = st.columns(4)
    n_sets = c1.select_slider("Seed sets", [20, 50, 100, 200], value=50)
    k = c2.select_slider("List length k", [5, 10, 20], value=10)
    seed = c3.number_input("Random seed", 0, 10_000, 42)
    name = c4.text_input("Experiment name", "my_experiment")
    if st.button("Run experiment", type="primary", icon=":material/play_arrow:"):
        config = {s: None for s in strategies}
        for p in presets:
            config[f"hybrid@{p}"] = preset_weights(p, engine.cfg["hybrid_presets"])
        if use_custom:
            config["hybrid@custom"] = custom
        if not config:
            st.warning("Select at least one strategy")
        else:
            bar = st.progress(0.0, text="Running…")
            res = run_experiment(engine, config, n_sets=n_sets, k=k, seed=int(seed), name=name.strip().replace(" ", "_") or "experiment",
                                 progress=lambda f: bar.progress(f, text=f"Running… {f:.0%}"))
            bar.empty()
            st.success(f"Saved to {res['dir']} ({res['seconds']} s)")
            show_results(res["summary"], res["ci95"], res["recommendations"], key="new")
            st.download_button("Download report (Markdown)", experiment_report(res), file_name=f"{name}_report.md",
                               icon=":material/download:")
ui.footer()
