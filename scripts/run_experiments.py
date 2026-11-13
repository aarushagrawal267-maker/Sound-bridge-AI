"""Step 3: reproducible offline experiments comparing recommendation strategies.

Usage:  python scripts/run_experiments.py [--sets 200] [--k 10]

Runs
1. strategy_comparison - every strategy with default settings
2. hybrid_presets      - the hybrid presets from config.yaml
3. novelty_sweep       - hybrid with novelty weight 0 .. 0.6 (relevance/novelty trade-off)
4. diversity_sweep     - MMR lambda 1.0 .. 0.3 (relevance/diversity trade-off)
Each run is saved under outputs/experiments/ with config, metrics and a Markdown report.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, path  # noqa: E402
from src.engine import SoundBridgeEngine  # noqa: E402
from src.experiment_tracker import run_experiment  # noqa: E402
from src.hybrid_recommender import STRATEGIES, RankingWeights, preset_weights  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", type=int, default=None)
    ap.add_argument("--k", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    n = args.sets or cfg["experiments"]["n_seed_sets"]
    k = args.k or cfg["experiments"]["k"]
    per = cfg["experiments"]["seeds_per_set"]
    seed = cfg["seed"]
    e = SoundBridgeEngine(load_model=False)

    runs = {
        "strategy_comparison": {s: None for s in STRATEGIES},
        "hybrid_presets": {f"hybrid@{name}": preset_weights(name, cfg["hybrid_presets"]) for name in cfg["hybrid_presets"]},
        "novelty_sweep": {f"hybrid@novelty={v}": RankingWeights(relevance=0.7, novelty=v, diversity=0.15)
                          for v in (0.0, 0.1, 0.2, 0.3, 0.45, 0.6)},
        "diversity_sweep": {f"diversity_mmr@lambda={v}": RankingWeights(mmr_lambda=v) for v in (1.0, 0.85, 0.7, 0.5, 0.3)},
    }
    lines = ["# Offline experiment results", "", f"Seed sets: {n} × {per} seeds, k = {k}, random seed {seed}.", ""]
    for name, strategies in runs.items():
        res = run_experiment(e, strategies, n_sets=n, per_set=per, k=k, seed=seed, name=name)
        print(f"\n=== {name} ({res['seconds']} s) -> {res['dir']}")
        print(res["summary"].round(3).T.to_string())
        lines += [Path(res["dir"]).joinpath("report.md").read_text(), ""]
    out = path("outputs_dir") / "reports" / "experiment_results.md"
    out.write_text("\n".join(lines))
    print("\nWrote", out)


if __name__ == "__main__":
    main()
