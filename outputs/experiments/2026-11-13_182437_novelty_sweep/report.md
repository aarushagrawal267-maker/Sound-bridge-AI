# Experiment report: novelty_sweep

* Run: 2026-11-13 18:24:37 (41.7 s)
* Seed sets: 200 x 3 seeds, random seed 42, top-k = 10
* Catalogue: 79,868 tracks

Relevance metrics are proxies (similarity to the seed profile and genre-label consistency). No listener judged these lists. Values are means over seed sets; ± values are 95% normal-approximation confidence half-widths.

| Metric | Hybrid (SoundBridge) (novelty=0.0) | Hybrid (SoundBridge) (novelty=0.1) | Hybrid (SoundBridge) (novelty=0.2) | Hybrid (SoundBridge) (novelty=0.3) | Hybrid (SoundBridge) (novelty=0.45) | Hybrid (SoundBridge) (novelty=0.6) |
|---|---|---|---|---|---|---|
| Relevance · encoder similarity (proxy) | 0.774 ± 0.010 | 0.775 ± 0.010 | 0.773 ± 0.010 | 0.772 ± 0.010 | 0.767 ± 0.010 | 0.760 ± 0.011 |
| Relevance · raw-feature similarity (proxy) | 0.732 ± 0.021 | 0.733 ± 0.021 | 0.733 ± 0.021 | 0.733 ± 0.020 | 0.727 ± 0.021 | 0.718 ± 0.021 |
| Genre-consistency precision@k (proxy) | 0.850 ± 0.030 | 0.832 ± 0.032 | 0.800 ± 0.036 | 0.753 ± 0.039 | 0.676 ± 0.045 | 0.603 ± 0.046 |
| Intra-list diversity | 0.327 ± 0.016 | 0.328 ± 0.016 | 0.330 ± 0.016 | 0.332 ± 0.016 | 0.344 ± 0.017 | 0.360 ± 0.017 |
| Artist diversity | 0.994 ± 0.003 | 0.995 ± 0.003 | 0.994 ± 0.003 | 0.995 ± 0.003 | 0.995 ± 0.003 | 0.995 ± 0.003 |
| Genre diversity | 0.291 ± 0.032 | 0.303 ± 0.033 | 0.320 ± 0.034 | 0.346 ± 0.035 | 0.395 ± 0.038 | 0.443 ± 0.037 |
| Genre entropy (bits) | 0.866 ± 0.136 | 0.921 ± 0.140 | 0.992 ± 0.144 | 1.115 ± 0.148 | 1.322 ± 0.154 | 1.533 ± 0.152 |
| Audio-feature diversity | 0.410 ± 0.017 | 0.410 ± 0.017 | 0.414 ± 0.017 | 0.419 ± 0.018 | 0.425 ± 0.017 | 0.437 ± 0.017 |
| Novelty (exposure-based) | 0.517 ± 0.015 | 0.560 ± 0.015 | 0.605 ± 0.014 | 0.649 ± 0.014 | 0.712 ± 0.014 | 0.766 ± 0.013 |
| Mean popularity | 38.328 ± 1.690 | 35.218 ± 1.628 | 32.041 ± 1.555 | 28.721 ± 1.480 | 23.552 ± 1.329 | 19.267 ± 1.242 |
| Lesser-known artist share | 0.183 ± 0.030 | 0.249 ± 0.034 | 0.327 ± 0.036 | 0.401 ± 0.036 | 0.508 ± 0.035 | 0.585 ± 0.034 |
| Mean artist exposure | 50.376 ± 1.617 | 46.018 ± 1.563 | 41.268 ± 1.513 | 36.887 ± 1.555 | 30.351 ± 1.499 | 24.652 ± 1.411 |
| Repeated-artist % | 0.006 ± 0.003 | 0.005 ± 0.003 | 0.006 ± 0.003 | 0.005 ± 0.003 | 0.004 ± 0.003 | 0.005 ± 0.003 |
| Duplicate-track % | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Mean pairwise similarity | 0.673 ± 0.016 | 0.672 ± 0.016 | 0.670 ± 0.016 | 0.668 ± 0.016 | 0.656 ± 0.017 | 0.640 ± 0.017 |
| Catalogue coverage | 0.024 | 0.024 | 0.024 | 0.024 | 0.023 | 0.022 |
| Artist coverage | 0.094 | 0.095 | 0.095 | 0.093 | 0.087 | 0.079 |
| Exposure Gini (lower = more even) | 0.150 | 0.144 | 0.146 | 0.159 | 0.205 | 0.263 |

## Configuration
```json
{
  "hybrid@novelty=0.0": {
    "relevance": 0.7,
    "novelty": 0.0,
    "diversity": 0.15,
    "genre": 0.2,
    "repetition": 0.2,
    "explore": 0.0,
    "max_per_artist": 2,
    "min_relevance": null,
    "max_popularity": null,
    "mmr_lambda": 0.7
  },
  "hybrid@novelty=0.1": {
    "relevance": 0.7,
    "novelty": 0.1,
    "diversity": 0.15,
    "genre": 0.2,
    "repetition": 0.2,
    "explore": 0.0,
    "max_per_artist": 2,
    "min_relevance": null,
    "max_popularity": null,
    "mmr_lambda": 0.7
  },
  "hybrid@novelty=0.2": {
    "relevance": 0.7,
    "novelty": 0.2,
    "diversity": 0.15,
    "genre": 0.2,
    "repetition": 0.2,
    "explore": 0.0,
    "max_per_artist": 2,
    "min_relevance": null,
    "max_popularity": null,
    "mmr_lambda": 0.7
  },
  "hybrid@novelty=0.3": {
    "relevance": 0.7,
    "novelty": 0.3,
    "diversity": 0.15,
    "genre": 0.2,
    "repetition": 0.2,
    "explore": 0.0,
    "max_per_artist": 2,
    "min_relevance": null,
    "max_popularity": null,
    "mmr_lambda": 0.7
  },
  "hybrid@novelty=0.45": {
    "relevance": 0.7,
    "novelty": 0.45,
    "diversity": 0.15,
    "genre": 0.2,
    "repetition": 0.2,
    "explore": 0.0,
    "max_per_artist": 2,
    "min_relevance": null,
    "max_popularity": null,
    "mmr_lambda": 0.7
  },
  "hybrid@novelty=0.6": {
    "relevance": 0.7,
    "novelty": 0.6,
    "diversity": 0.15,
    "genre": 0.2,
    "repetition": 0.2,
    "explore": 0.0,
    "max_per_artist": 2,
    "min_relevance": null,
    "max_popularity": null,
    "mmr_lambda": 0.7
  }
}
```