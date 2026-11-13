# Experiment report: hybrid_presets

* Run: 2026-11-13 18:23:55 (27.2 s)
* Seed sets: 200 x 3 seeds, random seed 42, top-k = 10
* Catalogue: 79,868 tracks

Relevance metrics are proxies (similarity to the seed profile and genre-label consistency). No listener judged these lists. Values are means over seed sets; ± values are 95% normal-approximation confidence half-widths.

| Metric | Hybrid (SoundBridge) (relevance_first) | Hybrid (SoundBridge) (balanced) | Hybrid (SoundBridge) (discovery) | Hybrid (SoundBridge) (explorer) |
|---|---|---|---|---|
| Relevance · encoder similarity (proxy) | 0.790 ± 0.010 | 0.774 ± 0.010 | 0.761 ± 0.011 | 0.734 ± 0.012 |
| Relevance · raw-feature similarity (proxy) | 0.745 ± 0.022 | 0.732 ± 0.021 | 0.722 ± 0.021 | 0.686 ± 0.022 |
| Genre-consistency precision@k (proxy) | 0.694 ± 0.042 | 0.821 ± 0.034 | 0.823 ± 0.033 | 0.896 ± 0.024 |
| Intra-list diversity | 0.266 ± 0.014 | 0.329 ± 0.016 | 0.357 ± 0.017 | 0.432 ± 0.018 |
| Artist diversity | 0.962 ± 0.009 | 0.995 ± 0.003 | 0.998 ± 0.002 | 0.999 ± 0.001 |
| Genre diversity | 0.415 ± 0.037 | 0.308 ± 0.034 | 0.294 ± 0.033 | 0.238 ± 0.028 |
| Genre entropy (bits) | 1.401 ± 0.152 | 0.942 ± 0.142 | 0.882 ± 0.140 | 0.633 ± 0.122 |
| Audio-feature diversity | 0.370 ± 0.015 | 0.414 ± 0.017 | 0.433 ± 0.018 | 0.488 ± 0.019 |
| Novelty (exposure-based) | 0.518 ± 0.014 | 0.582 ± 0.015 | 0.660 ± 0.015 | 0.654 ± 0.015 |
| Mean popularity | 38.136 ± 1.526 | 33.830 ± 1.579 | 27.995 ± 1.510 | 28.903 ± 1.526 |
| Lesser-known artist share | 0.177 ± 0.027 | 0.289 ± 0.035 | 0.417 ± 0.037 | 0.409 ± 0.040 |
| Mean artist exposure | 50.473 ± 1.433 | 43.626 ± 1.547 | 35.643 ± 1.588 | 36.169 ± 1.585 |
| Repeated-artist % | 0.038 ± 0.009 | 0.005 ± 0.003 | 0.002 ± 0.002 | 0.001 ± 0.001 |
| Duplicate-track % | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Mean pairwise similarity | 0.734 ± 0.014 | 0.671 ± 0.016 | 0.643 ± 0.017 | 0.568 ± 0.018 |
| Catalogue coverage | 0.024 | 0.024 | 0.023 | 0.024 |
| Artist coverage | 0.091 | 0.095 | 0.093 | 0.093 |
| Exposure Gini (lower = more even) | 0.174 | 0.142 | 0.161 | 0.157 |

## Configuration
```json
{
  "hybrid@relevance_first": {
    "relevance": 1.0,
    "novelty": 0.0,
    "diversity": 0.0,
    "genre": 0.2,
    "repetition": 0.1,
    "explore": 0.0,
    "max_per_artist": 2,
    "min_relevance": null,
    "max_popularity": null,
    "mmr_lambda": 0.7
  },
  "hybrid@balanced": {
    "relevance": 0.7,
    "novelty": 0.15,
    "diversity": 0.15,
    "genre": 0.2,
    "repetition": 0.2,
    "explore": 0.0,
    "max_per_artist": 2,
    "min_relevance": null,
    "max_popularity": null,
    "mmr_lambda": 0.7
  },
  "hybrid@discovery": {
    "relevance": 0.55,
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
  "hybrid@explorer": {
    "relevance": 0.45,
    "novelty": 0.25,
    "diversity": 0.3,
    "genre": 0.2,
    "repetition": 0.25,
    "explore": 0.0,
    "max_per_artist": 2,
    "min_relevance": null,
    "max_popularity": null,
    "mmr_lambda": 0.7
  }
}
```