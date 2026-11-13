# Offline experiment results

Seed sets: 200 × 3 seeds, k = 10, random seed 42.

# Experiment report: strategy_comparison

* Run: 2026-11-13 18:23:27 (39.3 s)
* Seed sets: 200 x 3 seeds, random seed 42, top-k = 10
* Catalogue: 79,868 tracks

Relevance metrics are proxies (similarity to the seed profile and genre-label consistency). No listener judged these lists. Values are means over seed sets; ± values are 95% normal-approximation confidence half-widths.

| Metric | Content-based · raw features (baseline) | Content-based · SoundBridge Encoder | Popularity-ranked | Genre-aware | Popularity-aware discovery | Diversity-aware (MMR) | Hybrid (SoundBridge) |
|---|---|---|---|---|---|---|---|
| Relevance · encoder similarity (proxy) | 0.630 ± 0.016 | 0.802 ± 0.009 | 0.764 ± 0.009 | 0.757 ± 0.013 | 0.791 ± 0.009 | 0.791 ± 0.009 | 0.774 ± 0.010 |
| Relevance · raw-feature similarity (proxy) | 0.909 ± 0.010 | 0.758 ± 0.022 | 0.707 ± 0.023 | 0.707 ± 0.025 | 0.749 ± 0.021 | 0.744 ± 0.020 | 0.732 ± 0.021 |
| Genre-consistency precision@k (proxy) | 0.150 ± 0.026 | 0.272 ± 0.037 | 0.192 ± 0.031 | 0.998 ± 0.002 | 0.226 ± 0.038 | 0.249 ± 0.033 | 0.821 ± 0.034 |
| Intra-list diversity | 0.388 ± 0.011 | 0.248 ± 0.014 | 0.332 ± 0.014 | 0.314 ± 0.018 | 0.280 ± 0.014 | 0.344 ± 0.014 | 0.329 ± 0.016 |
| Artist diversity | 0.967 ± 0.011 | 0.952 ± 0.012 | 0.941 ± 0.012 | 0.814 ± 0.022 | 0.949 ± 0.010 | 0.977 ± 0.006 | 0.995 ± 0.003 |
| Genre diversity | 0.753 ± 0.025 | 0.670 ± 0.031 | 0.687 ± 0.026 | 0.133 ± 0.010 | 0.655 ± 0.031 | 0.733 ± 0.028 | 0.308 ± 0.034 |
| Genre entropy (bits) | 2.717 ± 0.074 | 2.444 ± 0.106 | 2.540 ± 0.081 | 0.184 ± 0.058 | 2.399 ± 0.104 | 2.633 ± 0.091 | 0.942 ± 0.142 |
| Audio-feature diversity | 0.193 ± 0.010 | 0.353 ± 0.015 | 0.401 ± 0.015 | 0.401 ± 0.016 | 0.377 ± 0.014 | 0.423 ± 0.016 | 0.414 ± 0.017 |
| Novelty (exposure-based) | 0.532 ± 0.011 | 0.526 ± 0.011 | 0.298 ± 0.011 | 0.504 ± 0.017 | 0.711 ± 0.012 | 0.527 ± 0.011 | 0.582 ± 0.015 |
| Mean popularity | 36.768 ± 1.185 | 37.602 ± 1.261 | 66.397 ± 1.114 | 39.145 ± 1.839 | 22.623 ± 1.087 | 37.986 ± 1.174 | 33.830 ± 1.579 |
| Lesser-known artist share | 0.213 ± 0.023 | 0.191 ± 0.023 | 0.003 ± 0.006 | 0.162 ± 0.032 | 0.494 ± 0.030 | 0.188 ± 0.023 | 0.289 ± 0.035 |
| Mean artist exposure | 49.477 ± 1.170 | 49.907 ± 1.185 | 71.785 ± 1.118 | 52.355 ± 1.769 | 30.798 ± 1.244 | 49.659 ± 1.174 | 43.626 ± 1.547 |
| Repeated-artist % | 0.033 ± 0.011 | 0.048 ± 0.012 | 0.059 ± 0.012 | 0.186 ± 0.022 | 0.051 ± 0.010 | 0.023 ± 0.006 | 0.005 ± 0.003 |
| Duplicate-track % | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Mean pairwise similarity | 0.612 ± 0.011 | 0.752 ± 0.014 | 0.668 ± 0.014 | 0.686 ± 0.018 | 0.720 ± 0.014 | 0.656 ± 0.014 | 0.671 ± 0.016 |
| Catalogue coverage | 0.024 | 0.024 | 0.022 | 0.024 | 0.023 | 0.024 | 0.024 |
| Artist coverage | 0.090 | 0.090 | 0.068 | 0.079 | 0.082 | 0.093 | 0.095 |
| Exposure Gini (lower = more even) | 0.180 | 0.180 | 0.330 | 0.257 | 0.230 | 0.159 | 0.142 |

## Configuration
```json
{
  "content_raw": "default",
  "content": "default",
  "popularity_ranked": "default",
  "genre_aware": "default",
  "popularity_aware": "default",
  "diversity_mmr": "default",
  "hybrid": "default"
}
```

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

# Experiment report: diversity_sweep

* Run: 2026-11-13 18:25:12 (34.8 s)
* Seed sets: 200 x 3 seeds, random seed 42, top-k = 10
* Catalogue: 79,868 tracks

Relevance metrics are proxies (similarity to the seed profile and genre-label consistency). No listener judged these lists. Values are means over seed sets; ± values are 95% normal-approximation confidence half-widths.

| Metric | Diversity-aware (MMR) (lambda=1.0) | Diversity-aware (MMR) (lambda=0.85) | Diversity-aware (MMR) (lambda=0.7) | Diversity-aware (MMR) (lambda=0.5) | Diversity-aware (MMR) (lambda=0.3) |
|---|---|---|---|---|---|
| Relevance · encoder similarity (proxy) | 0.802 ± 0.009 | 0.800 ± 0.009 | 0.791 ± 0.009 | 0.748 ± 0.008 | 0.654 ± 0.009 |
| Relevance · raw-feature similarity (proxy) | 0.757 ± 0.021 | 0.754 ± 0.021 | 0.744 ± 0.020 | 0.696 ± 0.019 | 0.604 ± 0.021 |
| Genre-consistency precision@k (proxy) | 0.270 ± 0.037 | 0.267 ± 0.036 | 0.249 ± 0.033 | 0.194 ± 0.023 | 0.292 ± 0.022 |
| Intra-list diversity | 0.249 ± 0.014 | 0.291 ± 0.014 | 0.344 ± 0.014 | 0.446 ± 0.012 | 0.586 ± 0.012 |
| Artist diversity | 0.962 ± 0.009 | 0.963 ± 0.008 | 0.977 ± 0.006 | 0.983 ± 0.006 | 0.992 ± 0.004 |
| Genre diversity | 0.676 ± 0.030 | 0.689 ± 0.030 | 0.733 ± 0.028 | 0.792 ± 0.022 | 0.797 ± 0.022 |
| Genre entropy (bits) | 2.468 ± 0.102 | 2.501 ± 0.100 | 2.633 ± 0.091 | 2.836 ± 0.061 | 2.834 ± 0.060 |
| Audio-feature diversity | 0.354 ± 0.015 | 0.385 ± 0.016 | 0.423 ± 0.016 | 0.489 ± 0.017 | 0.589 ± 0.018 |
| Novelty (exposure-based) | 0.526 ± 0.011 | 0.525 ± 0.011 | 0.527 ± 0.011 | 0.525 ± 0.010 | 0.519 ± 0.010 |
| Mean popularity | 37.621 ± 1.244 | 37.829 ± 1.211 | 37.986 ± 1.174 | 38.151 ± 1.064 | 37.703 ± 1.037 |
| Lesser-known artist share | 0.195 ± 0.023 | 0.192 ± 0.023 | 0.188 ± 0.023 | 0.193 ± 0.023 | 0.196 ± 0.021 |
| Mean artist exposure | 49.811 ± 1.177 | 49.960 ± 1.174 | 49.659 ± 1.174 | 50.197 ± 1.085 | 50.489 ± 1.107 |
| Repeated-artist % | 0.038 ± 0.009 | 0.037 ± 0.008 | 0.023 ± 0.006 | 0.016 ± 0.006 | 0.008 ± 0.004 |
| Duplicate-track % | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Mean pairwise similarity | 0.751 ± 0.014 | 0.709 ± 0.014 | 0.656 ± 0.014 | 0.554 ± 0.012 | 0.414 ± 0.012 |
| Catalogue coverage | 0.024 | 0.024 | 0.024 | 0.024 | 0.024 |
| Artist coverage | 0.091 | 0.091 | 0.093 | 0.092 | 0.093 |
| Exposure Gini (lower = more even) | 0.174 | 0.172 | 0.159 | 0.162 | 0.157 |

## Configuration
```json
{
  "diversity_mmr@lambda=1.0": {
    "relevance": 0.7,
    "novelty": 0.15,
    "diversity": 0.15,
    "genre": 0.2,
    "repetition": 0.2,
    "explore": 0.0,
    "max_per_artist": 2,
    "min_relevance": null,
    "max_popularity": null,
    "mmr_lambda": 1.0
  },
  "diversity_mmr@lambda=0.85": {
    "relevance": 0.7,
    "novelty": 0.15,
    "diversity": 0.15,
    "genre": 0.2,
    "repetition": 0.2,
    "explore": 0.0,
    "max_per_artist": 2,
    "min_relevance": null,
    "max_popularity": null,
    "mmr_lambda": 0.85
  },
  "diversity_mmr@lambda=0.7": {
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
  "diversity_mmr@lambda=0.5": {
    "relevance": 0.7,
    "novelty": 0.15,
    "diversity": 0.15,
    "genre": 0.2,
    "repetition": 0.2,
    "explore": 0.0,
    "max_per_artist": 2,
    "min_relevance": null,
    "max_popularity": null,
    "mmr_lambda": 0.5
  },
  "diversity_mmr@lambda=0.3": {
    "relevance": 0.7,
    "novelty": 0.15,
    "diversity": 0.15,
    "genre": 0.2,
    "repetition": 0.2,
    "explore": 0.0,
    "max_per_artist": 2,
    "min_relevance": null,
    "max_popularity": null,
    "mmr_lambda": 0.3
  }
}
```
