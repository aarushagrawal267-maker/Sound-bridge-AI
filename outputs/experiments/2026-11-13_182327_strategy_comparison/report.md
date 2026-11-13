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