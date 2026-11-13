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