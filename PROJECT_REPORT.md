# SoundBridge AI — Project Report

**Research problem.** Can a music recommender balance user preference, musical diversity and discovery
opportunities for lesser-known artists, and explain its choices?

Every number in this report was produced by the code in this repository, on the dataset described below.
Re-running `python scripts/setup_all.py --experiments --figures` reproduces them; training and experiments are
seeded and were verified to reproduce exactly on a second run. No user-study results are reported, because no
participants have been run yet (section 9).

---

## 1. Dataset characteristics

**Source.** The Spotify Tracks Dataset (Kaggle, `maharshipandya/-spotify-tracks-dataset`): 114,000 rows, about
1,000 per Spotify search genre, 114 genres, 20 columns. It was obtained from two public GitHub mirrors that serve
byte-identical files (SHA-256 `b202fa49…77bd`).

**Schema detection.** The flexible loader found track ID, name, artists, album, genre, popularity, duration,
explicit and all 11 audio and structural features. It reported the following as **absent**: artist ID, album ID,
sub-genre, release date and image URL. They were not fabricated.

**Raw data quality.**

| Check | Result |
|---|---|
| Missing values | 1 row missing artist, album and track name |
| Tempo = 0 BPM (analysis failure) | 157 rows |
| Time signature = 0 | 163 rows |
| Duration outside (0, 6 h] | 0 rows |
| Rows duplicating a `track_id` (same track listed under several genres) | 24,259 |
| Same title and artists under a different `track_id` (alternate releases) | 8,534 |
| Popularity = 0 | 16,020 raw rows (4,635 after de-duplication) |

**Cleaned catalogue:** 79,868 unique songs, 17,635 primary artists (29,793 credited), 45,013 albums and 114 genre
labels. Mean popularity is 35.3 and the median is 35.

**Exploratory findings** (computed live in the Dataset Explorer; figures in `outputs/figures/`):
* The strongest feature relationship is energy and loudness (Spearman ρ = +0.76).
* No audio feature strongly tracks popularity. The largest |ρ| is instrumentalness (ρ = −0.19).
* Popular tracks (popularity ≥ 60) are less instrumental than lesser-known ones (1–25), by 0.61 SD. Other
  differences are smaller.
* Average popularity varies greatly by genre label, from 'pop' at 62 to 'iranian' at 2. This partly reflects
  which markets Spotify's popularity score measures.
* Artist exposure (each artist's best track popularity) is fairly evenly spread *in this sample*: Gini 0.26, with
  the top 10% of artists holding 18% of summed exposure. This is a consequence of the per-genre sampling. It is
  not evidence about the real music economy.
* **Sampling biases:** genre labels are near-uniform (468–996 tracks per label after de-duplication), tracks come
  from search results, popularity is a recency-weighted snapshot, and many genre labels are languages or regions.

## 2. Data cleaning and feature preparation
| Step | Rows before | Rows after |
|---|---|---|
| Exclude rows with missing or invalid core values | 114,000 | 113,836 |
| Merge rows sharing a `track_id` (genres become a list; popularity = max) | 113,836 | 89,578 |
| Merge alternative releases (normalised title and artists; keep the most popular, union of genres) | 89,578 | 79,868 |

* **No imputation of audio features.** Rows with invalid core values are excluded and counted.
* **Popularity 0 is treated as missing exposure information**, not as obscurity. It is excluded from popularity
  training, from Hidden Gems and from "lesser-known" labels, and is given neutral novelty when no artist-level
  signal exists.
* **Four separate feature sets:**
  1. Similarity: audio and structural features only.
  2. Artist and genre: multi-hot genres and artist codes, used as training targets and for genre logic.
  3. Popularity: popularity and tiers, used for discovery ranking only.
  4. Diversity: artist, album, genre and sound-cluster identifiers.

  Identifiers never enter similarity calculations; the tests check this.
* **Encoder inputs (15):** quantile-normal audio features, log duration, mode, key as sine and cosine, time
  signature and the explicit flag.
* **Raw baseline space:** z-scored values of the nine audio features.

## 3. The AI model: SoundBridge Encoder
A PyTorch multi-task MLP (15 → 256 → 256 → 32, LayerNorm, GELU, dropout 0.1) producing unit-length embeddings.

| Objective | Detail | Loss weight |
|---|---|---|
| Contrastive | InfoNCE, temperature 0.1. Positive = another track by the same primary artist (or a noise-augmented copy). Same-artist in-batch negatives are masked. | 1.0 |
| Genre | Cosine-prototype classifier, 114 labels, soft multi-label targets | 1.0 |
| Popularity | Smooth-L1 regression on popularity / 100, excluding popularity 0 | 0.5 |

* **Split:** artist-level. 15% of artists (2,645 artists, 12,110 tracks) are held out entirely.
* **Training:** 67,758 tracks, AdamW (lr 2e-3, weight decay 1e-4), one-cycle schedule, 40 epochs, batch size
  1,024, input noise σ = 0.08. Takes 143 s on a 4-core CPU. All three losses decrease steadily
  (`outputs/figures/training_curves.png`).

### 3.1 Held-out evaluation (RQ1)
Retrieval inside the held-out set, using 5,000 query tracks whose artist has at least one other held-out track:

| k = 10 | Encoder | Raw z-scored features | Untrained 15-d inputs |
|---|---|---|---|
| Same-artist precision@10 | **0.105** | 0.054 | 0.057 |
| Same-artist hit@10 | **0.434** | 0.282 | 0.307 |
| Genre agreement@10 (random: 0.016) | **0.227** | 0.137 | 0.147 |

| Prediction head | Encoder | Baseline on identical inputs |
|---|---|---|
| Genre top-1 accuracy | **0.256** | Logistic regression 0.198 |
| Genre top-5 accuracy | **0.547** | Logistic regression 0.467 |
| Popularity R² | 0.142 | XGBoost (400 trees) **0.169** |
| Popularity MAE | **12.85** | XGBoost 12.93; predicting the mean 14.78 |

**Robustness.** Retraining with different random seeds also changes which artists are held out:

| Seed (split) | Encoder artist P@10 | Raw artist P@10 | Ratio | Encoder genre agr.@10 | Raw genre agr.@10 | Genre top-5 | Popularity R² |
|---|---|---|---|---|---|---|---|
| 42 (deployed model) | 0.105 | 0.054 | 1.94× | 0.227 | 0.137 | 0.547 | 0.142 |
| 7 | 0.125 | 0.068 | 1.85× | 0.266 | 0.169 | 0.597 | 0.165 |
| 123 | 0.118 | 0.059 | 2.01× | 0.246 | 0.149 | 0.556 | 0.183 |

Absolute values vary with which artists are held out. The encoder's advantage over raw features (1.85–2.01×) is
stable across all three splits.

**Interpretation.**
* The learned space roughly doubles same-artist precision and raises genre agreement by about 65% relative to raw
  features, on artists never seen in training. This holds across seeds.
* Absolute precision remains modest, since audio descriptors alone cannot identify an artist reliably.
* Popularity is weakly predictable from sound for both models (R² ≤ 0.17). This supports treating popularity as
  exposure shaped by factors outside the audio, and it is why SoundBridge never treats popularity as quality.

### 3.2 Feature importance
* **Encoder genre head:** permutation importance (top-5 accuracy drop) ranks acousticness, energy and
  instrumentalness highest, followed by danceability, duration, valence and loudness. Key, mode and time signature
  contribute almost nothing.
* **XGBoost popularity baseline:** the largest mean |SHAP| values are for instrumentalness (2.4 points), energy,
  loudness and valence (`outputs/figures/shap_popularity_xgb.png`).
* Full table: `outputs/reports/feature_importance.csv`.

## 4. Recommendation algorithms
Full definitions are in README section 9. In brief:
* **Retrieval:** NearestNeighbors over encoder embeddings (centroid and per-seed queries) plus genre-aware retrieval
  inside the listener's genres, with song-level de-duplication against the seeds.
* **Strategies:**
  1. content-based on raw features (baseline)
  2. content-based on the encoder
  3. popularity-ranked (baseline)
  4. genre-aware
  5. popularity-aware discovery
  6. MMR
  7. hybrid: a weighted greedy reranker over relevance, novelty, diversity, genre affinity or exploration,
     repetition and feedback penalties
* **Hidden Gems:** relevance-percentile floor, then an artist-exposure filter, then a novelty-weighted rank with
  one track per artist.

### Development notes (problems found and fixed with data)
1. **Popularity-0 tracks were inflating "novelty".** For example, an alternate release by a well-known band scored
   as maximally novel. *Fix:* artist-level exposure, and popularity 0 treated as missing.
2. **For rock seeds, only 8% of the 600 nearest neighbours shared a seed genre.** As a result, even genre
   exploration 0 returned mostly other genres. *Fix:* genre-aware retrieval, which adds the best in-genre matches to
   the candidate pool. After the fix, the exploration control moves in-genre share from 67–100% at 0 to 0% at 1
   (test `test_genre_exploration_changes_genres`).
3. **An alternate version of a seed song** ("The Night We Met (feat. Phoebe Bridgers)" for the seed "The Night We
   Met") was being recommended. *Fix:* song-key de-duplication on bracket-stripped titles.

## 5. Ranking experiments
**Protocol.** 200 seed sets (random seed 42). Each set has 3 tracks from one randomly chosen genre, by different
artists, each with popularity ≥ 30. Each strategy produces a top-10 list. Values are means over seed sets.
Relevance metrics are **proxies**: no listener judged these lists.

### 5.1 Strategy comparison (RQ2, RQ3, RQ4, RQ5, RQ7)
| Metric | Raw features | Encoder | Popularity-ranked | Genre-aware | Pop.-aware discovery | MMR (λ 0.7) | **Hybrid** |
|---|---|---|---|---|---|---|---|
| Relevance, encoder sim. (proxy) | 0.630 | **0.802** | 0.764 | 0.757 | 0.791 | 0.791 | 0.774 |
| Relevance, raw-feature sim. (proxy) | **0.909** | 0.758 | 0.707 | 0.707 | 0.749 | 0.744 | 0.732 |
| Genre-consistency precision@10 (proxy) | 0.150 | 0.272 | 0.192 | **0.998** | 0.226 | 0.250 | 0.821 |
| Intra-list diversity | **0.388** | 0.248 | 0.332 | 0.314 | 0.280 | 0.344 | 0.329 |
| Artist diversity | 0.967 | 0.952 | 0.940 | 0.814 | 0.949 | 0.977 | **0.995** |
| Novelty (exposure-based) | 0.532 | 0.526 | 0.298 | 0.504 | **0.711** | 0.527 | 0.582 |
| Mean popularity | 36.8 | 37.6 | **66.4** | 39.1 | 22.6 | 38.0 | 33.8 |
| Lesser-known artist share | 0.213 | 0.191 | 0.003 | 0.162 | **0.494** | 0.188 | 0.290 |
| Repeated-artist % | 3.3% | 4.8% | 5.9% | 18.6% | 5.1% | 2.3% | **0.5%** |
| Duplicate-track % | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Artist coverage (union of 2,000 recs) | 0.090 | 0.090 | 0.068 | 0.079 | 0.082 | 0.093 | **0.095** |
| Exposure Gini (lower = more even) | 0.180 | 0.180 | 0.330 | 0.257 | 0.230 | 0.159 | **0.142** |

95% half-widths across seed sets are about ±0.01 for the similarity proxies, ±0.03–0.04 for the genre-consistency
precision and lesser-known share, and ±1.1–1.8 points for mean popularity (`ci95.csv` in each experiment folder).

### 5.2 Novelty sweep (RQ6)
Hybrid with relevance 0.7 and diversity 0.15, varying the novelty weight:

| Novelty weight | 0.0 | 0.1 | 0.2 | 0.3 | 0.45 | 0.6 |
|---|---|---|---|---|---|---|
| Relevance, encoder (proxy) | 0.774 | 0.775 | 0.773 | 0.772 | 0.767 | 0.760 |
| Relevance, raw features (proxy) | 0.732 | 0.733 | 0.733 | 0.733 | 0.727 | 0.718 |
| Lesser-known artist share | 0.183 | 0.250 | 0.327 | 0.401 | 0.508 | 0.585 |
| Mean popularity | 38.3 | 35.2 | 32.0 | 28.7 | 23.6 | 19.3 |
| Genre-consistency precision | 0.850 | 0.832 | 0.800 | 0.754 | 0.676 | 0.602 |
| Exposure Gini | 0.150 | 0.144 | 0.146 | 0.159 | 0.205 | 0.263 |

### 5.3 Diversity sweep (RQ5)
MMR with λ = 1.0 → 0.3: intra-list diversity rises from 0.249 to 0.586, mean pairwise similarity falls from 0.751
to 0.414, and repeated-artist % falls from 3.8% to 0.8%. Encoder relevance changes by 0.802 → 0.800 (λ 0.85),
0.791 (λ 0.7), 0.748 (λ 0.5) and 0.654 (λ 0.3). The trade-off is flat down to λ ≈ 0.7 and steep below 0.5.

### 5.4 Hybrid presets
| Preset | Relevance (enc.) | Intra-list div. | Lesser-known share | Genre precision | Exposure Gini |
|---|---|---|---|---|---|
| relevance_first | 0.790 | 0.266 | 0.178 | 0.694 | 0.174 |
| balanced | 0.774 | 0.329 | 0.290 | 0.821 | 0.142 |
| discovery | 0.761 | 0.357 | 0.417 | 0.822 | 0.161 |
| explorer | 0.734 | 0.432 | 0.409 | 0.896 | 0.157 |

## 6. Results by research question
* **RQ1: audio features and similarity.** Raw audio features carry real similarity signal: same-artist hit@10 of
  0.28, against near zero by chance. A representation trained with artist, genre and popularity supervision uses
  that signal much better: precision 0.105 vs 0.054, hit@10 0.434 vs 0.282, consistent across seeds.
* **RQ2: relevance of content-based recommendations.** On proxy relevance, the encoder strategy produces lists
  close to the seed profile (0.802), and they are more genre-consistent than raw-feature lists (0.272 vs 0.150).
  Raw-feature similarity naturally scores best on its own raw-feature proxy (0.909), which shows why one proxy
  must not be read alone. Whether listeners find these lists relevant is untested until the pilot runs.
* **RQ3: popularity and recommendation distribution.** Sorting relevant candidates by popularity nearly eliminates
  lesser-known artists (0.3% vs 19% for plain content) and almost doubles exposure concentration (Gini 0.330 vs
  0.180). In this dataset, popularity ranking amplifies exposure inequality.
* **RQ4: catalogue coverage.** Across 200 lists of 10, all strategies recommend about 2.2–2.4% of the catalogue;
  that figure is bounded by the 2,000 slots available. Artist coverage separates strategies more: hybrid 9.5% and
  MMR 9.3% vs popularity-ranked 6.8%. Diversity-aware strategies spread exposure over more artists with lower
  concentration.
* **RQ5: repetition.** The hybrid's repetition penalty and artist cap cut repeated-artist share from 4.8% (plain
  encoder) to 0.5%, with no duplicate tracks in any strategy. MMR controls within-list similarity smoothly (5.3).
* **RQ6: lesser-known artists without losing relevance.** Raising the novelty weight from 0 to 0.3 more than
  doubles the lesser-known artist share (18% → 40%). Encoder relevance changes by −0.002 and raw-feature relevance
  by +0.001, both well inside the ±0.01 intervals. Beyond 0.45, costs appear: relevance −0.014, genre consistency
  −0.25, and higher exposure concentration as lists pile onto the lowest-exposure artists. **Moderate discovery
  weighting is essentially free on these proxies; aggressive weighting is not.**
* **RQ7: strategy comparison.** No strategy wins every metric. Within ±0.03 relevance of the best encoder strategy,
  the hybrid gives the best artist diversity (0.995), the lowest exposure concentration (Gini 0.142), the highest
  artist coverage, 82% genre consistency and 1.5× the plain encoder's lesser-known share. The popularity-aware
  strategy maximises discovery (49% lesser-known) at a lower genre consistency.
* **RQ8: understandability.** Explanations are generated for every track from exact computations: feature
  comparisons, a cosine decomposition that sums exactly to the similarity, and integrated gradients whose
  completeness is verified in the tests. *Whether users understand them* is measured by the explanation ratings on
  *Why This Song?* and in the pilot study; no ratings have been collected yet.

## 7. Evaluation metrics
| Family | Metrics |
|---|---|
| Relevance | Encoder similarity, raw-feature similarity and genre-consistency precision@k (all proxies). Precision@k and NDCG@k from real likes and dislikes only. |
| Diversity | Intra-list diversity (mean pairwise cosine distance), artist and genre unique ratio, genre entropy, audio-feature diversity |
| Discovery | Novelty (artist-weighted exposure), lesser-known artist share, mean and median popularity, catalogue and artist coverage, exposure Gini |
| Repetition | Repeated-artist %, duplicate-track %, mean pairwise similarity |

## 8. Feedback system
Six feedback types update the session profile (Rocchio), the ranking weights (bounded, documented rules) and genre
penalties. A personal logistic-regression layer on encoder embeddings activates after 20 labels of both kinds.
Storage requires explicit consent. Events are kept as local JSON lines, excluded from git, and can be exported or
deleted. The tests verify that disliking a track removes it and that dislikes, genre penalties and weight
adjustments take effect.

**Collected feedback at the time of writing: 0 events.** The dashboard reports whatever is actually stored.

## 9. Real-world validation (pilot study)
**Implemented:** the *Pilot Study* page runs a consented, blind, randomised within-subjects A/B test, comparing a
relevance-only encoder baseline against the hybrid, with 8 tracks each from the participant's own seeds. It
measures:
* perceived relevance (1–5)
* perceived variety (1–5)
* the number of unfamiliar artists
* repetitiveness
* usefulness of explanations (1–5)
* an overall preference

It produces a report with means, bootstrap 95% intervals and preference counts.

**Status: not yet conducted. Sample size n = 0.** This project was built in an automated environment without access
to human participants, so no user results are reported or implied. To run the pilot, recruit 10–20 consenting
listeners and have each complete the page on this app. The report regenerates at
`outputs/reports/pilot_study_report.md`. With n ≤ 20, results should be read as descriptive; no significance claims
should be made.

## 10. Limitations
* **The sample is not Spotify:** per-genre search sampling, a snapshot popularity score, and genre labels that are
  search terms.
* **"Lesser-known" is relative to this sample.** Artist exposure is the best popularity among the artist's tracks
  *here*.
* **Relevance is evaluated by proxies.** Similarity proxies favour strategies built on the same space; two
  independent proxies are reported to limit this, but only user data can settle relevance.
* **Seed sets are synthetic:** three popular same-genre tracks. Real users' seeds may be more varied; mixed-genre
  profiles are handled by the max-over-seeds relevance term, but they are not evaluated separately here.
* **Audio features are coarse, algorithmic descriptors.** Valence and energy are not emotions. Lyrics, timbre
  detail and culture are absent.
* **Retrieval precision is modest in absolute terms** (0.105 same-artist precision@10), so improvements over the
  baseline should not be overstated.
* **Spotify's audio-features endpoint is unavailable to new applications**, so the catalogue cannot be extended with
  matching features from the official API.

## 11. Future research
* Run the pilot (n = 10–20), then a larger study with a between-subjects design to test whether moderate discovery
  weighting is also "free" for perceived relevance.
* Learn ranking weights online from feedback (for example, contextual bandits with exposure constraints).
* Replace or augment tabular features with learned audio embeddings from open audio corpora.
* Evaluate mixed-taste and cold-start profiles separately; study long-term exposure dynamics with simulation
  grounded in real feedback.
* Explore fairness constraints across genre and region labels, given the strong popularity differences between them.

---
*Artifacts:*
* `outputs/reports/model_evaluation.json`
* `outputs/reports/feature_importance.csv`
* `outputs/reports/seed_robustness.json`
* `outputs/reports/experiment_results.md`
* `outputs/experiments/*/` (config, summary, per-seed-set metrics, recommendations, CIs and report)
* `outputs/figures/*.png`
