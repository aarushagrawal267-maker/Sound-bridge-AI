"""Hybrid recommendation engine and the shared entry point for every strategy.

Strategies (all except ``content_raw`` operate on SoundBridge Encoder embeddings):

content_raw        cosine similarity on z-scored raw audio features (non-learned baseline)
content            cosine similarity in the learned embedding space
popularity_ranked  the 100 most similar tracks re-sorted by popularity (exposure-biased baseline)
genre_aware        similarity + genre affinity / exploration
popularity_aware   relevance floor, then relevance + novelty (+ exposure gap)
diversity_mmr      Maximal Marginal Relevance with an artist cap
hybrid             weighted relevance + novelty + diversity + genre - repetition

Hybrid objective (greedy, one track at a time):

    score = w_rel * rel_norm + w_nov * novelty + w_div * diversity + w_gen * genre_term
            - w_rep * repetition - personal / feedback penalties

`rel_norm` is relevance min-max scaled inside the candidate pool and `diversity` is
1 - (max similarity to already chosen tracks) scaled by the pool's mean pairwise distance,
so the weights act on comparable [0, 1] quantities. The weights are configurable and
are *not* assumed optimal; the Experiment Lab compares presets.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .content_recommender import Candidates
from .diversity_ranking import greedy_rerank
from .musical_dna import TasteProfile

STRATEGIES: dict[str, str] = {
    "content_raw": "Content-based · raw features (baseline)",
    "content": "Content-based · SoundBridge Encoder",
    "popularity_ranked": "Popularity-ranked",
    "genre_aware": "Genre-aware",
    "popularity_aware": "Popularity-aware discovery",
    "diversity_mmr": "Diversity-aware (MMR)",
    "hybrid": "Hybrid (SoundBridge)",
}


@dataclass
class RankingWeights:
    relevance: float = 0.70
    novelty: float = 0.15
    diversity: float = 0.15
    genre: float = 0.20
    repetition: float = 0.20
    explore: float = 0.0           # 0 = stay in your genres, 1 = favour related new genres
    max_per_artist: int = 2
    min_relevance: float | None = None
    max_popularity: int | None = None
    mmr_lambda: float = 0.7

    def as_dict(self) -> dict:
        return asdict(self)


def weights_from_preferences(familiarity: float, discovery: float, exploration: float, diversity: float) -> RankingWeights:
    """Translate the four 0-1 interface controls into ranking weights."""
    return RankingWeights(
        relevance=round(0.35 + 0.65 * familiarity, 3),
        novelty=round(0.45 * discovery, 3),
        diversity=round(0.45 * diversity, 3),
        genre=round(0.45 + 0.05 * exploration, 3),
        repetition=round(0.15 + 0.25 * diversity, 3),
        explore=round(exploration, 3),
        max_per_artist=1 if diversity >= 0.6 else 2,
    )


def preset_weights(name: str, presets: dict) -> RankingWeights:
    p = presets[name]
    return RankingWeights(relevance=p["relevance"], novelty=p["novelty"], diversity=p["diversity"],
                          repetition=p["repetition"])


class Recommender:
    def __init__(self, engine):
        self.e = engine
        self.cfg = engine.cfg["recommendation"]

    # ------------------------------------------------------------------ helpers
    def _candidates(self, profile: TasteProfile, raw: bool = False, pool: int | None = None) -> Candidates:
        pool = pool or self.cfg["candidate_pool"]
        if raw:
            c = self.e.content_raw.candidates(profile, n=pool, seed_vectors=self.e.raw_z[profile.seed_idx])
        else:
            c = self.e.content.candidates(profile, n=pool, extra=self.genre_candidates(profile, n=pool // 2))
        return self._dedupe(c, profile)

    def _dedupe(self, c: Candidates, profile: TasteProfile) -> Candidates:
        """Drop other versions of seed songs and keep one version of each song (best match)."""
        keys = self.e.song_key[c.idx]
        seed_keys = set(self.e.song_key[profile.seed_idx])
        seen: set = set()
        keep = np.zeros(len(c.idx), dtype=bool)
        for j in np.argsort(-c.relevance):
            k = keys[j]
            if k in seed_keys or k in seen:
                continue
            seen.add(k)
            keep[j] = True
        return Candidates(c.idx[keep], c.relevance[keep], c.profile_sim[keep], c.best_seed_sim[keep], c.best_seed[keep])

    def genre_candidates(self, profile: TasteProfile, n: int = 300) -> np.ndarray:
        """Genre-aware retrieval: the n best sound matches among tracks sharing a seed genre.

        Without this step the embedding neighbourhood of e.g. rock seeds is dominated by
        sonically similar tracks filed under other labels, leaving the genre-aware term
        nothing to choose from."""
        gm = self.e.genre_model
        if not gm.available or profile.genre_vector is None or not np.any(profile.genre_vector > 0):
            return np.array([], dtype=np.int64)
        cols = np.where(profile.genre_vector > 0)[0]
        mask = gm.g[:, cols].sum(1) > 0
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return idx
        rel = self.e.content.score_all(profile)[idx]
        return idx[np.argsort(-rel)[:n]]

    @staticmethod
    def _minmax(x: np.ndarray) -> np.ndarray:
        lo, hi = float(np.min(x)), float(np.max(x))
        return (x - lo) / (hi - lo) if hi > lo else np.ones_like(x)

    def _pool_spread(self, idx: np.ndarray, sample: int = 200) -> float:
        rng = np.random.default_rng(0)
        s = idx if len(idx) <= sample else rng.choice(idx, sample, replace=False)
        v = self.e.embeddings[s]
        m = v @ v.T
        return float(max(1 - m[np.triu_indices(len(s), 1)].mean(), 1e-3)) if len(s) > 1 else 1.0

    def _frame(self, idx, relevance, cands: Candidates, strategy: str, **components) -> pd.DataFrame:
        idx = np.asarray(idx, dtype=np.int64)
        t = self.e.tracks.loc[idx]
        pos = pd.Series(np.arange(len(cands.idx)), index=cands.idx)
        where = pos.reindex(idx).to_numpy()
        best_seed = np.array(
            [cands.best_seed[int(w)] if not np.isnan(w) else 0 for w in where], dtype=int
        ) if len(idx) else np.array([], dtype=int)
        nov = self.e.novelty[idx]
        df = pd.DataFrame(
            {
                "idx": idx,
                "rank": np.arange(1, len(idx) + 1),
                "track_id": t["track_id"].to_numpy(),
                "track_name": t["track_name"].to_numpy(),
                "artists": t["artists"].str.replace(";", ", ").to_numpy(),
                "primary_artist": t["primary_artist"].to_numpy(),
                "album_name": t["album_name"].to_numpy(),
                "genre": t["genre_str"].to_numpy(),
                "popularity": t["popularity"].to_numpy(),
                "artist_exposure": self.e.artist_exposure[idx],
                "relevance": np.asarray(relevance, dtype=float),
                "novelty": nov,
                "strategy": strategy,
                "best_seed_pos": best_seed,
            }
        )
        for k, v in components.items():
            df[k] = np.asarray(v, dtype=float) if v is not None else np.nan
        return df

    def _finish(self, df: pd.DataFrame, profile: TasteProfile) -> pd.DataFrame:
        """Add discovery score and model-independent raw-feature similarity."""
        if df.empty:
            return df
        idx = df["idx"].to_numpy()
        gm = self.e.genre_model
        uv = profile.genre_vector if profile.genre_vector is not None else gm.user_vector(profile.seed_idx)
        aff = gm.affinity(idx, uv) if gm.available else np.zeros(len(idx))
        df["genre_affinity"] = aff
        df["discovery_score"] = 0.7 * df["novelty"] + 0.3 * (1 - aff)
        rp = self.e.raw_profile(profile.seed_idx)
        rz = self.e.raw_z[idx]
        df["raw_similarity"] = (rz @ rp) / np.maximum(np.linalg.norm(rz, axis=1) * np.linalg.norm(rp), 1e-12)
        df["exposure_gap"] = self.e.gap[idx]
        df["best_seed"] = [profile.seed_idx[min(int(p), len(profile.seed_idx) - 1)] for p in df["best_seed_pos"]]
        return df

    # ------------------------------------------------------------------ main entry
    def recommend(
        self,
        profile: TasteProfile,
        strategy: str = "hybrid",
        weights: RankingWeights | None = None,
        k: int | None = None,
        personal_scores: dict[int, float] | None = None,
    ) -> pd.DataFrame:
        k = k or self.cfg["top_k"]
        w = weights or RankingWeights()
        e = self.e

        if strategy == "content_raw":
            c = self._candidates(profile, raw=True)
            return self._finish(self._frame(c.idx[:k], c.relevance[:k], c, strategy), profile)

        c = self._candidates(profile)
        if len(c.idx) == 0:
            return pd.DataFrame()

        # feedback: penalise disliked genres and tracks similar to disliked ones
        penalty = e.genre_model.penalty(c.idx, profile.penalised_genres)
        if profile.disliked_idx:
            dsim = e.embeddings[c.idx] @ e.embeddings[profile.disliked_idx].T
            penalty = penalty + np.clip((dsim.max(1) - 0.9) * 5, 0, 1)
        rel = c.relevance.copy()
        if personal_scores:
            rel = rel + 0.05 * np.array([personal_scores.get(int(i), 0.5) - 0.5 for i in c.idx])

        if strategy == "content":
            order = np.argsort(-(rel - 0.5 * penalty))[:k]
            return self._finish(self._frame(c.idx[order], c.relevance[order], c, strategy), profile)

        if strategy == "popularity_ranked":
            top = np.argsort(-rel)[:100]
            order = top[np.argsort(-e.popularity[c.idx[top]], kind="stable")][:k]
            return self._finish(self._frame(c.idx[order], c.relevance[order], c, strategy), profile)

        artists = e.artists[c.idx]
        albums = e.albums[c.idx]
        rel_n = self._minmax(rel)
        nov = e.novelty[c.idx]
        uv = profile.genre_vector if profile.genre_vector is not None else e.genre_model.user_vector(profile.seed_idx)
        genre_term = e.genre_model.genre_term(c.idx, uv, w.explore)

        keep = np.ones(len(c.idx), dtype=bool)
        if w.min_relevance is not None:
            keep &= c.relevance >= w.min_relevance
        if w.max_popularity is not None:
            keep &= e.popularity[c.idx] <= w.max_popularity
        if keep.sum() < k:  # relax rather than return a short list
            keep = np.argsort(-rel) < max(k * 3, 30)
        sel = np.where(keep)[0]

        if strategy == "genre_aware":
            res = greedy_rerank(c.idx[sel], e.embeddings[c.idx[sel]], rel_n[sel], artists[sel], k,
                                genre=genre_term[sel], penalty=penalty[sel], w_rel=0.6, w_gen=0.4,
                                max_per_artist=None)
        elif strategy == "popularity_aware":
            floor = w.min_relevance if w.min_relevance is not None else float(np.quantile(c.relevance, 0.5))
            keep2 = sel[c.relevance[sel] >= floor]
            if len(keep2) < k:
                keep2 = sel[np.argsort(-c.relevance[sel])[: max(k * 3, 30)]]
            nw = max(w.novelty, 0.3)
            res = greedy_rerank(c.idx[keep2], e.embeddings[c.idx[keep2]], rel_n[keep2], artists[keep2], k,
                                novelty=nov[keep2] + 0.1 * e.gap[c.idx[keep2]], penalty=penalty[keep2],
                                w_rel=1 - nw, w_nov=nw, max_per_artist=w.max_per_artist)
        elif strategy == "diversity_mmr":
            res = greedy_rerank(c.idx[sel], e.embeddings[c.idx[sel]], rel_n[sel], artists[sel], k,
                                penalty=penalty[sel], w_rel=w.mmr_lambda, w_div=1 - w.mmr_lambda,
                                max_per_artist=w.max_per_artist, div_scale=self._pool_spread(c.idx[sel]))
        elif strategy == "hybrid":
            res = greedy_rerank(c.idx[sel], e.embeddings[c.idx[sel]], rel_n[sel], artists[sel], k,
                                novelty=nov[sel], genre=genre_term[sel], penalty=penalty[sel], albums=albums[sel],
                                w_rel=w.relevance, w_nov=w.novelty, w_div=w.diversity, w_gen=w.genre,
                                w_rep=w.repetition, max_per_artist=w.max_per_artist,
                                div_scale=self._pool_spread(c.idx[sel]))
        else:
            raise ValueError(f"Unknown strategy {strategy!r}; choose from {list(STRATEGIES)}")

        pos = pd.Series(np.arange(len(c.idx)), index=c.idx)
        raw_rel = c.relevance[pos.loc[res.idx].to_numpy()]
        df = self._frame(res.idx, raw_rel, c, strategy, score=res.score, diversity=res.diversity,
                         genre_score=res.genre, repetition=res.repetition)
        return self._finish(df, profile)
