"""Content-based filtering: cosine similarity with nearest-neighbour retrieval.

Two similarity spaces share this code:

* the learned space - SoundBridge Encoder embeddings (the production model)
* the raw space     - z-scored audio features (a transparent baseline for comparison)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .musical_dna import TasteProfile


def normalise_rows(m: np.ndarray) -> np.ndarray:
    return (m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return normalise_rows(np.atleast_2d(a)) @ normalise_rows(np.atleast_2d(b)).T


@dataclass
class Candidates:
    idx: np.ndarray
    relevance: np.ndarray        # blended profile/seed similarity in [-1, 1]
    profile_sim: np.ndarray
    best_seed_sim: np.ndarray
    best_seed: np.ndarray        # which seed (position) each candidate is closest to


class SimilaritySpace:
    """A normalised vector space with a nearest-neighbour index."""

    def __init__(self, vectors: np.ndarray, name: str = "embedding"):
        self.name = name
        self.vectors = normalise_rows(vectors)
        self.index = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=1).fit(self.vectors)

    def __len__(self) -> int:
        return len(self.vectors)

    def neighbours(self, query: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
        n = min(n, len(self.vectors))
        dist, idx = self.index.kneighbors(normalise_rows(np.atleast_2d(query)), n_neighbors=n)
        return idx, 1.0 - dist

    def similarity(self, query: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray:
        q = normalise_rows(np.atleast_2d(query))
        v = self.vectors if idx is None else self.vectors[idx]
        return (v @ q.T).squeeze(-1) if q.shape[0] == 1 else v @ q.T


class ContentRecommender:
    """Retrieve tracks similar to a taste profile.

    relevance = alpha * cos(track, profile centroid) + (1 - alpha) * max_seed cos(track, seed)

    The max-over-seeds term keeps listeners with mixed tastes (e.g. jazz + metal seeds)
    from being served only tracks that sit halfway between both styles.
    """

    def __init__(self, space: SimilaritySpace, alpha: float = 0.5):
        self.space = space
        self.alpha = alpha

    def profile_vectors(self, profile: TasteProfile, seed_vectors: np.ndarray | None = None):
        if seed_vectors is None:
            return profile.centroid, profile.seed_embeddings
        seeds = normalise_rows(seed_vectors)
        return normalise_rows(seeds.mean(0, keepdims=True))[0], seeds

    def candidates(
        self,
        profile: TasteProfile,
        n: int = 600,
        exclude: set[int] | None = None,
        seed_vectors: np.ndarray | None = None,
        extra: np.ndarray | None = None,
    ) -> Candidates:
        """Nearest neighbours of the centroid and of each seed, plus optional `extra`
        indices (e.g. the best matches inside the listener's own genres)."""
        centroid, seeds = self.profile_vectors(profile, seed_vectors)
        exclude = set(profile.excluded) | set(exclude or ())
        pools = [self.space.neighbours(centroid, n)[0][0]]
        if extra is not None and len(extra):
            pools.append(np.asarray(extra, dtype=np.int64))
        per_seed = max(50, n // max(1, len(seeds)))
        for s in seeds:
            pools.append(self.space.neighbours(s, per_seed)[0][0])
        idx = np.unique(np.concatenate(pools))
        if exclude:
            idx = idx[~np.isin(idx, list(exclude))]
        return self.score(idx, centroid, seeds)

    def score(self, idx: np.ndarray, centroid: np.ndarray, seeds: np.ndarray) -> Candidates:
        v = self.space.vectors[idx]
        psim = v @ centroid
        ssim = v @ seeds.T
        best = ssim.max(1)
        rel = self.alpha * psim + (1 - self.alpha) * best
        order = np.argsort(-rel)
        return Candidates(idx[order], rel[order], psim[order], best[order], ssim.argmax(1)[order])

    def score_all(self, profile: TasteProfile, seed_vectors: np.ndarray | None = None) -> np.ndarray:
        """Blended relevance for every catalogue track (exact, brute force; ~ms on CPU)."""
        centroid, seeds = self.profile_vectors(profile, seed_vectors)
        v = self.space.vectors
        return self.alpha * (v @ centroid) + (1 - self.alpha) * (v @ seeds.T).max(1)

    def recommend(self, profile: TasteProfile, k: int = 10, **kw) -> Candidates:
        c = self.candidates(profile, **kw)
        return Candidates(c.idx[:k], c.relevance[:k], c.profile_sim[:k], c.best_seed_sim[:k], c.best_seed[:k])
