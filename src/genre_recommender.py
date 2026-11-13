"""Genre-aware recommendation.

Genre labels in the source dataset are the search genre under which each track was
collected; they are not authoritative and a track can carry several. This module:

* builds genre "centroids" in the learned embedding space and a genre-genre similarity
  matrix (related genres are discovered from sound, not hand-written)
* scores candidates by genre affinity to the listener's seeds, or by *exploration*:
  genres that differ from the listener's own but sit close to them in sound
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class GenreModel:
    def __init__(self, genre_multi_hot: np.ndarray, genres: list[str], embeddings: np.ndarray):
        self.genres = list(genres)
        self.g = genre_multi_hot.astype(np.float32)
        self.available = len(self.genres) > 0
        if not self.available:
            self.centroids = np.zeros((0, embeddings.shape[1]), dtype=np.float32)
            self.similarity = np.zeros((0, 0), dtype=np.float32)
            return
        counts = self.g.sum(0)
        cent = (self.g.T @ embeddings) / np.maximum(counts[:, None], 1)
        self.centroids = cent / np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-12)
        self.similarity = self.centroids @ self.centroids.T
        self.counts = counts
        rows = self.g.sum(1, keepdims=True)
        self.track_dist = self.g / np.maximum(rows, 1)  # per-track genre distribution

    def index(self, genre: str) -> int:
        return self.genres.index(genre)

    def user_vector(self, seed_idx: list[int], penalised: list[str] | None = None) -> np.ndarray:
        if not self.available:
            return np.zeros(0, dtype=np.float32)
        u = self.track_dist[seed_idx].mean(0)
        for gname in penalised or []:
            if gname in self.genres:
                u[self.index(gname)] = 0.0
        return u / max(u.sum(), 1e-12)

    def related_genres(self, genre: str, n: int = 8) -> pd.DataFrame:
        i = self.index(genre)
        sims = self.similarity[i].copy()
        sims[i] = -np.inf
        top = np.argsort(-sims)[:n]
        return pd.DataFrame({"genre": [self.genres[j] for j in top], "similarity": sims[top]})

    def affinity(self, idx: np.ndarray, user_vec: np.ndarray) -> np.ndarray:
        """Cosine similarity between candidate genre labels and the listener's genre mix."""
        if not self.available:
            return np.zeros(len(idx), dtype=np.float32)
        g = self.track_dist[idx]
        num = g @ user_vec
        den = np.linalg.norm(g, axis=1) * max(np.linalg.norm(user_vec), 1e-12)
        return np.where(den > 0, num / np.maximum(den, 1e-12), 0.0).astype(np.float32)

    def relatedness(self, idx: np.ndarray, user_vec: np.ndarray) -> np.ndarray:
        """How close (in sound) a candidate's genres are to the listener's genres, 0..1."""
        if not self.available:
            return np.zeros(len(idx), dtype=np.float32)
        per_genre = np.clip(self.similarity @ user_vec, 0, 1)  # similarity of each genre to user mix
        g = self.g[idx]
        return (g * per_genre).max(1).astype(np.float32)

    def exploration(self, idx: np.ndarray, user_vec: np.ndarray) -> np.ndarray:
        """Rewards genres that are new to the listener but sonically related."""
        return ((1 - self.affinity(idx, user_vec)) * self.relatedness(idx, user_vec)).astype(np.float32)

    def genre_term(self, idx: np.ndarray, user_vec: np.ndarray, explore: float) -> np.ndarray:
        """explore = 0 keeps to the listener's genres, 1 favours related new genres."""
        return ((1 - explore) * self.affinity(idx, user_vec) + explore * self.exploration(idx, user_vec)).astype(
            np.float32
        )

    def penalty(self, idx: np.ndarray, penalised: list[str]) -> np.ndarray:
        if not self.available or not penalised:
            return np.zeros(len(idx), dtype=np.float32)
        cols = [self.index(g) for g in penalised if g in self.genres]
        return (self.g[idx][:, cols].sum(1) > 0).astype(np.float32)

    def opportunities(self, user_vec: np.ndarray, n: int = 6) -> pd.DataFrame:
        """Genres the listener has not selected that are closest to their genre mix."""
        if not self.available:
            return pd.DataFrame(columns=["genre", "relatedness"])
        rel = self.similarity @ user_vec
        rel[user_vec > 0] = -np.inf
        top = np.argsort(-rel)[:n]
        return pd.DataFrame({"genre": [self.genres[j] for j in top], "relatedness": rel[top]})
