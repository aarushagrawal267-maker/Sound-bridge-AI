"""SoundBridgeEngine: loads the trained model and all artifacts once, and exposes the
retrieval, profiling and explanation services used by every page.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property

import joblib
import numpy as np
import pandas as pd

from .config import DNA_FEATURES, load_config, path
from .content_recommender import ContentRecommender, SimilaritySpace
from .feature_engineering import genre_matrix
from .genre_recommender import GenreModel
from .musical_dna import TasteProfile, build_profile
from .popularity_ranking import artist_exposure, blended_novelty, exposure_gap, exposure_known


class ArtifactsMissing(RuntimeError):
    pass


@dataclass
class FeedbackState:
    liked: list[int] = field(default_factory=list)
    disliked: list[int] = field(default_factory=list)
    penalised_genres: list[str] = field(default_factory=list)


class SoundBridgeEngine:
    def __init__(self, load_model: bool = True):
        self.cfg = load_config()
        mdir, pdir = path("models_dir"), path("processed_dir")
        needed = [pdir / "tracks.parquet", mdir / "embeddings.npy", mdir / "soundbridge_encoder.pt"]
        missing = [str(p) for p in needed if not p.exists()]
        if missing:
            raise ArtifactsMissing(
                "Model artifacts not found: " + ", ".join(missing)
                + ". Run `python scripts/setup_all.py` (or build_artifacts.py then train_model.py)."
            )
        self.tracks = pd.read_parquet(pdir / "tracks.parquet")
        self.tracks["genre"] = self.tracks["genre"].map(list)
        self.tracks["artist_list"] = self.tracks["artist_list"].map(list)
        self.pipe = joblib.load(mdir / "feature_pipeline.joblib")
        self.embeddings = np.load(mdir / "embeddings.npy")
        self.raw_z = np.load(mdir / "raw_features.npy")
        self.genre_probs = np.load(mdir / "genre_probs.npy").astype(np.float32)
        self.pred_popularity = np.load(mdir / "pred_popularity.npy")
        self.clusters = np.load(mdir / "clusters.npy").astype(int)
        self.coords = np.load(mdir / "coords_2d.npy")
        self.dna_features = [f for f in DNA_FEATURES if f in self.tracks.columns]
        self.popularity = self.tracks["popularity"].to_numpy(dtype=np.float32)
        self.artists = self.tracks["primary_artist"].to_numpy()
        self.albums = self.tracks["album_name"].to_numpy()
        gm, self.genres = genre_matrix(self.tracks)
        self.genre_multi_hot = gm
        self.genre_model = GenreModel(gm, self.genres, self.embeddings)
        self.space = SimilaritySpace(self.embeddings, "SoundBridge Encoder")
        self.raw_space = SimilaritySpace(self.raw_z, "Raw audio features")
        self.content = ContentRecommender(self.space)
        self.content_raw = ContentRecommender(self.raw_space)
        self.catalog_std = self.tracks[self.dna_features].astype(float).std()
        self.gap = exposure_gap(self.pred_popularity, self.popularity)
        self.artist_exposure = artist_exposure(self.tracks["primary_artist"], self.tracks["popularity"])
        self.novelty = blended_novelty(self.popularity, self.artist_exposure)
        self.exposure_known = exposure_known(self.popularity, self.artist_exposure)
        self.tracks["artist_exposure"] = self.artist_exposure.astype(int)
        self.lesser_known = int(self.cfg["recommendation"]["lesser_known_popularity"])
        from .preprocessing import base_title

        # song identity used to stop alternate versions of the same song appearing together
        self.song_key = (self.tracks["track_name"].map(base_title) + "|" + self.tracks["primary_artist"].str.lower()).to_numpy()
        self.model = None
        self.model_meta: dict = {}
        if load_model:
            from .model import SoundBridgeEncoder

            self.model, self.model_meta = SoundBridgeEncoder.load(mdir / "soundbridge_encoder.pt")
        self._search_key = (
            self.tracks["track_name"].str.lower() + " — " + self.tracks["artists"].str.lower().str.replace(";", ", ")
        ).to_numpy()

    # ------------------------------------------------------------------ data access
    @cached_property
    def model_inputs(self) -> np.ndarray:
        return self.pipe.model_inputs(self.tracks)

    @cached_property
    def evaluation(self) -> dict:
        p = path("outputs_dir") / "reports" / "model_evaluation.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def __len__(self) -> int:
        return len(self.tracks)

    def search(self, query: str, limit: int = 25, genre: str | None = None) -> pd.DataFrame:
        q = (query or "").strip().lower()
        mask = np.ones(len(self.tracks), dtype=bool)
        if q:
            mask &= np.char.find(self._search_key.astype(str), q) >= 0
        if genre and genre != "All genres":
            mask &= self.tracks["genre"].map(lambda g: genre in g).to_numpy()
        res = self.tracks.loc[mask]
        if q:
            starts = res["track_name"].str.lower().str.startswith(q) | res["primary_artist"].str.lower().str.startswith(q)
            res = res.assign(_s=starts.astype(int)).sort_values(["_s", "popularity"], ascending=False).drop(columns="_s")
        else:
            res = res.sort_values("popularity", ascending=False)
        return res.head(limit)

    def label(self, idx: int) -> str:
        r = self.tracks.loc[idx]
        return f"{r['track_name']} — {r['artists'].replace(';', ', ')}"

    # ------------------------------------------------------------------ profiles
    def profile(self, seed_idx: list[int], feedback: FeedbackState | None = None) -> TasteProfile:
        fb = feedback or FeedbackState()
        uv = self.genre_model.user_vector(list(seed_idx) + list(fb.liked), fb.penalised_genres)
        return build_profile(
            self.embeddings,
            list(seed_idx),
            liked_idx=fb.liked,
            disliked_idx=fb.disliked,
            learning_rate=self.cfg["feedback"]["profile_learning_rate"],
            genre_vector=uv,
            penalised_genres=fb.penalised_genres,
        )

    def raw_profile(self, seed_idx: list[int]) -> np.ndarray:
        return self.raw_z[list(seed_idx)].mean(0)

    def seed_genres(self, seed_idx: list[int]) -> set[str]:
        return {g for i in seed_idx for g in self.tracks.at[i, "genre"]}

    def predicted_genres(self, idx: int, n: int = 3) -> list[tuple[str, float]]:
        p = self.genre_probs[idx]
        top = np.argsort(-p)[:n]
        return [(self.genres[j], float(p[j])) for j in top]

    # ------------------------------------------------------------------ model services
    def encode_inputs(self, X: np.ndarray):
        from .model import encode

        return encode(self.model, X)

    def attribution(self, idx: int, target_embedding: np.ndarray) -> pd.DataFrame:
        from .explainability import group_attribution
        from .model import similarity_attribution

        attr, _ = similarity_attribution(self.model, self.model_inputs[idx], target_embedding)
        return group_attribution(attr, self.pipe.input_names)

    def related_genre_for(self, idx: int, seed_genres: set[str]) -> tuple[str, str, float] | None:
        gm = self.genre_model
        if not gm.available or not seed_genres:
            return None
        best = None
        for g in self.tracks.at[idx, "genre"]:
            if g in seed_genres:
                return None
            for s in seed_genres:
                sim = float(gm.similarity[gm.index(g), gm.index(s)])
                if best is None or sim > best[2]:
                    best = (g, s, sim)
        return best
