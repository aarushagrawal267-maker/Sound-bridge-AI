"""User feedback: consented local storage, profile updates and a personal preference layer.

Storage: JSON lines files under data/feedback/ (one event per line). Nothing is written
unless the user has ticked the consent box; without consent, feedback still adapts the
current session but is kept in memory only.

The system learns only from the feedback actually recorded on this machine. It does not
learn from a population of users, and the dashboard reports the real sample size.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import path

FEEDBACK_TYPES: dict[str, str] = {
    "like": "Like",
    "dislike": "Dislike",
    "known_artist": "Already know this artist",
    "not_my_genre": "Not my genre",
    "more_like_this": "Want more like this",
    "more_variety": "Want more variety",
}
POSITIVE = {"like", "more_like_this"}
NEGATIVE = {"dislike", "not_my_genre"}


@dataclass
class FeedbackEvent:
    session_id: str
    track_idx: int
    track_id: str
    feedback: str
    strategy: str = ""
    rank: int = -1
    relevance: float = float("nan")
    popularity: int = -1
    genre: str = ""
    artist: str = ""
    weights: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class JsonlStore:
    """Append-only record store: a Google Sheet tab when configured (see src/storage.py),
    otherwise a local JSON-lines file. Passing `directory` always forces local files."""

    def __init__(self, filename: str, directory: Path | None = None):
        self.path = (directory or path("feedback_dir")) / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sheet = None
        if directory is None:
            from .storage import make_table, sheets_enabled

            if sheets_enabled():
                self.sheet = make_table(Path(filename).stem)

    @property
    def backend(self) -> str:
        return "Google Sheet" if self.sheet is not None else f"local file ({self.path.name})"

    def append(self, record: dict) -> None:
        if self.sheet is not None:
            self.sheet.append(record)
            return
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def load(self) -> pd.DataFrame:
        if self.sheet is not None:
            return self.sheet.load()
        if not self.path.exists():
            return pd.DataFrame()
        rows = []
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return pd.DataFrame(rows)

    def clear(self) -> None:
        if self.sheet is not None:
            self.sheet.clear()
            return
        if self.path.exists():
            self.path.unlink()


class FeedbackStore(JsonlStore):
    def __init__(self, directory: Path | None = None):
        super().__init__("feedback.jsonl", directory)

    def record(self, event: FeedbackEvent) -> None:
        if event.feedback not in FEEDBACK_TYPES:
            raise ValueError(f"Unknown feedback type {event.feedback!r}")
        self.append(asdict(event))


class StudyStore(JsonlStore):
    """Responses from the structured user-testing workflow."""

    def __init__(self, directory: Path | None = None):
        super().__init__("study_responses.jsonl", directory)


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------- profile updates
@dataclass
class SessionFeedback:
    liked: list[int] = field(default_factory=list)
    disliked: list[int] = field(default_factory=list)
    penalised_genres: list[str] = field(default_factory=list)
    known_artists: list[str] = field(default_factory=list)
    variety_requests: int = 0
    more_like_this: int = 0
    events: list[dict] = field(default_factory=list)


def apply_event(state: SessionFeedback, event: FeedbackEvent, primary_genre: str | None = None) -> SessionFeedback:
    """Update the in-session preference state from one feedback event."""
    i = event.track_idx
    fb = event.feedback
    if fb in POSITIVE and i not in state.liked:
        state.liked.append(i)
        if i in state.disliked:
            state.disliked.remove(i)
    if fb == "more_like_this":
        state.more_like_this += 1
    if fb in NEGATIVE and i not in state.disliked:
        state.disliked.append(i)
        if i in state.liked:
            state.liked.remove(i)
    if fb == "not_my_genre" and primary_genre and primary_genre not in state.penalised_genres:
        state.penalised_genres.append(primary_genre)
    if fb == "known_artist" and event.artist and event.artist not in state.known_artists:
        state.known_artists.append(event.artist)
    if fb == "more_variety":
        state.variety_requests += 1
    state.events.append(asdict(event))
    return state


def adjust_weights(weights, state: SessionFeedback):
    """Nudge ranking weights from explicit feedback (bounded, transparent rules).

    * each "already know this artist"  -> novelty +0.04 (max +0.2)
    * each "want more variety"         -> diversity +0.06 (max +0.3), artist cap 1
    * each "want more like this"       -> relevance +0.04 (max +0.2)
    """
    from copy import copy

    w = copy(weights)
    w.novelty = round(w.novelty + min(0.04 * len(state.known_artists), 0.2), 3)
    w.diversity = round(w.diversity + min(0.06 * state.variety_requests, 0.3), 3)
    w.relevance = round(w.relevance + min(0.04 * state.more_like_this, 0.2), 3)
    if state.variety_requests:
        w.max_per_artist = 1
    return w


class PersonalPreferenceModel:
    """A small linear preference layer trained on the user's own like/dislike labels,
    using SoundBridge Encoder embeddings as features. Only fitted once enough labels of
    both classes exist."""

    def __init__(self, min_labels: int = 20):
        self.min_labels = min_labels
        self.model = None
        self.n_labels = 0

    def fit(self, embeddings: np.ndarray, liked: list[int], disliked: list[int]) -> bool:
        from sklearn.linear_model import LogisticRegression

        self.n_labels = len(liked) + len(disliked)
        if self.n_labels < self.min_labels or not liked or not disliked:
            self.model = None
            return False
        X = embeddings[list(liked) + list(disliked)]
        y = np.r_[np.ones(len(liked)), np.zeros(len(disliked))]
        self.model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500).fit(X, y)
        return True

    @property
    def ready(self) -> bool:
        return self.model is not None

    def predict(self, embeddings: np.ndarray, idx) -> dict[int, float]:
        if self.model is None:
            return {}
        p = self.model.predict_proba(embeddings[list(idx)])[:, 1]
        return {int(i): float(v) for i, v in zip(idx, p)}


def feedback_summary(df: pd.DataFrame) -> dict[str, object]:
    """Aggregate statistics for the Feedback Dashboard from real stored events."""
    if df.empty:
        return {"n_events": 0}
    out: dict[str, object] = {
        "n_events": int(len(df)),
        "n_sessions": int(df["session_id"].nunique()),
        "counts": df["feedback"].value_counts().to_dict(),
    }
    rated = df[df["feedback"].isin(POSITIVE | NEGATIVE)]
    if len(rated):
        out["like_rate"] = float(rated["feedback"].isin(POSITIVE).mean())
        by = rated.assign(pos=rated["feedback"].isin(POSITIVE)).groupby("strategy")["pos"].agg(["mean", "count"])
        out["like_rate_by_strategy"] = by.reset_index().rename(columns={"mean": "like_rate", "count": "n"})
    return out
