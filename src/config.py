"""Central configuration: paths, feature definitions and settings from config.yaml."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"

# The nine perceptual audio descriptors used for Musical DNA and explanations.
DNA_FEATURES = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
    "tempo",
    "loudness",
]

# Features bounded to [0, 1] by Spotify's audio-analysis definitions.
UNIT_FEATURES = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
]

# Extra structural inputs the encoder may use (not shown as "DNA").
STRUCTURAL_FEATURES = ["duration_ms", "mode", "key", "time_signature", "explicit"]

FEATURE_LABELS = {
    "danceability": "Danceability",
    "energy": "Energy",
    "valence": "Valence (musical positiveness)",
    "acousticness": "Acousticness",
    "instrumentalness": "Instrumentalness",
    "speechiness": "Speechiness",
    "liveness": "Liveness",
    "tempo": "Tempo (BPM)",
    "loudness": "Loudness (dB)",
    "duration_ms": "Duration",
    "mode": "Mode (major/minor)",
    "key": "Key",
    "time_signature": "Time signature",
    "explicit": "Explicit lyrics flag",
}

SHORT_LABELS = {
    "danceability": "Dance",
    "energy": "Energy",
    "valence": "Valence",
    "acousticness": "Acoustic",
    "instrumentalness": "Instrumental",
    "speechiness": "Speech",
    "liveness": "Live",
    "tempo": "Tempo",
    "loudness": "Loudness",
}

# Display ranges used to put tempo and loudness on a 0-1 scale for charts.
DISPLAY_RANGES = {"tempo": (50.0, 200.0), "loudness": (-30.0, 0.0)}

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    with open(path or CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def path(key: str) -> Path:
    """Resolve a configured path relative to the project root."""
    p = ROOT / load_config()["paths"][key]
    if p.suffix == "":
        p.mkdir(parents=True, exist_ok=True)
    return p


PROCESSED_TRACKS = "tracks.parquet"
ARTIFACT_FILES = {
    "tracks": "tracks.parquet",
    "embeddings": "embeddings.npy",
    "raw_matrix": "raw_features.npy",
    "genre_probs": "genre_probs.npy",
    "pred_popularity": "pred_popularity.npy",
    "coords": "coords_2d.npy",
    "clusters": "clusters.npy",
}
