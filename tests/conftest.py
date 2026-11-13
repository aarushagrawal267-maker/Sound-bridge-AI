import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _artifacts_ready() -> bool:
    return (ROOT / "data/processed/tracks.parquet").exists() and (ROOT / "models/soundbridge_encoder.pt").exists()


@pytest.fixture(scope="session")
def engine():
    if not _artifacts_ready():
        pytest.skip("model artifacts not built (run scripts/setup_all.py)")
    from src.engine import SoundBridgeEngine

    return SoundBridgeEngine()


@pytest.fixture(scope="session")
def seeds(engine):
    """Three real tracks from one genre, by different artists, chosen deterministically."""
    t = engine.tracks
    rock = t[(t["primary_genre"] == "alt-rock") & (t["popularity"] >= 50)].drop_duplicates("primary_artist")
    return rock["idx"].head(3).astype(int).tolist()


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """A tiny hand-written frame in the source schema, used to unit-test cleaning logic."""
    base = dict(danceability=0.5, energy=0.6, key=1, loudness=-7.0, mode=1, speechiness=0.05, acousticness=0.2,
                instrumentalness=0.0, liveness=0.1, valence=0.4, tempo=120.0, time_signature=4, duration_ms=200000,
                explicit=False, popularity=40)
    rows = [
        dict(base, track_id="A", artists="Band X", album_name="Al", track_name="Song One", track_genre="rock"),
        dict(base, track_id="A", artists="Band X", album_name="Al", track_name="Song One", track_genre="indie"),
        dict(base, track_id="B", artists="Band X", album_name="Live", track_name="Song One - Live", track_genre="rock", popularity=10),
        dict(base, track_id="C", artists="Singer Y;Band X", album_name="Al2", track_name="Other", track_genre="pop", energy=0.9),
        dict(base, track_id="D", artists="Z", album_name="Bad", track_name="Broken", track_genre="pop", tempo=0.0),
        dict(base, track_id="E", artists=None, album_name="None", track_name="No artist", track_genre="pop"),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
