"""Flexible loading of Spotify-style track datasets.

Different public Spotify datasets name their columns differently (``artists`` vs
``artist_name``, ``track_genre`` vs ``playlist_genre`` ...). This module detects the
available columns, maps them to a canonical schema and never invents missing values:
a feature that is absent in the source stays absent (the column is not created).
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd
import requests

from .config import ROOT, load_config

log = logging.getLogger(__name__)

# canonical name -> accepted source aliases (lower-case)
COLUMN_ALIASES: dict[str, list[str]] = {
    "track_id": ["track_id", "id", "spotify_id", "uri", "track_uri"],
    "track_name": ["track_name", "name", "song", "title", "song_name"],
    "artists": ["artists", "artist", "artist_name", "artist_names", "track_artist"],
    "artist_id": ["artist_id", "artist_ids", "id_artists"],
    "album_name": ["album_name", "album", "track_album_name"],
    "album_id": ["album_id", "track_album_id"],
    "genre": ["track_genre", "genre", "playlist_genre", "genres"],
    "subgenre": ["playlist_subgenre", "subgenre"],
    "popularity": ["popularity", "track_popularity", "song_popularity"],
    "duration_ms": ["duration_ms", "duration", "length"],
    "explicit": ["explicit"],
    "danceability": ["danceability"],
    "energy": ["energy"],
    "key": ["key"],
    "loudness": ["loudness"],
    "mode": ["mode"],
    "speechiness": ["speechiness"],
    "acousticness": ["acousticness"],
    "instrumentalness": ["instrumentalness"],
    "liveness": ["liveness"],
    "valence": ["valence"],
    "tempo": ["tempo"],
    "time_signature": ["time_signature"],
    "release_date": ["release_date", "track_album_release_date", "year"],
    "image_url": ["image_url", "album_image", "cover_url", "album_cover"],
}

REQUIRED_COLUMNS = ["track_name", "artists"]


def detect_columns(columns: list[str]) -> dict[str, str]:
    """Return a mapping {source_column: canonical_column} for recognised columns."""
    lower = {c.lower().strip(): c for c in columns}
    mapping: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower and lower[alias] not in mapping:
                mapping[lower[alias]] = canonical
                break
    return mapping


def schema_report(df: pd.DataFrame) -> dict[str, bool]:
    """Which canonical fields are available in an already-mapped frame."""
    return {c: (c in df.columns) for c in COLUMN_ALIASES}


def load_tracks(path: str | Path | None = None) -> pd.DataFrame:
    """Load a raw CSV and map it to the canonical schema."""
    cfg = load_config()
    path = Path(path) if path else ROOT / cfg["paths"]["raw_data"]
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python scripts/download_data.py` "
            "or place a Spotify tracks CSV there."
        )
    df = pd.read_csv(path, low_memory=False)
    # drop pandas index artefacts such as "Unnamed: 0"
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    mapping = detect_columns(list(df.columns))
    df = df[list(mapping)].rename(columns=mapping)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset lacks required columns: {missing}")
    if "explicit" in df.columns:
        df["explicit"] = (
            df["explicit"].astype(str).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0})
        )
    log.info("Loaded %d rows with columns %s", len(df), list(df.columns))
    return df


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_dataset(force: bool = False) -> Path:
    """Download the public dataset from a verified mirror (checksum enforced)."""
    cfg = load_config()
    dest = ROOT / cfg["paths"]["raw_data"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = cfg["dataset"]["sha256"]
    if dest.exists() and not force and file_sha256(dest) == expected:
        return dest
    last_error: Exception | None = None
    for url in cfg["dataset"]["urls"]:
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(resp.content)
            if file_sha256(tmp) != expected:
                tmp.unlink()
                raise ValueError(f"Checksum mismatch for {url}")
            tmp.replace(dest)
            return dest
        except Exception as exc:  # try the next mirror
            last_error = exc
            log.warning("Mirror failed: %s (%s)", url, exc)
    raise RuntimeError(
        "Could not download the dataset from any mirror. Download it manually from "
        f"{cfg['dataset']['kaggle_page']} and save it as {dest}"
    ) from last_error
