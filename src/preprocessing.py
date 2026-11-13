"""Cleaning and de-duplication of the raw track table.

Steps
1. Exclude rows whose core audio features are missing or invalid (no imputation of
   audio characteristics; the excluded rows are counted in the cleaning log).
2. Merge rows that share a track_id (the source lists one track once per genre):
   the genres are kept as an ordered list.
3. Merge different releases of the same song (same normalised title and artists),
   keeping the most popular release and the union of genres.
4. Derive helper columns (primary artist, artist list, popularity tier).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import DNA_FEATURES
from .data_validation import invalid_row_mask


@dataclass
class CleaningLog:
    steps: list[tuple[str, int, int]] = field(default_factory=list)

    def add(self, name: str, before: int, after: int) -> None:
        self.steps.append((name, before, after))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps, columns=["step", "rows_before", "rows_after"]).assign(
            removed=lambda d: d.rows_before - d.rows_after
        )


_TITLE_NOISE = re.compile(
    r"\s*[-(\[].*\b(remaster\w*|version|edit|mono|stereo|live|remix|mix)\b.*$", re.I
)


def normalise_title(title: str) -> str:
    t = str(title).lower().strip()
    t = _TITLE_NOISE.sub("", t)
    return re.sub(r"\s+", " ", t)


_BRACKETS = re.compile(r"\s*[(\[].*?[)\]]")


def base_title(title: str) -> str:
    """Title without bracketed parts or ' - ' suffixes ("Song (feat. X) - Live" -> "song")."""
    t = _BRACKETS.sub("", str(title).lower())
    t = t.split(" - ")[0]
    return re.sub(r"\s+", " ", t).strip()


def split_artists(value: str) -> list[str]:
    return [a.strip() for a in str(value).split(";") if a.strip()]


def popularity_tier(pop: pd.Series) -> pd.Series:
    bins = [-1, 20, 35, 50, 70, 100]
    labels = ["very low", "low", "moderate", "high", "very high"]
    return pd.cut(pop, bins=bins, labels=labels).astype(str)


def clean_tracks(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLog]:
    log = CleaningLog()
    features = [f for f in DNA_FEATURES if f in df.columns]
    n = len(df)
    df = df.loc[~invalid_row_mask(df, features)].copy()
    log.add("exclude rows with missing/invalid core values", n, len(df))

    has_genre = "genre" in df.columns
    if "track_id" in df.columns:
        n = len(df)
        agg: dict[str, object] = {c: "first" for c in df.columns if c not in ("track_id", "genre")}
        if "popularity" in df.columns:
            agg["popularity"] = "max"
        if has_genre:
            agg["genre"] = lambda s: list(dict.fromkeys(s.dropna().astype(str)))
        df = df.groupby("track_id", sort=False, as_index=False).agg(agg)
        log.add("merge rows sharing a track_id (genres -> list)", n, len(df))
    elif has_genre:
        df["genre"] = df["genre"].apply(lambda g: [str(g)] if pd.notna(g) else [])

    # merge different releases of the same song by the same artists
    n = len(df)
    df["_song_key"] = df["track_name"].map(normalise_title) + "|" + df["artists"].astype(str).str.lower()
    sort_col = "popularity" if "popularity" in df.columns else "track_name"
    df = df.sort_values(sort_col, ascending=False, kind="stable")
    if has_genre:
        merged_genres = df.groupby("_song_key")["genre"].agg(
            lambda s: list(dict.fromkeys(g for lst in s for g in lst))
        )
    versions = df.groupby("_song_key").size()
    df = df.drop_duplicates("_song_key", keep="first").copy()
    if has_genre:
        df["genre"] = df["_song_key"].map(merged_genres)
    df["n_versions"] = df["_song_key"].map(versions).astype(int)
    log.add("merge alternative releases of the same song", n, len(df))

    df["artist_list"] = df["artists"].map(split_artists)
    df["primary_artist"] = df["artist_list"].str[0]
    df["n_artists"] = df["artist_list"].str.len()
    if has_genre:
        df["primary_genre"] = df["genre"].str[0].fillna("unknown")
        df["genre_str"] = df["genre"].map(lambda g: ", ".join(g))
    if "popularity" in df.columns:
        df["popularity"] = df["popularity"].astype(int)
        df["popularity_tier"] = popularity_tier(df["popularity"])
    if "album_name" in df.columns:
        df["album_name"] = df["album_name"].fillna("Unknown album")
    for c in ("key", "mode", "time_signature", "explicit"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop(columns=["_song_key"]).reset_index(drop=True)
    df.insert(0, "idx", np.arange(len(df)))
    return df, log
