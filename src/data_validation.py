"""Data validation: ranges, duplicates, missing values and invalid records.

Validation never "repairs" a value by inventing it. It reports problems and returns
boolean masks that preprocessing uses to exclude invalid rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import UNIT_FEATURES

VALID_RANGES: dict[str, tuple[float, float]] = {
    **{f: (0.0, 1.0) for f in UNIT_FEATURES},
    "popularity": (0, 100),
    "tempo": (1e-9, 300.0),        # 0 BPM is an analysis failure, not a tempo
    "loudness": (-60.0, 5.0),
    "key": (-1, 11),
    "mode": (0, 1),
    "time_signature": (1, 7),       # 0 is an analysis failure
    "duration_ms": (1, 6 * 60 * 60 * 1000),  # positive and under 6 hours
}


@dataclass
class ValidationReport:
    n_rows: int
    missing: dict[str, int]
    out_of_range: dict[str, int]
    duplicate_track_ids: int
    duplicate_name_artist: int
    missing_artist: int
    non_finite: dict[str, int]
    invalid_row_count: int
    notes: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for col in sorted(set(self.missing) | set(self.out_of_range) | set(self.non_finite)):
            rows.append(
                {
                    "column": col,
                    "missing": self.missing.get(col, 0),
                    "out_of_range": self.out_of_range.get(col, 0),
                    "non_finite": self.non_finite.get(col, 0),
                }
            )
        return pd.DataFrame(rows)


def range_violations(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame: True where a present value lies outside its valid range."""
    out = pd.DataFrame(index=df.index)
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            out[col] = v.notna() & ((v < lo) | (v > hi))
    return out


def invalid_row_mask(df: pd.DataFrame, required_features: list[str]) -> pd.Series:
    """Rows that cannot be used for similarity: missing/invalid core values."""
    mask = pd.Series(False, index=df.index)
    for col in ["track_name", "artists"]:
        if col in df.columns:
            mask |= df[col].isna() | (df[col].astype(str).str.strip() == "")
    present = [f for f in required_features if f in df.columns]
    if present:
        vals = df[present].apply(pd.to_numeric, errors="coerce")
        mask |= vals.isna().any(axis=1)
        mask |= ~np.isfinite(vals.to_numpy(dtype=float)).all(axis=1)
    mask |= range_violations(df).any(axis=1)
    return mask


def validate(df: pd.DataFrame, required_features: list[str]) -> ValidationReport:
    numeric = [c for c in VALID_RANGES if c in df.columns]
    non_finite = {}
    for c in numeric:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        non_finite[c] = int((~np.isfinite(v) & ~np.isnan(v)).sum())
    rv = range_violations(df)
    notes = []
    if "tempo" in df.columns:
        n0 = int((pd.to_numeric(df["tempo"], errors="coerce") == 0).sum())
        if n0:
            notes.append(f"{n0} rows report tempo = 0 BPM (audio analysis failure); excluded.")
    if "time_signature" in df.columns:
        n0 = int((pd.to_numeric(df["time_signature"], errors="coerce") == 0).sum())
        if n0:
            notes.append(f"{n0} rows report time signature 0 (analysis failure); excluded.")
    if "popularity" in df.columns:
        n0 = int((df["popularity"] == 0).sum())
        notes.append(
            f"{n0} rows have popularity 0. Spotify popularity is relative and recency-weighted; "
            "0 often marks a duplicate release whose plays are credited to another version."
        )
    dup_ids = int(df["track_id"].duplicated().sum()) if "track_id" in df.columns else 0
    key = df["track_name"].astype(str).str.lower().str.strip() + "|" + df["artists"].astype(str).str.lower()
    if "track_id" in df.columns:
        dup_na = int(key[~df["track_id"].duplicated()].duplicated().sum())
    else:
        dup_na = int(key.duplicated().sum())
    return ValidationReport(
        n_rows=len(df),
        missing={c: int(df[c].isna().sum()) for c in df.columns},
        out_of_range={c: int(rv[c].sum()) for c in rv.columns},
        duplicate_track_ids=dup_ids,
        duplicate_name_artist=dup_na,
        missing_artist=int(df["artists"].isna().sum()),
        non_finite=non_finite,
        invalid_row_count=int(invalid_row_mask(df, required_features).sum()),
        notes=notes,
    )
