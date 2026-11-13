"""Step 1: load, validate, clean and feature-engineer the raw dataset.

Usage:  python scripts/build_artifacts.py
Outputs: data/processed/tracks.parquet, models/feature_pipeline.joblib,
         outputs/reports/data_validation.json, outputs/reports/cleaning_log.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DNA_FEATURES, load_config, path  # noqa: E402
from src.data_loader import download_dataset, load_tracks, schema_report  # noqa: E402
from src.data_validation import validate  # noqa: E402
from src.feature_engineering import FeaturePipeline  # noqa: E402
from src.preprocessing import clean_tracks  # noqa: E402


def main() -> None:
    cfg = load_config()
    raw_path = download_dataset()
    raw = load_tracks(raw_path)
    report = validate(raw, DNA_FEATURES)
    tracks, log = clean_tracks(raw)
    pipe = FeaturePipeline.fit(tracks, seed=cfg["seed"])

    processed = path("processed_dir")
    tracks.to_parquet(processed / "tracks.parquet", index=False)
    joblib.dump(pipe, path("models_dir") / "feature_pipeline.joblib")

    reports = path("outputs_dir") / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    log.to_frame().to_csv(reports / "cleaning_log.csv", index=False)
    summary = {
        "raw_rows": report.n_rows,
        "schema": schema_report(raw),
        "missing": {k: v for k, v in report.missing.items() if v},
        "out_of_range": {k: v for k, v in report.out_of_range.items() if v},
        "duplicate_track_id_rows": report.duplicate_track_ids,
        "duplicate_title_artist_releases": report.duplicate_name_artist,
        "invalid_rows": report.invalid_row_count,
        "notes": report.notes,
        "clean_tracks": len(tracks),
        "clean_primary_artists": int(tracks["primary_artist"].nunique()),
        "genres": int(len({g for lst in tracks["genre"] for g in lst})) if "genre" in tracks else 0,
        "model_inputs": pipe.input_names,
    }
    (reports / "data_validation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(log.to_frame().to_string(index=False))


if __name__ == "__main__":
    main()
