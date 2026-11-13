"""Download the public Spotify Tracks Dataset (checksum-verified).

Usage:  python scripts/download_data.py [--force]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import download_dataset  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    print("Dataset ready at", download_dataset(force=ap.parse_args().force))
