"""One command to make SoundBridge AI runnable from a fresh clone.

Usage:  python scripts/setup_all.py [--experiments] [--figures]

1. download the public dataset (checksum-verified)
2. validate, clean and feature-engineer it            (scripts/build_artifacts.py)
3. train + evaluate the SoundBridge Encoder on CPU     (scripts/train_model.py, ~2-5 min)
4. optionally re-run the offline experiments           (scripts/run_experiments.py, ~3-6 min)
5. optionally regenerate report figures                (scripts/make_figures.py)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(script: str, *args: str) -> None:
    t = time.time()
    print(f"\n>>> {script} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], check=True, cwd=ROOT)
    print(f"<<< {script} done in {time.time() - t:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", action="store_true", help="also re-run the offline experiments")
    ap.add_argument("--figures", action="store_true", help="also regenerate the report figures")
    args = ap.parse_args()
    run("build_artifacts.py")
    run("train_model.py")
    if args.experiments:
        run("run_experiments.py")
    if args.figures:
        run("make_figures.py")
    print("\nSoundBridge AI is ready. Launch it with:  streamlit run app.py")


if __name__ == "__main__":
    main()
