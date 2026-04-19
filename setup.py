"""
End-to-end project setup: build features and train all models.

Usage:
    python setup.py                  # default: build features + train all
    python setup.py --features-only  # rebuild data/processed/ from data/raw/
    python setup.py --models-only    # retrain all models from existing features

This script orchestrates the pipeline; each step is also runnable as a
standalone module from scripts/. Raw data (SuperTux .stl files) must be
present in data/raw/stl_levels/ before running the feature build.
"""

import argparse
import os
import sys
import subprocess


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(REPO_ROOT, "scripts")


def run(cmd, cwd=None):
    print(f"\n>>> {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=cwd or SCRIPTS, check=False)
    if res.returncode != 0:
        sys.exit(f"FAILED: {' '.join(cmd)} (exit {res.returncode})")


def build_features():
    raw_dir = os.path.join(REPO_ROOT, "data", "raw", "stl_levels")
    if not os.path.isdir(raw_dir) or not os.listdir(raw_dir):
        sys.exit(
            f"Missing raw data at {raw_dir}.\n"
            "Sparse-clone the SuperTux levels: \n"
            "  git clone --depth=1 --filter=blob:none --sparse "
            "https://github.com/SuperTux/supertux.git /tmp/supertux\n"
            "  cd /tmp/supertux && git sparse-checkout set data/levels\n"
            "Then copy the .stl files into data/raw/stl_levels/."
        )
    run([sys.executable, "build_tile_mapping.py"])
    run([sys.executable, "build_features.py"])


def train_models():
    run([sys.executable, "model_naive.py"])
    run([sys.executable, "model_classical.py"])
    run([sys.executable, "model_deep.py"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument("--models-only", action="store_true")
    args = parser.parse_args()

    if not args.models_only:
        build_features()
    if not args.features_only:
        train_models()
    print("\nDone. Run `python app.py` to launch the web UI.")


if __name__ == "__main__":
    main()
