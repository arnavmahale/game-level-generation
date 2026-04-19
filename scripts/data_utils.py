"""Shared data loading utilities for all models."""

import json
import numpy as np
import os

NUM_CATEGORIES = 8
GRID_HEIGHT = 20
GRID_WIDTH = 40

CATEGORY_NAMES = {
    0: "empty", 1: "solid", 2: "slope", 3: "platform",
    4: "bonus", 5: "water", 6: "hazard", 7: "decoration",
}


def get_data_path(filename):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "data", "processed", filename)


def get_model_path(filename):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "models", filename)


def load_levels(path=None):
    """
    Load training levels. Defaults to the platformer-filtered dataset
    (chunks that look like classic side-scroller screens, with decoration
    collapsed to empty). Pass an explicit path to use the raw dataset.
    """
    if path is None:
        path = get_data_path("levels_platformer.jsonl")
    levels = []
    with open(path) as f:
        for line in f:
            levels.append(json.loads(line))
    return np.array(levels, dtype=np.int64)


def simplify_for_gameplay(levels):
    """
    Collapse visual-only categories that the in-browser game treats as air.

    The GameCanvas physics treat solid/slope/platform as walls, hazard as
    death tiles, and everything else (empty, bonus, water, decoration)
    as air. So decoration (7) is pure visual noise from a gameplay POV.

    We map: 7 -> 0 (decoration becomes empty). Bonus and water stay as
    distinct categories because they're visually meaningful — but you can
    extend this if you want even cleaner outputs.
    """
    levels = np.asarray(levels).copy()
    levels[levels == 7] = 0
    return levels


def filter_platformer_chunks(levels,
                             bottom_rows=5, top_rows=10,
                             min_bottom_solid=0.5, max_top_solid=0.3):
    """
    Keep only chunks that look like classic side-scroller screens:
    - bottom `bottom_rows` rows are >= `min_bottom_solid` walkable
    - top `top_rows` rows are <= `max_top_solid` walkable

    This filters out cave interiors, all-sky chunks, and decoration-heavy
    fragments that don't read as "platformer level" to a player.
    """
    walkable = np.isin(levels, (1, 2, 3))
    bottom = walkable[:, -bottom_rows:, :].mean(axis=(1, 2))
    top = walkable[:, :top_rows, :].mean(axis=(1, 2))
    keep = (bottom >= min_bottom_solid) & (top <= max_top_solid)
    return levels[keep]


def train_test_split_levels(levels, test_fraction=0.15, seed=42):
    """Two-way split (kept for any callers that only need train vs held-out)."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(levels))
    split = int(len(levels) * (1 - test_fraction))
    return levels[indices[:split]], levels[indices[split:]]


def train_val_test_split_levels(levels, val_fraction=0.10, test_fraction=0.10, seed=42):
    """
    Three-way split: train / val / test (defaults 80/10/10).

    val is used during training for best-checkpoint selection; test is
    held out and only touched at final evaluation.
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(levels))
    n = len(levels)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    n_train = n - n_val - n_test
    tr = indices[:n_train]
    va = indices[n_train:n_train + n_val]
    te = indices[n_train + n_val:]
    return levels[tr], levels[va], levels[te]
