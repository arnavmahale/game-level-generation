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
    if path is None:
        path = get_data_path("levels_categorized.jsonl")
    levels = []
    with open(path) as f:
        for line in f:
            levels.append(json.loads(line))
    return np.array(levels, dtype=np.int64)


def train_test_split_levels(levels, test_fraction=0.15, seed=42):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(levels))
    split = int(len(levels) * (1 - test_fraction))
    return levels[indices[:split]], levels[indices[split:]]
