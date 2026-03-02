"""
Naive baseline: random ground fill.

Learns the average ground height from training data, then generates
levels with solid ground below that height and air above. A small
random height variation is added per column.
"""

import numpy as np
import pickle
from data_utils import (
    load_levels, train_test_split_levels, get_model_path,
    NUM_CATEGORIES, GRID_HEIGHT, GRID_WIDTH,
)

EMPTY = 0
SOLID = 1


class NaiveBaseline:
    def __init__(self):
        self.mean_ground_height = None
        self.height_std = None

    def fit(self, levels):
        """Learn average ground height from training data."""
        ground_heights = []
        for level in levels:
            for col in range(GRID_WIDTH):
                column = level[:, col]
                solid_rows = np.where(column == SOLID)[0]
                if len(solid_rows) > 0:
                    ground_heights.append(GRID_HEIGHT - solid_rows.min())
                else:
                    ground_heights.append(0)
        self.mean_ground_height = np.mean(ground_heights)
        self.height_std = np.std(ground_heights)

    def generate(self, n=1, seed=None):
        rng = np.random.RandomState(seed)
        levels = np.zeros((n, GRID_HEIGHT, GRID_WIDTH), dtype=np.int64)

        for idx in range(n):
            for col in range(GRID_WIDTH):
                h = int(self.mean_ground_height + rng.normal(0, self.height_std))
                h = max(1, min(h, GRID_HEIGHT))
                levels[idx, GRID_HEIGHT - h:, col] = SOLID

        return levels

    def save(self, path=None):
        if path is None:
            path = get_model_path("naive_baseline.pkl")
        with open(path, "wb") as f:
            pickle.dump({"mean": self.mean_ground_height, "std": self.height_std}, f)

    def load(self, path=None):
        if path is None:
            path = get_model_path("naive_baseline.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.mean_ground_height = data["mean"]
        self.height_std = data["std"]


if __name__ == "__main__":
    levels = load_levels()
    train, test = train_test_split_levels(levels)

    model = NaiveBaseline()
    model.fit(train)
    model.save()

    sample = model.generate(n=1, seed=42)[0]
    print(f"Trained on {len(train)} levels")
    print(f"Mean ground height: {model.mean_ground_height:.1f} +/- {model.height_std:.1f}")
    print(f"Sample category counts: {dict(zip(*np.unique(sample, return_counts=True)))}")
