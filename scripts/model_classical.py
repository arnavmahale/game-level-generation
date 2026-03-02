"""
Classical ML model: bigram transition model with positional priors.

Generates levels column-by-column. Each tile is sampled based on:
- Its vertical neighbor (tile above)
- Its horizontal neighbor (tile to the left)
- A positional prior (row-based)

Transition probabilities are learned from training data.
"""

import numpy as np
import pickle
from data_utils import (
    load_levels, train_test_split_levels, get_model_path,
    NUM_CATEGORIES, GRID_HEIGHT, GRID_WIDTH,
)


class BigramModel:
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.vertical_trans = None   # P(tile | tile_above)
        self.horizontal_trans = None  # P(tile | tile_left)
        self.row_priors = None       # P(tile | row_index)
        self.first_col_probs = None  # P(tile | row) for the first column

    def fit(self, levels):
        N = NUM_CATEGORIES
        alpha = self.smoothing

        # Vertical transitions: P(tile[r,c] | tile[r-1,c])
        self.vertical_trans = np.full((N, N), alpha)
        for level in levels:
            for r in range(1, GRID_HEIGHT):
                for c in range(GRID_WIDTH):
                    self.vertical_trans[level[r - 1, c], level[r, c]] += 1
        self.vertical_trans /= self.vertical_trans.sum(axis=1, keepdims=True)

        # Horizontal transitions: P(tile[r,c] | tile[r,c-1])
        self.horizontal_trans = np.full((N, N), alpha)
        for level in levels:
            for r in range(GRID_HEIGHT):
                for c in range(1, GRID_WIDTH):
                    self.horizontal_trans[level[r, c - 1], level[r, c]] += 1
        self.horizontal_trans /= self.horizontal_trans.sum(axis=1, keepdims=True)

        # Row priors: P(tile | row)
        self.row_priors = np.full((GRID_HEIGHT, N), alpha)
        for level in levels:
            for r in range(GRID_HEIGHT):
                for c in range(GRID_WIDTH):
                    self.row_priors[r, level[r, c]] += 1
        self.row_priors /= self.row_priors.sum(axis=1, keepdims=True)

        # First column distribution
        self.first_col_probs = np.full((GRID_HEIGHT, N), alpha)
        for level in levels:
            for r in range(GRID_HEIGHT):
                self.first_col_probs[r, level[r, 0]] += 1
        self.first_col_probs /= self.first_col_probs.sum(axis=1, keepdims=True)

    def generate(self, n=1, seed=None, prior_weight=0.5, trans_weight=0.25):
        rng = np.random.RandomState(seed)
        results = np.zeros((n, GRID_HEIGHT, GRID_WIDTH), dtype=np.int64)

        for idx in range(n):
            grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int64)

            # First column
            for r in range(GRID_HEIGHT):
                if r == 0:
                    probs = self.first_col_probs[r]
                else:
                    vert = self.vertical_trans[grid[r - 1, 0]]
                    prior = self.row_priors[r]
                    probs = prior_weight * prior + (1 - prior_weight) * vert
                    probs /= probs.sum()
                grid[r, 0] = rng.choice(NUM_CATEGORIES, p=probs)

            # Remaining columns: weighted average of prior, vertical, horizontal
            for c in range(1, GRID_WIDTH):
                for r in range(GRID_HEIGHT):
                    horiz = self.horizontal_trans[grid[r, c - 1]]
                    prior = self.row_priors[r]
                    if r > 0:
                        vert = self.vertical_trans[grid[r - 1, c]]
                        probs = (prior_weight * prior
                                 + trans_weight * vert
                                 + trans_weight * horiz)
                    else:
                        probs = prior_weight * prior + (1 - prior_weight) * horiz
                    probs /= probs.sum()
                    grid[r, c] = rng.choice(NUM_CATEGORIES, p=probs)

            results[idx] = grid

        return results

    def save(self, path=None):
        if path is None:
            path = get_model_path("bigram_model.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "vertical_trans": self.vertical_trans,
                "horizontal_trans": self.horizontal_trans,
                "row_priors": self.row_priors,
                "first_col_probs": self.first_col_probs,
            }, f)

    def load(self, path=None):
        if path is None:
            path = get_model_path("bigram_model.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vertical_trans = data["vertical_trans"]
        self.horizontal_trans = data["horizontal_trans"]
        self.row_priors = data["row_priors"]
        self.first_col_probs = data["first_col_probs"]


if __name__ == "__main__":
    levels = load_levels()
    train, test = train_test_split_levels(levels)

    model = BigramModel()
    model.fit(train)
    model.save()

    sample = model.generate(n=1, seed=42)[0]
    print(f"Trained on {len(train)} levels")
    print(f"Sample category counts: {dict(zip(*np.unique(sample, return_counts=True)))}")
