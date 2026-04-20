"""
Pre-compute the train-split tile-category distribution so the deployed app
can score JS divergence without loading the 150MB training JSONL at runtime.

Output: models/train_tile_dist.npy — shape (NUM_CATEGORIES,), normalized.
"""

import numpy as np

from data_utils import (
    load_levels, train_val_test_split_levels, get_model_path, NUM_CATEGORIES,
)


def main():
    levels = load_levels()
    train, _, _ = train_val_test_split_levels(levels)
    counts = np.bincount(train.flatten(), minlength=NUM_CATEGORIES).astype(float)
    dist = counts / counts.sum()
    out_path = get_model_path("train_tile_dist.npy")
    np.save(out_path, dist)
    print(f"wrote {out_path}  shape={dist.shape}  dist={dist.round(4).tolist()}")


if __name__ == "__main__":
    main()
