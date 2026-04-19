"""
Sliding-window feature pipeline.

For each .stl file in data/raw/stl_levels/, extract all interactive-layer
sectors (via parse_stl), remap tile IDs to 8 gameplay categories, then
slide a GRID_HEIGHT x GRID_WIDTH window across the 2D sector in both
directions. Chunks are filtered to the playable band using an empty-
fraction range — this rejects all-sky chunks (too sparse) and all-
underground chunks (too dense) without needing to heuristically locate
the "floor" per sector (which is unreliable in caves and vertical levels).

Output: data/processed/levels_categorized.jsonl — one JSON 2D array per line.

Categories:
  0 = empty, 1 = solid, 2 = slope, 3 = platform,
  4 = bonus, 5 = water, 6 = hazard, 7 = decoration
"""

import json
import os
from collections import Counter

import numpy as np

from parse_stl import (
    walk_stl_files, parse_level, load_tile_mapping, remap_to_categories,
)

GRID_HEIGHT = 20
GRID_WIDTH = 40
STRIDE_X = 4
STRIDE_Y = 4
# Keep chunks that look like playable bands: some air, some ground.
EMPTY_FRAC_MIN = 0.20
EMPTY_FRAC_MAX = 0.80
# Also require meaningful ground presence so we don't keep all-decoration.
MIN_SOLID_FRAC = 0.05
SOLID_CATEGORIES = (1, 2, 3)
EMPTY_CATEGORY = 0
DECORATION_CATEGORY = 7


def sliding_windows_2d(grid, h=GRID_HEIGHT, w=GRID_WIDTH,
                       stride_y=STRIDE_Y, stride_x=STRIDE_X):
    """
    Emit every (h, w) window from `grid`, sliding in both dimensions.

    If the grid is smaller than (h, w) in either dimension, pad with
    EMPTY_CATEGORY on the short side so exactly one chunk is emitted.
    """
    H, W = grid.shape
    if H < h or W < w:
        pad_y = max(0, h - H)
        pad_x = max(0, w - W)
        padded = np.full((H + pad_y, W + pad_x), EMPTY_CATEGORY, dtype=grid.dtype)
        padded[pad_y:, :W] = grid  # anchor to bottom-left
        yield padded[:h, :w]
        return
    for y in range(0, H - h + 1, stride_y):
        for x in range(0, W - w + 1, stride_x):
            yield grid[y:y + h, x:x + w]


def is_playable_band(chunk):
    """
    Accept chunks with both air and ground in reasonable proportions.
    Rejects all-sky (too much empty), all-underground (too little empty),
    and all-decoration (no solid).
    """
    empty_frac = float((chunk == EMPTY_CATEGORY).sum()) / chunk.size
    if not (EMPTY_FRAC_MIN <= empty_frac <= EMPTY_FRAC_MAX):
        return False
    solid_frac = float(np.isin(chunk, SOLID_CATEGORIES).sum()) / chunk.size
    if solid_frac < MIN_SOLID_FRAC:
        return False
    return True


def process():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    levels_root = os.path.join(base, "data", "raw", "stl_levels")
    mapping_path = os.path.join(base, "data", "processed", "tile_id_to_category.json")
    output_path = os.path.join(base, "data", "processed", "levels_categorized.jsonl")

    tile_map = load_tile_mapping(mapping_path)

    n_sectors = 0
    n_candidate = 0
    kept = 0
    rejected_too_empty = 0
    rejected_too_solid = 0
    rejected_no_solid = 0
    category_counts = Counter()

    with open(output_path, "w") as f_out:
        for stl_path in sorted(walk_stl_files(levels_root)):
            for sector in parse_level(stl_path):
                n_sectors += 1
                categorized = remap_to_categories(
                    sector["tiles"], tile_map, default_category=DECORATION_CATEGORY
                )
                for chunk in sliding_windows_2d(categorized):
                    n_candidate += 1
                    empty_frac = float((chunk == EMPTY_CATEGORY).sum()) / chunk.size
                    solid_frac = float(np.isin(chunk, SOLID_CATEGORIES).sum()) / chunk.size
                    if empty_frac > EMPTY_FRAC_MAX:
                        rejected_too_empty += 1
                        continue
                    if empty_frac < EMPTY_FRAC_MIN:
                        rejected_too_solid += 1
                        continue
                    if solid_frac < MIN_SOLID_FRAC:
                        rejected_no_solid += 1
                        continue
                    f_out.write(json.dumps(chunk.tolist()) + "\n")
                    kept += 1
                    category_counts.update(chunk.flatten().tolist())

    total_tiles = sum(category_counts.values())
    print(f"Sectors processed: {n_sectors}")
    print(f"Candidate chunks (before filter): {n_candidate:,}")
    print(f"  rejected — too empty (>{int(EMPTY_FRAC_MAX*100)}% air): {rejected_too_empty:,}")
    print(f"  rejected — too solid (<{int(EMPTY_FRAC_MIN*100)}% air): {rejected_too_solid:,}")
    print(f"  rejected — too little solid terrain: {rejected_no_solid:,}")
    print(f"Chunks kept: {kept:,}")
    print(f"Output: {output_path}")
    print("Category distribution in kept chunks:")
    for cat in sorted(category_counts):
        pct = 100.0 * category_counts[cat] / total_tiles
        print(f"  {cat}: {category_counts[cat]:>9,}  ({pct:5.2f}%)")


if __name__ == "__main__":
    process()
