"""
Converts raw SuperTux level chunks (tile IDs) into category-mapped grids
and filters out near-empty chunks.

Categories:
  0 = empty, 1 = solid, 2 = slope, 3 = platform,
  4 = bonus, 5 = water, 6 = hazard, 7 = decoration
"""

import ast
import json
import os

GRID_HEIGHT = 20
GRID_WIDTH = 40
MIN_NONEMPTY_FRACTION = 0.10


def load_tile_mapping(path):
    with open(path) as f:
        return {int(k): v for k, v in json.load(f).items()}


def remap_level(grid, tile_map):
    """Map raw tile IDs to gameplay categories."""
    return [
        [0 if t == 0 else tile_map.get(t, 7) for t in row]
        for row in grid
    ]


def nonempty_fraction(grid):
    total = len(grid) * len(grid[0])
    nonempty = sum(1 for row in grid for t in row if t != 0)
    return nonempty / total


def process_levels(raw_path, tile_map_path, output_path):
    tile_map = load_tile_mapping(tile_map_path)

    kept = 0
    filtered = 0

    with open(raw_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            grid = ast.literal_eval(line)

            if len(grid) != GRID_HEIGHT or len(grid[0]) != GRID_WIDTH:
                filtered += 1
                continue

            remapped = remap_level(grid, tile_map)

            if nonempty_fraction(remapped) < MIN_NONEMPTY_FRACTION:
                filtered += 1
                continue

            f_out.write(json.dumps(remapped) + "\n")
            kept += 1

    return kept, filtered


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, "data", "raw", "all_levels.txt")
    tile_map_path = os.path.join(base_dir, "data", "processed", "tile_id_to_category.json")
    output_path = os.path.join(base_dir, "data", "processed", "levels_categorized.jsonl")

    kept, filtered = process_levels(raw_path, tile_map_path, output_path)

    print(f"Kept: {kept}")
    print(f"Filtered: {filtered}")
    print(f"Output: {output_path}")
