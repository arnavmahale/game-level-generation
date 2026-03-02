"""
Parses SuperTux tiles.strf to build a mapping from tile IDs to gameplay categories.

Categories:
  0 = empty (air)
  1 = solid ground
  2 = slope (solid + slope)
  3 = unisolid platform (one-way)
  4 = coin / bonus block
  5 = water
  6 = hazard (spikes, lava, fire)
  7 = decoration (non-solid, non-interactive)
"""

import re
import json
import os

# Attribute bitfield flags from src/supertux/tile.hpp
SOLID = 0x0001
UNISOLID = 0x0002
SLOPE = 0x0010
COIN = 0x0040
ICE = 0x0100
WATER = 0x0200
HURTS = 0x0400
FIRE = 0x0800

CATEGORY_NAMES = {
    0: "empty",
    1: "solid",
    2: "slope",
    3: "platform",
    4: "bonus",
    5: "water",
    6: "hazard",
    7: "decoration",
}


def attr_to_category(attr):
    """Convert a numeric attribute bitfield to a gameplay category."""
    if attr & HURTS or attr & FIRE:
        return 6
    if attr & WATER:
        return 5
    if attr & COIN:
        return 4
    if attr & SLOPE:
        return 2
    if attr & UNISOLID:
        return 3
    if attr & SOLID:
        return 1
    return 7  # non-solid, non-interactive = decoration


def parse_tiles_strf(filepath):
    """Parse tiles.strf and return a dict mapping tile_id -> category."""
    with open(filepath, "r") as f:
        content = f.read()

    tile_map = {}

    # Parse bulk (tiles ...) blocks WITH attributes
    bulk_with_attrs = re.compile(
        r"\(tiles\s*\n"
        r"\s*\(width (\d+)\)\s*\(height (\d+)\)\s*\n"
        r"\s*\(ids\s*([\d\s]+?)\)\s*\n"
        r"\s*\(attributes\s*([\d\s]+?)\)",
        re.DOTALL,
    )
    for m in bulk_with_attrs.finditer(content):
        ids = list(map(int, m.group(3).split()))
        attrs = list(map(int, m.group(4).split()))
        for tile_id, attr in zip(ids, attrs):
            if tile_id == 0:
                continue
            tile_map[tile_id] = attr_to_category(attr)

    # Parse bulk (tiles ...) blocks WITHOUT attributes (decoration)
    bulk_no_attrs = re.compile(
        r"\(tiles\s*\n"
        r"\s*(?:;[^\n]*\n\s*)?"
        r"\(width (\d+)\)\s*\(height (\d+)\)\s*\n"
        r"\s*\(ids\s*([\d\s]+?)\)\s*\n"
        r"\s*\(images",
        re.DOTALL,
    )
    for m in bulk_no_attrs.finditer(content):
        ids = list(map(int, m.group(3).split()))
        for tile_id in ids:
            if tile_id == 0 or tile_id in tile_map:
                continue
            tile_map[tile_id] = 7  # no attributes = decoration

    # Parse individual (tile ...) blocks with boolean properties
    single_pattern = re.compile(r"\(tile\s*\n(.*?)\n\s*\)", re.DOTALL)

    for m in single_pattern.finditer(content):
        block = m.group(1)

        id_match = re.search(r"\(id (\d+)\)", block)
        if not id_match:
            continue
        tile_id = int(id_match.group(1))

        has_water = bool(re.search(r"\(water #t\)", block))
        has_hurts = bool(re.search(r"\(hurts #t\)", block))
        has_fire = bool(re.search(r"\(fire #t\)", block))
        has_solid = bool(re.search(r"\(solid #t\)", block))
        has_unisolid = bool(re.search(r"\(unisolid #t\)", block))
        has_coin = bool(re.search(r'\(object-name "coin"\)', block))
        has_bonus = bool(re.search(r'\(object-name "bonusblock"\)', block))
        has_brick = bool(re.search(r'\(object-name "brick"\)', block))

        if has_hurts or has_fire:
            tile_map[tile_id] = 6
        elif has_water:
            tile_map[tile_id] = 5
        elif has_coin or has_bonus or has_brick:
            tile_map[tile_id] = 4
        elif has_unisolid:
            tile_map[tile_id] = 3
        elif has_solid:
            tile_map[tile_id] = 1
        elif tile_id not in tile_map:
            tile_map[tile_id] = 7  # decoration

    return tile_map


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    strf_path = os.path.join(base_dir, "data", "raw", "tiles.strf")
    output_path = os.path.join(base_dir, "data", "processed", "tile_id_to_category.json")

    tile_map = parse_tiles_strf(strf_path)

    with open(output_path, "w") as f:
        json.dump(tile_map, f, indent=2)

    # Print summary
    from collections import Counter
    counts = Counter(tile_map.values())
    print(f"Mapped {len(tile_map)} tile IDs to categories:")
    for cat_id in sorted(counts):
        print(f"  {cat_id} ({CATEGORY_NAMES[cat_id]}): {counts[cat_id]} tiles")
