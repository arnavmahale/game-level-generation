"""
S-expression parser for SuperTux .stl level files.

Extracts the interactive (solid) tilemap from each sector of each level,
returning full-width 2D arrays of raw tile IDs. Downstream code is
responsible for remapping tile IDs to gameplay categories and for
slicing into training chunks.
"""

import os
import json
import numpy as np


# ---------------------------------------------------------------------------
# S-expression tokenizer + parser
# ---------------------------------------------------------------------------

def tokenize(text):
    """Split S-expression text into tokens. Handles nested strings and comments."""
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c == ";":
            while i < n and text[i] != "\n":
                i += 1
        elif c == "(" or c == ")":
            tokens.append(c)
            i += 1
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                if text[j] == "\\":
                    j += 2
                else:
                    j += 1
            tokens.append(text[i:j + 1])
            i = j + 1
        else:
            j = i
            while j < n and not text[j].isspace() and text[j] not in "()":
                j += 1
            tokens.append(text[i:j])
            i = j
    return tokens


def parse_sexpr(tokens, pos=0):
    """Recursive-descent S-expression parser. Returns (node, next_pos)."""
    if tokens[pos] != "(":
        return _atom(tokens[pos]), pos + 1
    pos += 1
    node = []
    while tokens[pos] != ")":
        child, pos = parse_sexpr(tokens, pos)
        node.append(child)
    return node, pos + 1


def _atom(tok):
    """Convert a non-paren token into its typed value."""
    if tok.startswith('"') and tok.endswith('"'):
        return tok[1:-1]
    try:
        return int(tok)
    except ValueError:
        try:
            return float(tok)
        except ValueError:
            return tok


def parse_file(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    tokens = tokenize(text)
    ast, _ = parse_sexpr(tokens)
    return ast


# ---------------------------------------------------------------------------
# Extract tilemaps
# ---------------------------------------------------------------------------

def get_children(node, key):
    """Return all direct children of `node` whose head symbol is `key`."""
    return [c for c in node if isinstance(c, list) and c and c[0] == key]


def get_child_value(node, key, default=None):
    """Return the single value of a (key value) child, or default."""
    for c in node:
        if isinstance(c, list) and len(c) >= 2 and c[0] == key:
            return c[1]
    return default


def decode_rle(tile_tokens):
    """
    Decode SuperTux's run-length encoding.

    Format: a flat sequence where a negative integer N is a run count
    for the *following* value: `-1674 0` -> 1674 zeros. Positive
    integers are single tile IDs. Everything should be ints by the
    time we get here.
    """
    out = []
    i = 0
    n = len(tile_tokens)
    while i < n:
        v = tile_tokens[i]
        if not isinstance(v, int):
            i += 1
            continue
        if v < 0:
            if i + 1 >= n:
                break
            nxt = tile_tokens[i + 1]
            if not isinstance(nxt, int):
                i += 1
                continue
            out.extend([nxt] * (-v))
            i += 2
        else:
            out.append(v)
            i += 1
    return out


def extract_tilemap(tm_node):
    """
    Turn a (tilemap ...) AST node into a dict with decoded 2D array.
    Returns None if the tilemap is malformed (missing fields, bad size).
    """
    width = get_child_value(tm_node, "width")
    height = get_child_value(tm_node, "height")
    z_pos = get_child_value(tm_node, "z-pos", 0)
    solid_raw = get_child_value(tm_node, "solid", "#f")
    solid = solid_raw == "#t"

    tiles_child = None
    for c in tm_node:
        if isinstance(c, list) and c and c[0] == "tiles":
            tiles_child = c[1:]
            break
    if tiles_child is None or width is None or height is None:
        return None

    flat = decode_rle(tiles_child)
    expected = int(width) * int(height)
    if len(flat) < expected:
        flat = flat + [0] * (expected - len(flat))
    elif len(flat) > expected:
        flat = flat[:expected]

    arr = np.array(flat, dtype=np.int64).reshape(int(height), int(width))
    return {
        "width": int(width),
        "height": int(height),
        "z_pos": z_pos,
        "solid": solid,
        "tiles": arr,
    }


def pick_interactive(tilemaps):
    """
    Build the interactive layer for a sector.

    SuperTux sectors commonly have multiple (solid #t) tilemaps with the
    same dimensions — these are layered collision (e.g., ice overlaid on
    stone). They should be *unioned*, not chosen between.

    Strategy:
    1. Keep only solid tilemaps.
    2. Group by (width, height). Cross-dim groups represent different
       regions (e.g., elevator shafts) and aren't compatible.
    3. Pick the dimension group whose union has the most non-zero tiles.
    4. Union within that group: for each cell, first non-zero ID wins.

    Returns None if no solid tilemap exists.
    """
    solid = [t for t in tilemaps if t["solid"]]
    if not solid:
        return None

    groups = {}
    for t in solid:
        key = (t["width"], t["height"])
        groups.setdefault(key, []).append(t)

    def group_content(g):
        union = np.zeros_like(g[0]["tiles"])
        for t in g:
            union = np.where(union == 0, t["tiles"], union)
        return union

    best_dim, best_union, best_count = None, None, -1
    for dim, g in groups.items():
        union = group_content(g)
        count = int((union != 0).sum())
        if count > best_count:
            best_dim, best_union, best_count = dim, union, count

    return {
        "width": best_dim[0],
        "height": best_dim[1],
        "z_pos": 0,
        "solid": True,
        "tiles": best_union,
    }


def parse_level(path):
    """
    Parse a .stl file into a list of interactive-layer arrays, one per sector.

    Returns: list of dicts: [{level_name, sector_name, width, height, tiles}]
    """
    ast = parse_file(path)
    if not (isinstance(ast, list) and ast and ast[0] == "supertux-level"):
        return []

    level_name = get_child_value(ast, "name", os.path.basename(path))
    if isinstance(level_name, list):
        level_name = level_name[-1] if level_name else os.path.basename(path)

    out = []
    for sector in get_children(ast, "sector"):
        sector_name = get_child_value(sector, "name", "unnamed")
        tilemaps = [extract_tilemap(tm) for tm in get_children(sector, "tilemap")]
        tilemaps = [t for t in tilemaps if t is not None]
        interactive = pick_interactive(tilemaps)
        if interactive is None:
            continue
        out.append({
            "level_name": str(level_name),
            "sector_name": str(sector_name),
            "width": interactive["width"],
            "height": interactive["height"],
            "tiles": interactive["tiles"],
        })
    return out


# ---------------------------------------------------------------------------
# Apply tile ID → category mapping
# ---------------------------------------------------------------------------

def load_tile_mapping(mapping_path):
    with open(mapping_path) as f:
        raw = json.load(f)
    return {int(k): int(v) for k, v in raw.items()}


def remap_to_categories(tiles, tile_map, default_category=7):
    """
    Remap a 2D tile-ID array to 8-category array. Tile ID 0 always maps
    to category 0 (empty). Unknown non-zero IDs get default_category
    (7 = decoration in the existing scheme) since the mapping omits
    0 as a key.
    """
    out = np.full_like(tiles, default_category)
    out[tiles == 0] = 0
    for tid in np.unique(tiles):
        if int(tid) == 0:
            continue
        cat = tile_map.get(int(tid), default_category)
        out[tiles == tid] = cat
    return out


# ---------------------------------------------------------------------------
# CLI: parse all levels, report stats
# ---------------------------------------------------------------------------

def walk_stl_files(root):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".stl"):
                yield os.path.join(dirpath, name)


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    levels_root = os.path.join(base, "data", "raw", "stl_levels")
    mapping_path = os.path.join(base, "data", "processed", "tile_id_to_category.json")

    tile_map = load_tile_mapping(mapping_path)

    stl_paths = sorted(walk_stl_files(levels_root))
    print(f"Found {len(stl_paths)} .stl files")

    all_sectors = []
    widths, heights = [], []
    unknown_ids = set()
    total_tiles = 0
    n_files_with_sectors = 0

    for path in stl_paths:
        try:
            sectors = parse_level(path)
        except Exception as e:
            print(f"  FAIL {os.path.relpath(path, levels_root)}: {e}")
            continue
        if sectors:
            n_files_with_sectors += 1
        for s in sectors:
            all_sectors.append(s)
            widths.append(s["width"])
            heights.append(s["height"])
            total_tiles += s["tiles"].size
            for tid in np.unique(s["tiles"]):
                if int(tid) != 0 and int(tid) not in tile_map:
                    unknown_ids.add(int(tid))

    print(f"Files yielding at least one sector: {n_files_with_sectors}/{len(stl_paths)}")
    print(f"Total sectors extracted: {len(all_sectors)}")
    if widths:
        print(f"Width  — min {min(widths)}, median {int(np.median(widths))}, max {max(widths)}, mean {np.mean(widths):.1f}")
        print(f"Height — min {min(heights)}, median {int(np.median(heights))}, max {max(heights)}, mean {np.mean(heights):.1f}")
    print(f"Total tiles: {total_tiles:,}")
    print(f"Unknown tile IDs (not in mapping, excluding 0): {len(unknown_ids)}")
    if unknown_ids:
        sample = sorted(unknown_ids)[:20]
        print(f"  sample: {sample}")


if __name__ == "__main__":
    main()
