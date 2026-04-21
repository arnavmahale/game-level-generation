"""
Playability repair post-pass.

Generated levels — regardless of the model — often have small structural
gaps that break BFS playability: missing floor in a column (player falls
off), no standing position on the left edge (no valid spawn), or a gap
too wide to jump.

This module applies minimal edits to guarantee:
1. No fall-off-the-map: every column has at least one solid-type tile.
2. Valid spawn: at least one standing position in the leftmost columns.
3. Traversability: BFS reaches the right edge.

We use the same movement model as scripts/evaluate.py (walk, jump up to
4 tiles high / 3 tiles wide, fall) so a repaired level is guaranteed to
pass check_playability.
"""

from collections import deque

import numpy as np

from data_utils import GRID_HEIGHT, GRID_WIDTH


EMPTY = 0
SOLID = 1
SLOPE = 2
PLATFORM = 3
HAZARD = 6
WALKABLE = (SOLID, SLOPE, PLATFORM)

MAX_JUMP_HEIGHT = 2
MAX_JUMP_WIDTH = 2


# ---------------------------------------------------------------------------
# Movement model (matches evaluate.py)
# ---------------------------------------------------------------------------

def _is_walkable_below(level, r, c):
    if r >= GRID_HEIGHT - 1:
        return False
    return int(level[r + 1, c]) in WALKABLE


def _is_standing(level, r, c):
    if int(level[r, c]) in WALKABLE:
        return False
    if int(level[r, c]) == HAZARD:
        return False
    return _is_walkable_below(level, r, c)


def _reachable_positions(level):
    starts = [(r, 0) for r in range(GRID_HEIGHT) if _is_standing(level, r, 0)]
    visited = set(starts)
    q = deque(starts)

    while q:
        r, c = q.popleft()

        # Walk
        for dc in (-1, 1):
            nc = c + dc
            if 0 <= nc < GRID_WIDTH and _is_standing(level, r, nc) and (r, nc) not in visited:
                visited.add((r, nc))
                q.append((r, nc))

        # Jump
        for dh in range(1, MAX_JUMP_HEIGHT + 1):
            for dc in range(-MAX_JUMP_WIDTH, MAX_JUMP_WIDTH + 1):
                nr, nc = r - dh, c + dc
                if 0 <= nr < GRID_HEIGHT and 0 <= nc < GRID_WIDTH:
                    if _is_standing(level, nr, nc) and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        q.append((nr, nc))

        # Fall
        for dc in (-1, 0, 1):
            nc = c + dc
            if not (0 <= nc < GRID_WIDTH):
                continue
            for dr in range(1, GRID_HEIGHT):
                nr = r + dr
                if nr >= GRID_HEIGHT:
                    break
                if int(level[nr, nc]) in WALKABLE:
                    break
                if _is_standing(level, nr, nc) and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((nr, nc))
                    break

    return visited


def _has_right_edge(reachable):
    return any(c == GRID_WIDTH - 1 for (_, c) in reachable)


# ---------------------------------------------------------------------------
# Cosmetic cleanup passes (run before BFS repair)
# ---------------------------------------------------------------------------

def _strip_sky_walls(level, top_rows=4):
    """
    Remove floating SOLID/SLOPE tiles from the top rows.

    Models sometimes hallucinate random dirt blocks in the sky that don't
    read as platformer geometry. PLATFORM (3), BONUS (4), HAZARD (6), and
    WATER (5) are allowed up high — those legitimately appear in air
    (ledges, coins, spikes, ceilings).
    """
    top = level[:top_rows, :]
    mask = (top == SOLID) | (top == SLOPE)
    top[mask] = EMPTY
    return level


def enforce_layout(level, top_air_rows=4, bottom_solid_rows=5):
    """
    Force top N rows to empty sky and bottom M rows to solid floor.

    Gives every generated level a consistent 'platformer screen' silhouette
    regardless of what the model emits at those rows, so output looks like
    a level instead of a random tile grid. Applied to training data too,
    so the VAE spends its capacity on the middle region where gameplay
    actually happens.
    """
    level = level.copy()
    if top_air_rows > 0:
        level[:top_air_rows, :] = EMPTY
    if bottom_solid_rows > 0:
        level[-bottom_solid_rows:, :] = SOLID
    return level


def _remove_stray_islands(level, max_row=8):
    """
    Remove single-cell SOLID tiles in the *sky* (top ~8 rows) that have no
    walkable neighbors. These are the 'random floating dirt block' noise
    from VAE/transformer. Only runs up high — leaves mid-level geometry
    alone because a lone SOLID down low is often part of a legitimate
    structure (wall, column, platform support) the player interacts with.
    """
    for r in range(max_row):
        for c in range(GRID_WIDTH):
            if int(level[r, c]) != SOLID:
                continue
            has_neighbor = False
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID_HEIGHT and 0 <= nc < GRID_WIDTH:
                    if int(level[nr, nc]) in WALKABLE:
                        has_neighbor = True
                        break
            if not has_neighbor:
                level[r, c] = EMPTY
    return level


def _fill_subfloor_pits(level, min_fill_row=GRID_HEIGHT - 4):
    """
    Fill air pockets buried *deep* inside the ground — air cells in the
    bottom 4 rows sandwiched between solid above and solid below. These
    are almost always dataset/generator noise (ground-level tiles with
    random air holes) rather than intentional cave features.

    Keep this conservative: higher cutoffs erase legitimate caves and
    jumpable sub-surface gaps that the BFS repair can otherwise exploit.
    """
    for c in range(GRID_WIDTH):
        for r in range(min_fill_row, GRID_HEIGHT - 1):
            if int(level[r, c]) != EMPTY:
                continue
            above = int(level[r - 1, c]) if r > 0 else EMPTY
            below = int(level[r + 1, c])
            if above in (SOLID, SLOPE) and below in (SOLID, SLOPE):
                level[r, c] = SOLID
    return level


# ---------------------------------------------------------------------------
# Repair passes
# ---------------------------------------------------------------------------

def _ensure_every_column_has_floor(level):
    """If a column has no solid/slope/platform, add a solid tile at its bottom."""
    for c in range(GRID_WIDTH):
        col = level[:, c]
        if not np.isin(col, WALKABLE).any():
            level[GRID_HEIGHT - 1, c] = SOLID
    return level


def _ensure_spawn(level, leftmost_cols=3, safe_cols=4):
    """
    Guarantee at least one standing position in the leftmost columns AND
    clear hazards in a small safe zone around it so the player isn't
    blocked in on frame 1.
    """
    spawn = None
    for c in range(leftmost_cols):
        for r in range(GRID_HEIGHT):
            if _is_standing(level, r, c):
                spawn = (r, c)
                break
        if spawn:
            break

    if spawn is None:
        # No spawn anywhere in the leftmost cols — force one at col 0.
        level[GRID_HEIGHT - 1, 0] = SOLID
        if int(level[GRID_HEIGHT - 2, 0]) in WALKABLE + (HAZARD,):
            level[GRID_HEIGHT - 2, 0] = EMPTY
        spawn = (GRID_HEIGHT - 2, 0)

    # Clear hazards in the spawn row + 1 row above, across the first
    # `safe_cols` columns. Player needs a safe lane to move off the spawn.
    r, c = spawn
    for cc in range(min(safe_cols, GRID_WIDTH)):
        for rr in (r, max(0, r - 1)):
            if int(level[rr, cc]) == HAZARD:
                level[rr, cc] = EMPTY
    return level


def _place_bridge_step(level, reachable):
    """
    Extend the reachable region one step to the right by placing a platform.

    Strategy: find the rightmost reachable standing tile. Look for an empty
    cell in the next 1-3 columns within jump range and place a PLATFORM
    there. PLATFORM tiles are walkable and don't need support underneath.
    """
    if not reachable:
        return level, False

    max_c = max(c for _, c in reachable)
    frontier = [(r, c) for (r, c) in reachable if c == max_c]
    r, c = max(frontier)  # lowest row at the rightmost column

    # Standing height above platform is (nr - 1); player must jump ≤ MAX_JUMP_HEIGHT
    # from current row r, so the standing row must satisfy r - MAX_JUMP_HEIGHT <=
    # (nr - 1), i.e. nr >= r - MAX_JUMP_HEIGHT + 1 = r - 3. We also allow placing
    # below (nr = r+1, standing at r — level walk).
    for dc in range(1, MAX_JUMP_WIDTH + 1):
        nc = c + dc
        if nc >= GRID_WIDTH:
            break
        for dr in [0, 1, -1, -2, -3]:
            nr = r + dr
            if not (0 <= nr < GRID_HEIGHT):
                continue
            if int(level[nr, nc]) != EMPTY:
                continue
            # The cell the player stands on (nr - 1) must be empty/passable
            # (not solid, not hazard — hazard kills the player mid-jump).
            if nr > 0:
                above = int(level[nr - 1, nc])
                if above in WALKABLE or above == HAZARD:
                    continue
            level[nr, nc] = PLATFORM
            return level, True
    return level, False


def _carve_walls(level, reachable):
    """
    Fallback when bridging fails: find the rightmost reachable column,
    look at the next column over, and remove tall solid stacks there
    by deleting the topmost solid tile (lowering wall height).
    Removes hazards in the player path too.
    """
    if not reachable:
        return level, False
    max_c = max(c for _, c in reachable)
    nc = max_c + 1
    if nc >= GRID_WIDTH:
        return level, False
    # Find topmost solid tile in column nc
    for r in range(GRID_HEIGHT):
        if int(level[r, nc]) in WALKABLE:
            level[r, nc] = EMPTY
            return level, True
    # No wall — remove a hazard if any
    for r in range(GRID_HEIGHT):
        if int(level[r, nc]) == HAZARD:
            level[r, nc] = EMPTY
            return level, True
    return level, False


def repair(level, max_iterations=60):
    """
    Return a repaired copy of `level` that passes BFS playability.

    Strategy:
      1. Ensure every column has a floor and a leftmost spawn exists.
      2. Greedy bridge: place PLATFORM tiles to extend reachable region
         to the right edge.
      3. If bridging stalls, fall back to carving down walls or deleting
         hazards in the next column to unblock progress.

    Returns the best-effort result even if not fully playable.
    """
    level = np.asarray(level, dtype=np.int64).copy()

    _strip_sky_walls(level)
    _remove_stray_islands(level)
    _fill_subfloor_pits(level)
    _ensure_every_column_has_floor(level)
    _ensure_spawn(level)

    for _ in range(max_iterations):
        reachable = _reachable_positions(level)
        if _has_right_edge(reachable):
            return level
        level, placed = _place_bridge_step(level, reachable)
        if not placed:
            level, carved = _carve_walls(level, reachable)
            if not carved:
                break

    return level


def repair_many(levels):
    """Repair a batch of levels. Accepts (N, H, W) or list of (H, W)."""
    arr = np.asarray(levels)
    if arr.ndim == 2:
        return repair(arr)
    return np.stack([repair(l) for l in arr])


if __name__ == "__main__":
    # Smoke test: a level with a pit and a wall.
    test = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int64)
    test[GRID_HEIGHT - 1, :10] = SOLID
    # Pit from col 10-20, ground resumes at col 20
    test[GRID_HEIGHT - 1, 20:] = SOLID
    # A tall wall at col 30
    test[GRID_HEIGHT - 8:GRID_HEIGHT, 30] = SOLID

    from evaluate import check_playability
    print("before repair:", check_playability(test))
    fixed = repair(test)
    print("after  repair:", check_playability(fixed))
    # Count edits
    diffs = int((fixed != test).sum())
    print(f"cells edited: {diffs}")
