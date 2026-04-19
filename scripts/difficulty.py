"""
Structural difficulty scoring + bucketing for conditional VAE.

Score per chunk =
    0.2 * norm(BFS_jump_count_to_right_edge)
  + 0.4 * norm(hazards_on_reachable_path)
  + 0.4 * norm(max_contiguous_empty_column_run)

Buckets are tertiles of the weighted score across the training set.

Inputs are 8-category levels (as stored on disk). Scoring treats the
three gameplay categories that matter — walkable (1/2/3), hazard (6),
and everything else as air — which matches the in-browser physics.
"""

from collections import deque

import numpy as np

from data_utils import GRID_HEIGHT, GRID_WIDTH


WALKABLE = (1, 2, 3)
HAZARD = 6
MAX_JUMP_HEIGHT = 2
MAX_JUMP_WIDTH = 2

WEIGHTS = (0.2, 0.4, 0.4)  # (jumps, hazards, gap)


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


def _max_gap_width(level):
    has_walkable = np.isin(level, WALKABLE).any(axis=0)
    max_run = cur = 0
    for v in has_walkable:
        if not v:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run


def bfs_scores(level):
    """
    Returns (jumps_to_right, hazards_near_path, max_gap_width).

    jumps_to_right: number of jump-type edges on shortest left→right path.
      0 when the level is unreachable (so unreachable chunks score low on
      jumps but still score on gap/hazard, which usually drives them hard).
    hazards_near_path: count of HAZARD tiles within 1 Chebyshev-step of any
      reachable standing position.
    max_gap_width: longest run of consecutive columns with no walkable tile.
    """
    starts = [(r, 0) for r in range(GRID_HEIGHT) if _is_standing(level, r, 0)]
    gap = _max_gap_width(level)
    if not starts:
        return 0, 0, gap

    jumps_at = {s: 0 for s in starts}
    q = deque(starts)
    right_jumps = None

    while q:
        r, c = q.popleft()
        if c == GRID_WIDTH - 1 and right_jumps is None:
            right_jumps = jumps_at[(r, c)]

        for dc in (-1, 1):
            nc = c + dc
            if 0 <= nc < GRID_WIDTH and _is_standing(level, r, nc) and (r, nc) not in jumps_at:
                jumps_at[(r, nc)] = jumps_at[(r, c)]
                q.append((r, nc))

        for dh in range(1, MAX_JUMP_HEIGHT + 1):
            for dc in range(-MAX_JUMP_WIDTH, MAX_JUMP_WIDTH + 1):
                nr, nc = r - dh, c + dc
                if 0 <= nr < GRID_HEIGHT and 0 <= nc < GRID_WIDTH:
                    if _is_standing(level, nr, nc) and (nr, nc) not in jumps_at:
                        jumps_at[(nr, nc)] = jumps_at[(r, c)] + 1
                        q.append((nr, nc))

        for dc in (-1, 0, 1):
            nc = c + dc
            if not 0 <= nc < GRID_WIDTH:
                continue
            for dr in range(1, GRID_HEIGHT):
                nr = r + dr
                if nr >= GRID_HEIGHT:
                    break
                if int(level[nr, nc]) in WALKABLE:
                    break
                if _is_standing(level, nr, nc) and (nr, nc) not in jumps_at:
                    jumps_at[(nr, nc)] = jumps_at[(r, c)]
                    q.append((nr, nc))
                    break

    hazard_cells = set()
    for r, c in jumps_at:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID_HEIGHT and 0 <= nc < GRID_WIDTH:
                    if int(level[nr, nc]) == HAZARD:
                        hazard_cells.add((nr, nc))

    return (right_jumps or 0), len(hazard_cells), gap


def score_levels(levels):
    return np.array([bfs_scores(l) for l in levels], dtype=np.float32)


def assign_buckets(levels, n_buckets=3, weights=WEIGHTS):
    """
    Returns (buckets, info). `buckets` is (N,) int in [0, n_buckets-1].
    `info` carries the normalization stats so we can apply them at inference.
    """
    raw = score_levels(levels)
    lo = raw.min(axis=0)
    hi = raw.max(axis=0)
    rng = np.maximum(hi - lo, 1e-6)
    norm = (raw - lo) / rng
    score = (norm * np.array(weights, dtype=np.float32)).sum(axis=1)
    # Rank-based equal-sized buckets — robust to ties at 0 (many chunks have
    # no hazards and small gaps, so a quantile split on raw score collapses).
    ranks = np.argsort(np.argsort(score))
    buckets = np.clip((ranks * n_buckets // len(score)), 0, n_buckets - 1)
    info = {
        "lo": lo.tolist(),
        "hi": hi.tolist(),
        "weights": list(weights),
        "n_buckets": n_buckets,
    }
    return buckets.astype(np.int64), info


if __name__ == "__main__":
    from data_utils import load_levels
    levels = load_levels()
    buckets, info = assign_buckets(levels)
    print("bucket counts:", np.bincount(buckets).tolist())
    print("weights:", info["weights"])
    raw = score_levels(levels)
    for b in range(info["n_buckets"]):
        mask = buckets == b
        print(f"  bucket {b}: n={mask.sum():5d} mean(jumps,hazards,gap)="
              f"({raw[mask, 0].mean():.1f}, {raw[mask, 1].mean():.1f}, {raw[mask, 2].mean():.1f})")
