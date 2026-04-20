"""
Structural difficulty scoring + bucketing for conditional VAE.

Per chunk, compute three 0–1 signals:
    hazard_rate  = hazards_near_reachable_path / (|reachable| + 1)
    jumps_rate   = jumps_on_shortest_left→right_path / (|reachable| + 1)
    gap_severity = min(max_gap_width, 3) / 3          # 3+ cols ≈ unjumpable

Score = 0.2*jumps_rate + 0.4*hazard_rate + 0.4*gap_severity.
Buckets are rank-based equal-sized tertiles over *reachable* chunks only.
Chunks with no reachable starting position are excluded from training —
they are structurally degenerate and pollute whichever bucket they land in.

Why normalization: the previous scoring used raw counts, which scale with
how much reachable area exists. Content-rich chunks rack up hazards/jumps
(hard); sparse chunks score near zero (easy). The VAE then learned to
spam hazards in the 'easy' bucket. Normalizing by reachable area and
saturating gap width decouples obstacle *intensity* from level *volume*.
"""

from collections import deque

import numpy as np

from data_utils import GRID_HEIGHT, GRID_WIDTH


WALKABLE = (1, 2, 3)
HAZARD = 6
MAX_JUMP_HEIGHT = 2
MAX_JUMP_WIDTH = 2
GAP_SAT = 3  # gaps wider than this are equally "impossible"

WEIGHTS = (0.2, 0.4, 0.4)  # (jumps_rate, hazard_rate, gap_severity)


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
    Returns (jumps_to_right, hazards_near_path, max_gap_width, reachable_count).
    reachable_count = 0 indicates no valid starting position on the left column;
    the chunk should be excluded from conditional training.
    """
    starts = [(r, 0) for r in range(GRID_HEIGHT) if _is_standing(level, r, 0)]
    gap = _max_gap_width(level)
    if not starts:
        return 0, 0, gap, 0

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

    return (right_jumps or 0), len(hazard_cells), gap, len(jumps_at)


def score_levels(levels):
    """Returns (N, 4) raw features: [jumps, hazards, gap, reachable_count]."""
    return np.array([bfs_scores(l) for l in levels], dtype=np.float32)


def assign_buckets(levels, n_buckets=3, weights=WEIGHTS):
    """
    Returns (buckets, valid_mask, info).
      buckets: (N,) int. For unreachable chunks, bucket is 0 (placeholder).
      valid_mask: (N,) bool. True where the chunk should participate in
        training. Callers must filter levels & buckets by this mask.
      info: normalization/weights metadata.
    """
    raw = score_levels(levels)
    jumps = raw[:, 0]
    hazards = raw[:, 1]
    gap = raw[:, 2]
    reach = raw[:, 3]

    inv_reach = 1.0 / (reach + 1.0)
    hazard_rate = hazards * inv_reach
    jumps_rate = jumps * inv_reach
    gap_severity = np.minimum(gap, GAP_SAT) / float(GAP_SAT)

    score = (
        weights[0] * jumps_rate
        + weights[1] * hazard_rate
        + weights[2] * gap_severity
    ).astype(np.float32)

    valid_mask = reach > 0
    buckets = np.zeros(len(levels), dtype=np.int64)

    # Rank-based equal-sized bucketing over valid chunks only.
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) > 0:
        valid_scores = score[valid_idx]
        ranks = np.argsort(np.argsort(valid_scores))
        valid_buckets = np.clip((ranks * n_buckets // len(valid_scores)), 0, n_buckets - 1)
        buckets[valid_idx] = valid_buckets

    info = {
        "weights": list(weights),
        "n_buckets": n_buckets,
        "gap_saturation": GAP_SAT,
        "n_valid": int(valid_mask.sum()),
        "n_dropped": int((~valid_mask).sum()),
    }
    return buckets, valid_mask, info


if __name__ == "__main__":
    from data_utils import load_levels
    levels = load_levels()
    buckets, mask, info = assign_buckets(levels)
    print(f"total: {len(levels)}  valid: {info['n_valid']}  dropped: {info['n_dropped']}")
    print("weights:", info["weights"])
    raw = score_levels(levels)
    for b in range(info["n_buckets"]):
        sel = (buckets == b) & mask
        n = sel.sum()
        j = raw[sel, 0].mean()
        h = raw[sel, 1].mean()
        g = raw[sel, 2].mean()
        r = raw[sel, 3].mean()
        print(f"  bucket {b}: n={n:5d}  mean jumps={j:.2f}  hazards={h:.2f}  gap={g:.2f}  reach={r:.1f}")
