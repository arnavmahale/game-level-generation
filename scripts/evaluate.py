"""
Evaluation metrics for generated levels.

Metrics:
1. Tile Distribution Similarity (JS divergence vs real levels)
2. Playability (A* pathfinding - can a player traverse left to right?)
3. Structural Metrics (ground coverage, contiguity, platform count)
4. Diversity (pairwise hamming distance between generated levels)
"""

import numpy as np
from collections import Counter
from data_utils import NUM_CATEGORIES, GRID_HEIGHT, GRID_WIDTH

EMPTY = 0
SOLID = 1
SLOPE = 2
PLATFORM = 3
HAZARD = 6
WALKABLE = {SOLID, SLOPE, PLATFORM}


# --- 1. Tile Distribution Similarity ---

def tile_distribution(levels):
    """Get normalized category frequency distribution."""
    counts = np.bincount(levels.flatten(), minlength=NUM_CATEGORIES).astype(float)
    return counts / counts.sum()


def js_divergence(p, q):
    """Jensen-Shannon divergence between two distributions."""
    m = 0.5 * (p + q)
    # Avoid log(0)
    p_safe = np.where(p > 0, p, 1e-10)
    q_safe = np.where(q > 0, q, 1e-10)
    m_safe = np.where(m > 0, m, 1e-10)
    kl_pm = np.sum(p_safe * np.log(p_safe / m_safe))
    kl_qm = np.sum(q_safe * np.log(q_safe / m_safe))
    return 0.5 * (kl_pm + kl_qm)


def distribution_similarity(real_levels, generated_levels):
    """JS divergence between real and generated tile distributions. Lower = better."""
    p = tile_distribution(real_levels)
    q = tile_distribution(generated_levels)
    return js_divergence(p, q)


# --- 2. Playability (A* pathfinding) ---

def is_walkable(level, row, col):
    """Check if a position has ground support (tile below is walkable)."""
    if row >= GRID_HEIGHT - 1:
        return False
    return level[row + 1, col] in WALKABLE


def is_standing_position(level, row, col):
    """Player can stand here: current cell is not solid, cell below is walkable."""
    if level[row, col] in WALKABLE:
        return False
    if level[row, col] == HAZARD:
        return False
    return is_walkable(level, row, col)


def check_playability(level, max_jump_height=4, max_jump_width=3):
    """
    BFS-based check: can a player reach from leftmost column to rightmost?
    Player can walk, jump (up to max_jump_height tiles up, max_jump_width across),
    and fall.
    """
    # Find all valid starting positions (leftmost column)
    start_positions = set()
    for r in range(GRID_HEIGHT):
        if is_standing_position(level, r, 0):
            start_positions.add((r, 0))

    if not start_positions:
        return False

    visited = set()
    queue = list(start_positions)
    visited.update(start_positions)

    while queue:
        r, c = queue.pop(0)

        if c == GRID_WIDTH - 1:
            return True

        # Generate possible moves
        moves = []

        # Walk left/right
        for dc in [-1, 1]:
            nc = c + dc
            if 0 <= nc < GRID_WIDTH and is_standing_position(level, r, nc):
                moves.append((r, nc))

        # Jump: can go up to max_jump_height up, and max_jump_width horizontally
        for dh in range(1, max_jump_height + 1):
            for dc in range(-max_jump_width, max_jump_width + 1):
                nr, nc = r - dh, c + dc
                if 0 <= nr < GRID_HEIGHT and 0 <= nc < GRID_WIDTH:
                    if is_standing_position(level, nr, nc):
                        moves.append((nr, nc))

        # Fall: drop down from current position
        for dc in [-1, 0, 1]:
            nc = c + dc
            if 0 <= nc < GRID_WIDTH:
                for dr in range(1, GRID_HEIGHT):
                    nr = r + dr
                    if nr >= GRID_HEIGHT:
                        break
                    if is_standing_position(level, nr, nc):
                        moves.append((nr, nc))
                        break
                    if level[nr, nc] in WALKABLE:
                        break

        for pos in moves:
            if pos not in visited:
                visited.add(pos)
                queue.append(pos)

    return False


def playability_rate(levels):
    """Fraction of levels that are playable. Higher = better."""
    playable = sum(check_playability(level) for level in levels)
    return playable / len(levels)


# --- 3. Structural Metrics ---

def ground_coverage(level):
    """Fraction of tiles that are walkable (solid/slope/platform)."""
    walkable_count = sum(1 for r in range(GRID_HEIGHT) for c in range(GRID_WIDTH)
                         if level[r, c] in WALKABLE)
    return walkable_count / (GRID_HEIGHT * GRID_WIDTH)


def ground_contiguity(level):
    """Fraction of ground tiles that are adjacent to another ground tile."""
    ground_tiles = 0
    adjacent = 0
    for r in range(GRID_HEIGHT):
        for c in range(GRID_WIDTH):
            if level[r, c] in WALKABLE:
                ground_tiles += 1
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < GRID_HEIGHT and 0 <= nc < GRID_WIDTH:
                        if level[nr, nc] in WALKABLE:
                            adjacent += 1
                            break
    return adjacent / max(ground_tiles, 1)


def hazard_density(level):
    """Fraction of tiles that are hazards."""
    return np.sum(level == HAZARD) / (GRID_HEIGHT * GRID_WIDTH)


def structural_metrics(levels):
    """Compute average structural metrics across levels."""
    coverages = [ground_coverage(l) for l in levels]
    contiguities = [ground_contiguity(l) for l in levels]
    hazards = [hazard_density(l) for l in levels]
    return {
        "ground_coverage": np.mean(coverages),
        "ground_contiguity": np.mean(contiguities),
        "hazard_density": np.mean(hazards),
    }


# --- 4. Diversity ---

def pairwise_diversity(levels, n_samples=100):
    """Average pairwise hamming distance between generated levels. Higher = more diverse."""
    n = min(len(levels), n_samples)
    indices = np.random.RandomState(42).choice(len(levels), n, replace=False)
    sampled = levels[indices].reshape(n, -1)
    total_dist = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_dist += np.mean(sampled[i] != sampled[j])
            count += 1
    return total_dist / max(count, 1)


# --- Full Evaluation ---

def evaluate_model(name, generated_levels, real_levels):
    """Run all metrics on a set of generated levels."""
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")

    js = distribution_similarity(real_levels, generated_levels)
    print(f"  JS Divergence (lower=better):  {js:.4f}")

    play = playability_rate(generated_levels)
    print(f"  Playability Rate:              {play:.2%}")

    struct = structural_metrics(generated_levels)
    print(f"  Ground Coverage:               {struct['ground_coverage']:.2%}")
    print(f"  Ground Contiguity:             {struct['ground_contiguity']:.2%}")
    print(f"  Hazard Density:                {struct['hazard_density']:.4f}")

    div = pairwise_diversity(generated_levels)
    print(f"  Diversity (higher=more varied): {div:.4f}")

    return {
        "js_divergence": js,
        "playability": play,
        **struct,
        "diversity": div,
    }


if __name__ == "__main__":
    import torch
    from data_utils import load_levels, train_test_split_levels, get_model_path
    from model_naive import NaiveBaseline
    from model_classical import BigramModel
    from model_deep import ConvVAE, generate_levels

    levels = load_levels()
    train, test = train_test_split_levels(levels)
    N_GEN = 100

    # Real levels baseline
    evaluate_model("Real Levels (test set)", test, train)

    # Naive
    naive = NaiveBaseline()
    naive.load()
    naive_levels = naive.generate(n=N_GEN, seed=42)
    evaluate_model("Naive Baseline", naive_levels, train)

    # Classical
    bigram = BigramModel()
    bigram.load()
    bigram_levels = bigram.generate(n=N_GEN, seed=42)
    evaluate_model("Classical (Bigram)", bigram_levels, train)

    # VAE
    device = torch.device("cpu")
    vae = ConvVAE(latent_dim=64).to(device)
    vae.load_state_dict(torch.load(get_model_path("vae_best.pth"), map_location=device, weights_only=True))
    vae_levels = generate_levels(vae, n=N_GEN, temperature=1.0, seed=42, device=device)
    evaluate_model("Deep Learning (VAE)", vae_levels, train)
