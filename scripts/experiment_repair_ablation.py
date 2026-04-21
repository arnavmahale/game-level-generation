"""
Repair ablation experiment.

Question: how much of the shipped app's playability comes from the BFS
repair post-pass, and where does it matter most?

Design:
  - Sample N levels from each (model, difficulty) cell.
  - For each sample, evaluate metrics in TWO conditions:
      * "no_repair"   — model output + enforce_layout only (silhouette).
                         This is the raw generative behavior the model
                         actually learned.
      * "with_repair" — full shipped pipeline: enforce_layout → repair.
  - Report per-cell playability rate (BFS with arc-aware physics) plus
    JS divergence (distribution fidelity), ground coverage, hazard
    density, and pairwise diversity.

Cells:
  - naive     (difficulty=50)        [baseline]
  - bigram    (difficulty=50)        [classical]
  - vae-easy  (bucket=0)
  - vae-med   (bucket=1)
  - vae-hard  (bucket=2)

Also reports the real test split as a ceiling (diversity + distribution
reference; playability from the same BFS). Repair does not apply to it.

Outputs:
  experiments/results/repair_ablation.json      — raw metrics
  experiments/results/repair_ablation.md        — summary table
  experiments/results/repair_ablation.png       — grouped bar chart
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import (
    load_levels, train_val_test_split_levels, get_model_path,
    GRID_HEIGHT, GRID_WIDTH,
)
from model_naive import NaiveBaseline
from model_classical import BigramModel
from model_deep import ConvVAE, generate_levels as vae_generate, three_class_to_api
from repair import enforce_layout, repair as repair_level
from evaluate import (
    check_playability, tile_distribution, js_divergence,
    structural_metrics, pairwise_diversity,
)


RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments", "results",
)


def collapse_to_api(level):
    """Match app.py's final collapse: slope/platform→solid, bonus/water/decoration→empty."""
    out = np.asarray(level).copy()
    out[(out == 2) | (out == 3)] = 1
    out[(out == 4) | (out == 5) | (out == 7)] = 0
    return out


def gen_naive(naive, n, seed, difficulty=50):
    # Match app.py: difficulty 0-100 → height_std 1.0-16.0.
    height_std = 1.0 + (difficulty / 100.0) * 15.0
    original = naive.height_std
    naive.height_std = height_std
    try:
        return naive.generate(n=n, seed=seed)
    finally:
        naive.height_std = original


def gen_bigram(bigram, n, seed, difficulty=50):
    t = difficulty / 100.0
    prior_weight = 0.2 + t * 0.6
    trans_weight = 0.4 - t * 0.3
    return bigram.generate(n=n, seed=seed, prior_weight=prior_weight, trans_weight=trans_weight)


def gen_vae(vae, n, seed, bucket, device):
    levels_3c = vae_generate(
        vae, n=n, bucket=bucket, seed=seed, device=device, guidance_scale=2.0,
    )
    # 3-class → 8-cat API encoding (hazards to 6) so enforce_layout sees
    # the same shape the app does.
    out = np.stack([three_class_to_api(l) for l in levels_3c])
    return out


def pipeline(levels, apply_repair):
    """Run the same post-processing the app does, gated on repair flag."""
    out = []
    for level in levels:
        layered = enforce_layout(level)
        if apply_repair:
            layered = repair_level(layered)
        out.append(collapse_to_api(layered))
    return np.stack(out)


def evaluate_cell(samples, train_levels):
    """Compute all metrics on a batch of samples (already post-processed)."""
    playable = [bool(check_playability(l)) for l in samples]
    play_rate = float(np.mean(playable))
    gen_dist = tile_distribution(samples)
    real_dist = tile_distribution(train_levels)
    js = float(js_divergence(real_dist, gen_dist))
    struct = structural_metrics(samples)
    diversity = float(pairwise_diversity(samples, n_samples=min(len(samples), 100)))
    return {
        "n": len(samples),
        "playability": play_rate,
        "playable_count": int(sum(playable)),
        "js_divergence": js,
        "ground_coverage": float(struct["ground_coverage"]),
        "ground_contiguity": float(struct["ground_contiguity"]),
        "hazard_density": float(struct["hazard_density"]),
        "diversity": diversity,
    }


def build_cells(n, seed, device):
    """Return dict: cell_name -> numpy array of raw (un-postprocessed) samples."""
    print(f"loading models…", flush=True)
    naive = NaiveBaseline(); naive.load()
    bigram = BigramModel(); bigram.load()
    vae = ConvVAE(latent_dim=64).to(device)
    vae.load_state_dict(torch.load(get_model_path("cvae_best.pth"), map_location=device, weights_only=True))
    vae.eval()

    cells = {}
    print(f"generating naive×50 (n={n})…", flush=True)
    cells["naive@50"] = gen_naive(naive, n=n, seed=seed, difficulty=50)
    print(f"generating bigram×50 (n={n})…", flush=True)
    cells["bigram@50"] = gen_bigram(bigram, n=n, seed=seed, difficulty=50)
    for bucket, name in [(0, "easy"), (1, "medium"), (2, "hard")]:
        print(f"generating vae×{name} (n={n})…", flush=True)
        cells[f"vae@{name}"] = gen_vae(vae, n=n, seed=seed, bucket=bucket, device=device)
    return cells


def run(n, seed):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cpu")

    levels = load_levels()
    train, _val, test = train_val_test_split_levels(levels)

    cells = build_cells(n=n, seed=seed, device=device)

    results = {"config": {"n_per_cell": n, "seed": seed}, "cells": {}}

    # Real test split as ceiling. Same post-processing as app (enforce_layout
    # keeps silhouette; repair is obviously a no-op on already-good levels).
    # We report 'real' under both keys so the row exists in both columns.
    print(f"evaluating real test ceiling (n={len(test)})…", flush=True)
    test_collapsed = np.stack([collapse_to_api(l) for l in test])
    results["cells"]["real@test"] = {
        "no_repair": evaluate_cell(test_collapsed, train),
        "with_repair": evaluate_cell(test_collapsed, train),
    }

    for cell_name, raw_samples in cells.items():
        print(f"evaluating {cell_name} …", flush=True)
        no_rep = pipeline(raw_samples, apply_repair=False)
        with_rep = pipeline(raw_samples, apply_repair=True)
        results["cells"][cell_name] = {
            "no_repair": evaluate_cell(no_rep, train),
            "with_repair": evaluate_cell(with_rep, train),
        }

    return results


def render_markdown(results):
    cells = results["cells"]
    cfg = results["config"]
    lines = []
    lines.append("# Repair Ablation Results\n")
    lines.append(f"- N per cell: {cfg['n_per_cell']}\n- Seed: {cfg['seed']}\n")
    lines.append("## Playability (arc-aware BFS)\n")
    lines.append("| Cell | No repair | With repair | Δ |")
    lines.append("|---|---:|---:|---:|")
    for name, cell in cells.items():
        nr = cell["no_repair"]["playability"]
        wr = cell["with_repair"]["playability"]
        delta = wr - nr
        lines.append(f"| {name} | {nr:.1%} | {wr:.1%} | {delta:+.1%} |")
    lines.append("")
    lines.append("## Distribution similarity (JS divergence vs real; lower=better)\n")
    lines.append("| Cell | No repair | With repair | Δ |")
    lines.append("|---|---:|---:|---:|")
    for name, cell in cells.items():
        nr = cell["no_repair"]["js_divergence"]
        wr = cell["with_repair"]["js_divergence"]
        lines.append(f"| {name} | {nr:.4f} | {wr:.4f} | {wr - nr:+.4f} |")
    lines.append("")
    lines.append("## Structural metrics (with repair)\n")
    lines.append("| Cell | Ground cov | Contiguity | Hazard dens | Diversity |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, cell in cells.items():
        c = cell["with_repair"]
        lines.append(
            f"| {name} | {c['ground_coverage']:.1%} | {c['ground_contiguity']:.1%} "
            f"| {c['hazard_density']:.4f} | {c['diversity']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def render_plot(results, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = results["cells"]
    names = list(cells.keys())
    nr = [cells[k]["no_repair"]["playability"] for k in names]
    wr = [cells[k]["with_repair"]["playability"] for k in names]

    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, nr, width, label="No repair", color="#d88c8c")
    ax.bar(x + width / 2, wr, width, label="With repair", color="#5fa776")
    ax.set_ylabel("Playability rate (BFS)")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_title(f"Repair ablation — playability per (model, difficulty) cell "
                 f"(n={results['config']['n_per_cell']})")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(wr):
        ax.text(x[i] + width / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=8)
    for i, v in enumerate(nr):
        ax.text(x[i] - width / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="samples per cell")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    results = run(n=args.n, seed=args.seed)

    json_path = os.path.join(RESULTS_DIR, "repair_ablation.json")
    md_path = os.path.join(RESULTS_DIR, "repair_ablation.md")
    png_path = os.path.join(RESULTS_DIR, "repair_ablation.png")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(md_path, "w") as f:
        f.write(render_markdown(results))
    if not args.no_plot:
        render_plot(results, png_path)

    print(f"\nwrote {json_path}")
    print(f"wrote {md_path}")
    if not args.no_plot:
        print(f"wrote {png_path}")
    print()
    print(render_markdown(results))


if __name__ == "__main__":
    main()
