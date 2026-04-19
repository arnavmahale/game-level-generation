# Project Context Document - Game Level Generation

## Project Overview
Generating Mario-style 2D platformer levels using deep learning, trained on SuperTux game level data. This is for AIPI 540 (Duke University, Spring 2026) individual project. The project builds on a prior group project (CS4150 at Northeastern) but is entirely new code and approach.

## Prior Work (CS4150 Final Project)
- **Team**: Arnav Mahale, Jai Amin, Stephen Triandafellos
- **What they did**: Parsed SuperTux levels, trained a Conv-VAE, deployed on HuggingFace, rendered in Unreal Engine 5 Paper2D
- **Key limitations identified in their own report**:
  1. Only 3 tile classes (air=0, ground=1, hazard=2) — too simple
  2. Only used bottom 15 rows of 20-row grid, threw away top 5
  3. No quantitative evaluation — relied entirely on subjective playtesting
  4. No pathfinding validation for playability
  5. Small model: 32-dim latent, 30 epochs, 15x40 input
  6. UE5 Paper2D rendering was complex and Arnav wasn't involved in that part
  7. Random air blocks inside ground was a problem — fixed with column-collapse post-processing
- **Old repo**: `Project-main/` directory contains old code (vae_train.ipynb, supertux_parser.ipynb, UE5 C++ files)
- **Old data files**: `all_levels.txt` (raw tile IDs), `true.txt` (3-class version used for training)
- **IMPORTANT**: This new project must be entirely new work per course rules. No code reuse.

## Data Source
- **SuperTux GitHub**: https://github.com/SuperTux/supertux
- **Level format**: S-expression (.stl files) with tilemaps stored as run-length encoded integer arrays
- **Built-in levels**: ~110 across 5 world directories (world1, world2, bonus1, revenge_in_redmond, misc)
- **Tileset definition**: `data/images/tiles.strf` (9,296 lines) — defines tile IDs, attributes, and images
- **Community addons**: Available via GitHub addon repository (not used yet, potential data augmentation)

## Data Pipeline — What Was Built

### Step 1: Tile ID to Category Mapping (`scripts/build_tile_mapping.py`)
- **Input**: `data/raw/tiles.strf` (downloaded from SuperTux repo)
- **Output**: `data/processed/tile_id_to_category.json`
- **What it does**: Parses the S-expression tileset file to map each tile ID to a gameplay category

**Categories (8 total)**:
| ID | Name | Description |
|----|------|-------------|
| 0 | empty | Air/sky |
| 1 | solid | Ground blocks (any theme) |
| 2 | slope | Walkable inclines |
| 3 | platform | One-way platforms (unisolid) |
| 4 | bonus | Coins, bonus blocks, bricks |
| 5 | water | Water tiles |
| 6 | hazard | Spikes, lava, fire |
| 7 | decoration | Non-solid background elements |

**Design decision**: Map by gameplay function, not visual theme. A snow ground block and forest ground block are both category 1 (solid). Visual theming can be applied at render time.

**Tile attribute bitfield** (from `src/supertux/tile.hpp`):
- SOLID=0x0001, UNISOLID=0x0002, SLOPE=0x0010, COIN=0x0040
- ICE=0x0100, WATER=0x0200, HURTS=0x0400, FIRE=0x0800

**Parser challenges and fixes**:
1. **First attempt**: Only parsed bulk `(tiles ...)` blocks that had `(attributes ...)` sections. Missed 574 tile IDs (10.5% of tiles in level data).
2. **Root cause**: Many bulk blocks have NO attributes section — these are decoration tiles (attribute 0). The regex required attributes to be present.
3. **Fix**: Added a second regex pattern to catch bulk blocks WITHOUT attributes, defaulting those tile IDs to category 7 (decoration).
4. **Also parses**: Individual `(tile ...)` blocks with boolean properties like `(water #t)`, `(hurts #t)`, `(solid #t)`, `(object-name "bonusblock")`.
5. **Final coverage**: 6,180 tile IDs mapped. 1,525 of 1,569 non-zero tile IDs in levels mapped (99.9%). Remaining 44 unmapped IDs (0.1% of tiles) default to decoration.

### Step 2: Level Processing (`scripts/build_features.py`)
- **Input**: `data/raw/all_levels.txt` (2,433 level chunks, each 20x40, with raw tile IDs)
- **Output**: `data/processed/levels_categorized.jsonl` (1,509 filtered levels with 8 categories)

**How `all_levels.txt` was created** (by old project's parser):
- Original SuperTux levels are variable-width (up to 300+ columns wide)
- Parser sliced them into non-overlapping 20x40 chunks
- This created some chunks that are mostly or entirely empty air (from sky sections of tall levels)

**Empty chunk problem discovered**:
- 248 chunks (10.2%) were 100% empty air
- 661 chunks (27.2%) had <5% non-air tiles
- 936 chunks (38.5%) had <10% non-air tiles
- This would bias the model toward generating empty levels

**Solution**: Filter out chunks with <10% non-air tiles
- Kept: 1,509 levels
- Filtered: 924 levels
- Empty tile percentage dropped from 67.7% to 49.9%

**Final processed data distribution**:
| Category | % of tiles |
|----------|-----------|
| 0 (empty) | 49.85% |
| 1 (solid) | 26.87% |
| 7 (decoration) | 22.26% |
| 5 (water) | 0.58% |
| 6 (hazard) | 0.17% |
| 2 (slope) | 0.13% |
| 3 (platform) | 0.12% |
| 4 (bonus) | 0.03% |

**Note**: Categories 2-6 are very rare (<1% each). This was flagged as a potential experiment variable — compare model performance with different numbers of categories (e.g., 3 vs 8).

## Models — What Was Built

### Shared utilities (`scripts/data_utils.py`)
- Loads processed levels from JSONL
- Train/test split (85/15, seed=42)
- Constants: NUM_CATEGORIES=8, GRID_HEIGHT=20, GRID_WIDTH=40

### Model 1: Naive Baseline (`scripts/model_naive.py`)
- **Approach**: Learn average ground height from training data, generate levels with solid ground below that height and air above, with random per-column height variation (Gaussian noise)
- **Learned parameters**: mean_ground_height=6.9, height_std=8.1
- **Earlier version was more complex**: Originally did per-position frequency sampling (learned P(category | row, col) for every cell). Arnav pointed out this was too sophisticated for a "naive" baseline — should be truly naive like "just put ground at the bottom". Simplified to current approach.
- **Output**: `models/naive_baseline.pkl`

### Model 2: Classical ML — Bigram Transition Model (`scripts/model_classical.py`)
- **Approach**: Column-by-column generation using learned transition probabilities
- **Learned parameters**:
  - Vertical transitions: P(tile | tile_above) — 8x8 matrix
  - Horizontal transitions: P(tile | tile_left) — 8x8 matrix
  - Row priors: P(tile | row_index) — 20x8 matrix
  - First column distribution: P(tile | row) for column 0

**Generation failures and fixes**:
1. **First version**: Combined probabilities multiplicatively: `probs = horiz * vert * prior`. This caused near-total collapse to empty because empty→empty is the dominant transition in all three probability sources. Multiplying three distributions that all favor empty created near-certainty of empty for every cell.
   - Result: 3.2% ground coverage, 0% playability, 0.079 diversity
2. **Fix**: Changed to weighted average combination: `probs = 0.5*prior + 0.25*vert + 0.25*horiz`. This prevents any single strong prior from dominating.
   - Result: 27.0% ground coverage, 86% playability, 0.618 diversity

- **Output**: `models/bigram_model.pkl`

### Model 3: Deep Learning — Convolutional VAE (`scripts/model_deep.py`)
- **Architecture**: Conv-VAE with residual blocks
  - Encoder: Conv2d(8→64, stride=2) + ResBlock + Conv2d(64→128, stride=2) + ResBlock + Conv2d(128→256, stride=1) → fc to 64-dim latent
  - Decoder: fc(64→256*5*10) + ResBlock + ConvTranspose2d(256→128) + ResBlock + ConvTranspose2d(128→64) + ConvTranspose2d(64→8)
  - All layers use BatchNorm
  - Input: one-hot encoded 8-channel 20x40 grid
  - Output: 8-channel logits, softmax for probabilities
- **Latent dim**: 64 (vs 32 in old project)
- **Full grid**: Uses all 20 rows (vs old project's 15 rows)

**Training failures and fixes**:
1. **NumPy version incompatibility**: torch 2.2.2 is incompatible with numpy 2.x. Error: "A module compiled using NumPy 1.x cannot run in NumPy 2.4.2". Fixed by downgrading to numpy 1.26.4.
2. **Class weight problem (first training run)**:
   - Used inverse-frequency weighting: `weights = 1/counts`. This gave bonus tiles (0.03%) a weight ~1600x higher than empty (50%).
   - Result: Nearly uniform output across all 8 categories (each ~12.5%). Model learned to over-predict rare categories.
   - Loss: beta=0.5 was also too high for KL term
3. **Fix**: Changed to sqrt-inverse frequency: `weights = 1/sqrt(freq)`, normalized to mean=1. Also reduced beta from 0.5 to 0.1 and increased epochs from 100 to 150.
   - Result: {0: 411, 1: 229, 7: 74, 2: 24, 3: 29, 6: 16, 4: 15, 5: 2} — much closer to real distribution

**Training details**:
- Optimizer: Adam, lr=1e-3
- Scheduler: ReduceLROnPlateau (patience=10, factor=0.5)
- Batch size: 32
- Best val loss: ~0.97 (saved as vae_best.pth)
- Train/val: 1282/227 levels
- Device: CPU (no GPU available locally)

**Generation**: Sample z ~ N(0,1), decode to logits, temperature-scaled softmax, categorical sampling per pixel.

- **Output**: `models/vae_best.pth`

## Evaluation Metrics (`scripts/evaluate.py`)

### Metrics implemented:
1. **Tile Distribution Similarity**: Jensen-Shannon divergence between real and generated tile category distributions. Lower = better.
2. **Playability**: BFS-based pathfinding. Checks if player can traverse from leftmost column to rightmost. Player can walk (left/right on ground), jump (up to 4 tiles high, 3 wide), and fall. Standing requires: current cell not solid/hazard, cell below is walkable (solid/slope/platform).
3. **Structural Metrics**: Ground coverage (% walkable tiles), ground contiguity (% of ground tiles adjacent to other ground), hazard density.
4. **Diversity**: Average pairwise Hamming distance between generated levels (sampled 100 pairs).

### Results (100 generated levels per model):

| Metric | Real Levels | Naive | Bigram | VAE |
|--------|------------|-------|--------|-----|
| JS Divergence (↓) | 0.0009 | 0.0880 | 0.0000 | 0.0198 |
| Playability (↑) | 17.62% | 0.00% | 86.00% | 45.00% |
| Ground Coverage | 28.31% | 37.82% | 27.02% | 27.29% |
| Ground Contiguity | 61.60% | 100.0% | 90.12% | 52.22% |
| Hazard Density | 0.0022 | 0.0000 | 0.0020 | 0.0185 |
| Diversity (↑) | 0.6327 | 0.3399 | 0.6175 | 0.6740 |

### Key observations:
- Real levels show only 17.6% playability because many chunks are mid-level sections without ground at the starting edge
- Naive has 0% playability because solid columns with no gaps means no valid standing positions
- Bigram outperforms VAE on most metrics after the weighted-average fix
- VAE has highest diversity but lower spatial coherence (noisy tile placement)
- VAE over-represents rare categories slightly due to sqrt-inverse weighting (by design)

## Visual Comparison
- Saved to `data/outputs/model_comparison.png`
- Shows: Real level, Naive baseline, Bigram, VAE side by side
- Color mapping: sky blue=empty, brown=solid, tan=slope, wood=platform, gold=bonus, blue=water, red=hazard, green=decoration

## Environment Setup
- Python 3.11 (via Homebrew, `/usr/local/bin/python3.11`)
- Python 3.14 is system default but too new for PyTorch
- Virtual environment at `game-level-generation/venv/`
- Key packages: torch 2.2.2, numpy 1.26.4, scikit-learn 1.8.0, matplotlib
- **IMPORTANT**: numpy must stay <2.0 for torch 2.2.2 compatibility

## Repository Structure
```
game-level-generation/
├── README.md
├── .gitignore
├── PROJECT_CONTEXT.md          <- this file
├── scripts/
│   ├── data_utils.py           <- shared data loading utilities
│   ├── build_tile_mapping.py   <- parse tiles.strf → category mapping
│   ├── build_features.py       <- remap levels and filter empty chunks
│   ├── model_naive.py          <- naive baseline (random ground fill)
│   ├── model_classical.py      <- bigram transition model
│   ├── model_deep.py           <- convolutional VAE
│   └── evaluate.py             <- evaluation metrics
├── models/
│   ├── naive_baseline.pkl
│   ├── bigram_model.pkl
│   └── vae_best.pth
├── data/
│   ├── raw/
│   │   ├── all_levels.txt      <- 2,433 raw level chunks (tile IDs)
│   │   └── tiles.strf          <- SuperTux tileset definitions
│   ├── processed/
│   │   ├── tile_id_to_category.json  <- tile ID → category mapping
│   │   └── levels_categorized.jsonl  <- 1,509 processed levels
│   └── outputs/
│       └── model_comparison.png
├── notebooks/                   <- empty (for exploration only)
└── venv/                        <- Python 3.11 virtual environment
```

## What's Still TODO
1. **Tune VAE** — lower temperature sampling, possible post-processing to improve spatial coherence
2. **Interactive web app** (`app.py`) — must be publicly accessible, good UX, inference only. Plan: web-based with HTML Canvas rendering (no Unreal Engine). Users can pick model, adjust parameters, generate and visualize levels. Possibly play them in-browser with simple platformer physics.
3. **Deployment** — live for 1+ week after submission
4. **Experiment** — planned: tile vocabulary ablation (compare 3 categories vs 8 categories vs 20 categories). Could also do: training set size sensitivity, temperature sweep, latent dim comparison.
5. **setup.py** — script to run full pipeline
6. **requirements.txt** — document dependencies
7. **README.md** — project description and setup instructions
8. **.gitignore update** — ignore venv, large data files, model weights
9. **Written report** — NeurIPS-style paper with all required sections
10. **Demo Day pitch** — 5-min presentation

## Git Requirements (from rubric)
- Must use branching (develop/feature branches)
- Must use PRs
- No direct commits to main
- Only main branch will be evaluated

## Course Requirements Checklist
- [x] Three modeling approaches (naive, classical, deep learning)
- [x] Clear documentation of where each model can be found
- [ ] At least one focused experiment
- [ ] Interactive app (publicly accessible, good UX)
- [ ] Live for 1+ week
- [ ] Written report (NeurIPS-style)
- [ ] Demo Day pitch (5 min)
- [ ] Git best practices (branching, PRs)
- [ ] Proper repo structure
- [ ] Code quality (modularized, no loose code, descriptive names)
