# Game Level Generation

Generative models that produce playable 2D platformer levels in the style of *SuperTux* / Super Mario Bros. Three models (naive baseline, bigram transitions, conditional Conv-VAE) are trained on ~15k 20×40 chunks mined from 110 hand-crafted SuperTux levels, with a BFS playability repair post-pass and a fixed sky/floor layout enforcement.

**Course:** AIPI 540 — Deep Learning Applications (Duke, Spring 2026)
**Author:** Arnav Mahale

---

## Live demo

*(Deployed link will go here.)*

## Quick start (local)

```bash
# 1. Python 3.11 + venv (torch 2.2.2 requires numpy<2.0)
/usr/local/bin/python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Build features + train all models
python setup.py

# 3. Build the React frontend
cd frontend && npm ci && npm run build && cd ..

# 4. Run the app
python app.py        # http://localhost:5050
```

The naive and bigram models train in seconds. The conditional VAE trains for 150 epochs on CPU (~3 hrs); pass `--features-only` or `--models-only` to run pieces independently.

## Project structure

```
.
├── app.py                        # Flask backend + static SPA
├── setup.py                      # End-to-end pipeline orchestrator
├── requirements.txt
├── Dockerfile                    # Multi-stage: React build → Python+gunicorn
├── data/
│   ├── raw/                      # SuperTux .stl level files + tile defs
│   └── processed/                # tile_id_to_category.json, levels_*.jsonl
├── models/                       # Trained checkpoints (.pkl, .pth)
├── scripts/
│   ├── parse_stl.py              # S-expression parser for SuperTux levels
│   ├── build_tile_mapping.py     # Build tile_id → 8-category mapping
│   ├── build_features.py         # Sliding-window training data
│   ├── data_utils.py             # Constants, IO, train/val/test split
│   ├── difficulty.py             # BFS structural difficulty → buckets
│   ├── model_naive.py            # Baseline: learned mean ground height
│   ├── model_classical.py        # Bigram transition model
│   ├── model_deep.py             # Conditional Conv-VAE
│   ├── repair.py                 # Layout + BFS playability repair
│   └── evaluate.py               # Metrics + full eval driver
├── frontend/                     # React + Vite SPA
└── PROJECT_CONTEXT.md            # Design decisions and failure-mode log
```

## Models

| Model | Type | Description | File |
|---|---|---|---|
| Naive | Baseline | Per-column ground height drawn from training mean ± std | `scripts/model_naive.py` |
| Bigram | Classical ML | Horizontal + vertical transitions, weighted-averaged with row prior | `scripts/model_classical.py` |
| Conditional Conv-VAE | Deep | 64-dim latent + 3-bucket difficulty conditioning; 3-class output | `scripts/model_deep.py` |

All three expose the same `fit(levels) / generate(n, seed, ...) / save() / load()` interface.

## Difficulty conditioning

Every training chunk is scored by a weighted BFS-based metric (`0.2·jumps + 0.4·hazards + 0.4·max_gap_width`), rank-bucketed into three equal tertiles (`easy / medium / hard`), and the tertile index is fed to the VAE decoder as a one-hot concatenated onto the latent. At inference the UI slider maps 0–100 onto the three buckets, so "difficulty" is a learned structural property rather than just a temperature knob. See `scripts/difficulty.py`.

## Tile categories

Raw levels are parsed into 20×40 grids of 8 gameplay categories:

| ID | Category | ID | Category |
|---|---|---|---|
| 0 | empty | 4 | bonus |
| 1 | solid | 5 | water |
| 2 | slope | 6 | hazard |
| 3 | platform | 7 | decoration |

The deep model trains on the 3 gameplay categories the in-browser game actually uses (empty / solid / hazard); outputs are remapped to the 8-category schema for the API. This avoids wasting model capacity on visual-only distinctions (bonus/water/decoration → empty at playtime; slope/platform → solid).

## Training / eval protocol

- **Split:** 80 / 10 / 10 train / val / test, deterministic (`seed=42`). Val drives VAE checkpoint selection; test is held out for the final evaluation table only.
- **Class weights:** sqrt-inverse frequency (categories 2–6 are each <1% of pixels).
- **Layout enforcement:** top 4 rows forced to empty, bottom 2 rows forced to solid, baked into training data and reapplied at inference so the VAE doesn't spend capacity modelling the fixed sky/floor bands.
- **Playability repair:** BFS-based post-pass that adds a minimal number of platform tiles to make the level traversable left-to-right.

Run the full evaluation (JS divergence, playability, ground coverage/contiguity, hazard density, pairwise diversity):

```bash
python scripts/evaluate.py
```

## Deployment

```bash
docker build -t level-gen .
docker run -p 8080:8080 level-gen        # http://localhost:8080
```

The Dockerfile is multi-stage: builds the React frontend with Node 20, then copies the static assets into a Python 3.11 image that serves them via gunicorn.

## Data source

SuperTux is open source (GPL-3.0). Levels are in `data/raw/stl_levels/`, copied from the [SuperTux GitHub repo](https://github.com/SuperTux/supertux). Tile definitions come from `data/raw/tiles.strf`.

## License

Code: MIT. Data: GPL-3.0 (from SuperTux).
