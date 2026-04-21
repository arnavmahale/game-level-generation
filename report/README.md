# TerrainGen technical report

## Files

- `report.tex` — main report, self-contained LaTeX source.
- `references.bib` — bibliography.
- `figures/error_cases.png` — error-analysis figure (5 mispredictions).
- `figures/error_cases.json` — metadata for the 5 cases.

`report.tex` also pulls `../experiments/results/repair_ablation.png` as the playability-outcome figure.

## Compiling

Easiest path — Overleaf:

1. Create a new project, upload `report.tex`, `references.bib`, the `figures/` directory, and `experiments/results/repair_ablation.png`.
2. In Overleaf, place `repair_ablation.png` at path `../experiments/results/repair_ablation.png` or edit the `\includegraphics` path in `report.tex` to match your upload layout.

Local path (requires a TeX installation — `tectonic` is the lightest-weight option):

```bash
# install once
brew install tectonic

# from this directory
tectonic report.tex
```

Output is `report.pdf` in this directory.

## Regenerating figures

- `../scripts/experiment_repair_ablation.py --n 200 --seed 42` — playability plot.
- `../scripts/report_error_cases.py` — 5-case error figure.

Both run from the `game-level-generation/` repo root and expect the trained model checkpoints in `../models/`.
