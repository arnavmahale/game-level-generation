# Repair Ablation Results

- N per cell: 200
- Seed: 42

## Playability (arc-aware BFS)

| Cell | No repair | With repair | Δ |
|---|---:|---:|---:|
| real@test | 23.0% | 23.0% | +0.0% |
| naive@50 | 0.0% | 56.0% | +56.0% |
| bigram@50 | 0.0% | 84.5% | +84.5% |
| vae@easy | 0.0% | 31.5% | +31.5% |
| vae@medium | 0.5% | 78.5% | +78.0% |
| vae@hard | 0.0% | 15.5% | +15.5% |

## Distribution similarity (JS divergence vs real; lower=better)

| Cell | No repair | With repair | Δ |
|---|---:|---:|---:|
| real@test | 0.0111 | 0.0111 | +0.0000 |
| naive@50 | 0.0225 | 0.0193 | -0.0032 |
| bigram@50 | 0.0112 | 0.0113 | +0.0001 |
| vae@easy | 0.0191 | 0.0171 | -0.0020 |
| vae@medium | 0.0119 | 0.0114 | -0.0004 |
| vae@hard | 0.0172 | 0.0168 | -0.0004 |

## Structural metrics (with repair)

| Cell | Ground cov | Contiguity | Hazard dens | Diversity |
|---|---:|---:|---:|---:|
| real@test | 37.4% | 99.3% | 0.0172 | 0.3475 |
| naive@50 | 42.9% | 100.0% | 0.0000 | 0.2796 |
| bigram@50 | 35.2% | 96.8% | 0.0140 | 0.2631 |
| vae@easy | 36.1% | 95.0% | 0.0604 | 0.2801 |
| vae@medium | 38.3% | 96.0% | 0.0136 | 0.2180 |
| vae@hard | 31.9% | 93.2% | 0.0570 | 0.3046 |
