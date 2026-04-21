#!/usr/bin/env bash
# Deploy to Hugging Face Spaces via an orphan branch so legacy non-LFS blobs
# in history don't trip the LFS hook. README.md gets a HF-specific frontmatter
# block prepended here so the GitHub copy stays clean.
set -euo pipefail

cd "$(dirname "$0")/.."

HF_FRONTMATTER=$(cat <<'EOF'
---
title: TerrainGen
emoji: 🏔️
colorFrom: blue
colorTo: yellow
sdk: docker
pinned: false
---

EOF
)

MSG="${1:-Deploy}"

git branch -D hf-deploy 2>/dev/null || true
git checkout --orphan hf-deploy
git add -A
git add --renormalize .
# Prepend HF frontmatter to README on the deploy branch only.
printf '%s\n%s' "$HF_FRONTMATTER" "$(cat README.md)" > README.md
git add README.md
git commit -m "$MSG"
git push hf hf-deploy:main --force
git checkout main
