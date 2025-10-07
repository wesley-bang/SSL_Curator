#!/usr/bin/env bash
set -euo pipefail

python -m scoring.score --config configs/scoring.yaml data.manifest="$1" --out "${2:-data/scores/train_scores.parquet}"
