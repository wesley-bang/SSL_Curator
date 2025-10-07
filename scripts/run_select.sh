#!/usr/bin/env bash
set -euo pipefail

python -m selector --config configs/select/cvar.yaml --scores "${1:-data/scores/train_scores.parquet}" --quotas configs/quotas.yaml --out "${2:-data/curated/curated_train.csv}"
