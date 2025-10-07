#!/usr/bin/env bash
set -euo pipefail

scripts/run_score.sh "${1:-data/manifests/urgent2026/train.jsonl}" "${2:-data/scores/train_scores.parquet}"
scripts/run_select.sh "${2:-data/scores/train_scores.parquet}" "${3:-data/curated/curated_train.csv}"
scripts/run_train.sh "${3:-data/curated/curated_train.csv}" "${4:-bsrrn_experiment}"
scripts/run_eval.sh "${5:-experiments/${4:-bsrrn_experiment}/train/checkpoints/best.ckpt}"
