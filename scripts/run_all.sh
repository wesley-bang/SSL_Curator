#!/usr/bin/env bash
set -euo pipefail

python -m scoring.score \
  --manifest data/toy_manifest.csv \
  --out scores.parquet \
  --ssl hubert_base \
  --layers-content 9 10 11 12 \
  --layers-speaker 3 4 5 6 \
  --mos dnsmos \
  --seed 1337

python -m selector.select \
  --scores scores.parquet \
  --quotas configs/quotas.yaml \
  --alpha 0.10 \
  --K-hours 10 \
  --diversity-min-cos 0.02 \
  --uncert-beta 0.0 \
  --out curated/curated_train_K=10.csv \
  --alpha-sweep logs/alpha_sweep.csv \
  --slice-stats logs/slice_stats.json

python -m eval.tail_metrics \
  --scores scores.parquet \
  --selected curated/curated_train_K=10.csv \
  --out eval/eval_report.md
