#!/usr/bin/env bash
set -euo pipefail

python -m trainers.bsrrn --config configs/base.yaml --curated_list "${1:-data/curated/curated_train.csv}" --exp.name "${2:-bsrrn_experiment}"
