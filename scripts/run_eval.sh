#!/usr/bin/env bash
set -euo pipefail

python -m eval.eval --config configs/eval.yaml --ckpt "${1:-experiments/bsrrn_experiment/train/checkpoints/best.ckpt}"
