#!/usr/bin/env bash
set -euo pipefail

python -m trainers.flowse --config configs/base.yaml --curated_list "" --exp.name ""
