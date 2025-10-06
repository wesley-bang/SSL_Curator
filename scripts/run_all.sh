#!/usr/bin/env bash
set -euo pipefail

resolve_python() {
  if [ -n "${PYTHON:-}" ]; then
    if command -v "$PYTHON" >/dev/null 2>&1 || [ -x "$PYTHON" ]; then
      echo "$PYTHON"
      return 0
    fi
    echo "Provided PYTHON=$PYTHON not found" >&2
    return 1
  fi

  if [ -x "./.venv/bin/python" ]; then
    echo "./.venv/bin/python"
    return 0
  fi
  if [ -x "./.venv/Scripts/python.exe" ]; then
    echo "./.venv/Scripts/python.exe"
    return 0
  }

  for candidate in python python3 "py -3" "py -3.11"; do
    if command -v ${candidate%% *} >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done

  echo "python"
  return 1
}

if ! PY_CMD=$(resolve_python); then
  cat >&2 <<\EOF
Unable to locate a Python interpreter. Install Python 3.10+ and make sure it is on PATH,
or set the PYTHON environment variable to the interpreter you installed packages into.
EOF
  exit 1
fi

exec_py() {
  eval "$PY_CMD" "${@}"
}

exec_py -m scoring.score \
  --manifest data/toy_manifest.csv \
  --out scores.parquet \
  --ssl hubert_base \
  --layers-content 9 10 11 12 \
  --layers-speaker 3 4 5 6 \
  --mos dnsmos \
  --seed 1337

exec_py -m selector.select \
  --scores scores.parquet \
  --quotas configs/quotas.yaml \
  --alpha 0.10 \
  --K-hours 10 \
  --diversity-min-cos 0.02 \
  --uncert-beta 0.0 \
  --out curated/curated_train_K=10.csv \
  --alpha-sweep logs/alpha_sweep.csv \
  --slice-stats logs/slice_stats.json

exec_py -m eval.tail_metrics \
  --scores scores.parquet \
  --selected curated/curated_train_K=10.csv \
  --out eval/eval_report.md
