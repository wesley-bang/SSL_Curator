# urgent2026 Coverage-Constrained CVaR Data Curation

This repository implements the URGENT 2026 Track-1 data curation pipeline. It scores a speech manifest with deterministic proxy measures, enforces coverage constraints via CVaR-optimised selection with diversity and speaker leakage controls, and emits lightweight evaluation summaries.

## Quickstart
1. `python -m venv .venv && .venv/Scripts/activate` (Windows) or `source .venv/bin/activate` (Unix)
2. `pip install -r requirements.txt`
3. `bash scripts/run_all.sh`
4. Inspect outputs: `scores.parquet`, `curated/curated_train_K=10.csv`, `logs/*.{csv,json}`, `eval/eval_report.md`

## Pipeline Overview
- **scoring.score**: Generates toy data if needed, synthesises audio when files are missing, computes MOS proxies and SSL embeddings (or deterministic fallbacks), and writes `scores.parquet`.
- **selector.select**: Loads quotas, computes a weighted loss proxy, fills mandatory slice coverage, performs a diversity-aware budget fill, and records CVaR diagnostics plus slice statistics.
- **eval.tail_metrics**: Joins the curated set with scores, summarises proxy means, tail hardness p10/p25, and per-slice counts in Markdown.

## Configuration
- `configs/quotas.yaml` encodes minimum counts per `{distortion}|{lang}|{sr}` slice. Adjust values to tighten or relax coverage.
- `configs/bs_example.yaml` is a placeholder backbone stub for downstream training glue.
- The scoring CLI regenerates `data/toy_manifest.csv` deterministically when missing; modify the helper in `scoring/score.py` for custom corpora.

## CLI Highlights
- `python -m scoring.score --help` exposes MOS backend, SSL layers, and seeding controls.
- `python -m selector.select --help` documents loss weights, diversity margin, speaker checks, and diagnostic outputs.
- `python -m eval.tail_metrics --help` prints available report options.

## Testing
Run `pytest -q` to execute the unit tests covering CVaR, quota feasibility, and diversity guards.