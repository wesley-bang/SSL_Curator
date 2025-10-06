# URGENT 2026 – Curator Codebase Spec
**Spec-Version:** 1.1 (2025-10-07)

> **AI implementers (GPT-5-Codex in VS Code):**  
> This file is the **source of truth**. Read it and follow it exactly.  
> - **Do not** invent paths or rename files not listed here.  
> - All parameters must come from YAML configs (no magic numbers).  
> - Keep CLIs stable. If you improve something, keep the same public interfaces.  
> - If something is underspecified, emit a TODO stub **without** changing the public API.

## How to invoke me (simple)
- Keep this file **open** in VS Code when you prompt GPT-5-Codex.
- To build any layer, just say in chat:
  **`Execute: BATCH <N>`**  
  where `<N>` ∈ { **0, 1, 2, 3, 4, 5, 6** } (see **Batch Index** below).
- You can also run everything: **`Execute: BATCH 0-6`** (sequential). We recommend running them in order to catch errors early.

## OUTPUT FORMAT (must follow exactly)
For every file you generate, output **one fenced code block per file** with a header line:

```

// FILE: path/to/file.ext <full file content with no omissions>

```

Rules:
- **One file per block.**
- **Full content** (no ellipses, no “snippets”).
- Use **exact paths** defined in this spec.
- **Overwrite** files if they already exist.

## Batch Index (what each batch does & why)

- **Batch 0 — Datasets (foundation):**  
  Create `datasets/` and `configs/datasets/`; implement the single CLI to **download/prepare/manifest** data.  
  **Why:** All later stages depend on standardized manifests.

- **Batch 1 — Scaffold:**  
  Create the full repo tree, stubs, `requirements.txt`, `pyproject.toml`, `.gitignore`, `Makefile`, minimal `README.md`.  
  **Why:** A visible structure reduces confusion and anchors subsequent code.

- **Batch 2 — Selector plumbing:**  
  Implement `selector/registry.py`, `selector/base.py`, `selector/__main__.py`, `selector/cvar.py`, plus `configs/select/cvar.yaml`, `configs/quotas.yaml`.  
  **Why:** Selection is the contribution; needs a stable plugin API/CLI.

- **Batch 3 — Scoring stub:**  
  Implement `scoring/score.py` + `configs/scoring.yaml`. Read manifest JSONL → emit required `scores.parquet` columns.  
  **Why:** Enables end-to-end wiring before MOS/SSL backends are finalized.

- **Batch 4 — Trainers:**  
  Implement `trainers/bsrrn.py` (+ config hooks) to call the official baseline and route artifacts to `experiments/{exp_name}`.  
  **Why:** Centralizes training hygiene (configs, logs, checkpoints).

- **Batch 5 — Eval, scripts, tests:**  
  Add `eval/` stubs, shell scripts under `scripts/`, and smoke tests.  
  **Why:** Close the loop and make CI/automation easy.

- **Batch 6 — README polish:**  
  Replace `README.md` with Quickstart, repo map, datasets usage, add-a-selector guide, reproducibility.  
  **Why:** New collaborators should understand the repo in 60 seconds.

---

## Executive Summary (≤200 words)

We are building a clean, extensible pipeline for **Universal Speech Enhancement (URGENT 2026 – Track-1)** focused on **data curation over scale**. The pipeline is:

**score → select → train → eval**  
1) **Scoring** computes per-utterance proxies (MOS, SSL embeddings, SDR/eSTOI proxies).  
2) **Selection** chooses a training subset via **pluggable strategies** (CVaR, coverage-only, MOS-threshold) with **coverage quotas** and **diversity controls**.  
3) **Training** wraps the official baselines (BSRNN/FlowSE) to train on the curated list.  
4) **Evaluation** runs official metrics and our tail/slice analyses.

Everything is **config-driven**, **reproducible**, and designed for **fast iteration**: new selectors are one file with a decorator; experiments are self-contained under `experiments/{exp_name}`. A **datasets** module standardizes “download/prepare/manifest” into a single CLI so training needs no code changes when switching datasets. The repo must be understandable in 60 seconds, with discoverable commands and sensible defaults.

---

## North-Star Principles

- **One folder = one purpose:** `scoring/`, `selector/`, `trainers/`, `eval/`, `datasets/`, `tools/`.  
- **Config-driven:** YAML for paths, quotas, α, K hours, models, dataset choice.  
- **Pluggable selection:** add a file + `@register_selector`.  
- **Experiment hygiene:** each run writes a frozen config snapshot, selection CSV + meta hashes, logs, metrics, and checkpoints.  
- **No hard-coding:** respect env overrides (e.g., `DATA_ROOT`) and YAML.  
- **Flexibility:** easy subsetting (e.g., English-only, specific SR, distortion slices).

---

## Repository Layout (authoritative)

```

urgent2026-curator/
├─ README.md
├─ requirements.txt
├─ pyproject.toml
├─ .gitignore
├─ Makefile
├─ docs/
│  └─ CODEBASE_SPEC.md          # ← this file
├─ configs/
│  ├─ base.yaml
│  ├─ data.yaml                 # global dataset routing
│  ├─ scoring.yaml
│  ├─ quotas.yaml
│  ├─ select/
│  │  ├─ cvar.yaml
│  │  ├─ coverage_only.yaml
│  │  └─ mos_threshold.yaml
│  ├─ train/
│  │  ├─ bsrrn.yaml
│  │  └─ flowse.yaml
│  ├─ eval.yaml
│  └─ datasets/
│     ├─ urgent2026.yaml
│     └─ TEMPLATE.yaml
├─ data/
│  ├─ raw/                      # downloads
│  ├─ external/                 # pre-simulated or simulated audio
│  ├─ processed/                # (optional) normalized outputs
│  ├─ manifests/
│  │  └─ urgent2026/{train,dev,test}.jsonl
│  ├─ scores/
│  └─ curated/
├─ scoring/
│  ├─ **init**.py
│  ├─ score.py                  # CLI: manifests → scores.parquet
│  ├─ mos_backends/             # DNSMOS/NISQA/UTMOS adapters
│  ├─ ssl_backends/             # HuBERT/WavLM (S3PRL) embeddings
│  ├─ proxies/                  # SDR/eSTOI proxies, ASR consistency
│  └─ utils.py
├─ selector/
│  ├─ **init**.py
│  ├─ **main**.py               # CLI: dispatch via registry
│  ├─ registry.py
│  ├─ base.py
│  ├─ cvar.py
│  ├─ coverage_only.py
│  └─ mos_threshold.py
├─ datasets/
│  ├─ **init**.py
│  ├─ cli.py                    # CLI: download | prepare | verify | manifest
│  ├─ registry.py
│  ├─ base.py
│  ├─ urgent2026.py
│  ├─ utils.py
│  └─ cards/
│     ├─ urgent2026.md
│     └─ README.md
├─ trainers/
│  ├─ **init**.py
│  ├─ bsrrn.py                  # wraps official trainer
│  └─ flowse.py
├─ eval/
│  ├─ **init**.py
│  ├─ eval.py                   # official metrics + slice reports
│  └─ tail_metrics.py
├─ tools/
│  ├─ manifest_utils.py
│  ├─ hashing.py
│  └─ namegen.py
├─ scripts/
│  ├─ get_data.sh
│  ├─ prepare_manifests.sh
│  ├─ run_score.sh
│  ├─ run_select.sh
│  ├─ run_train.sh
│  ├─ run_eval.sh
│  └─ run_all.sh
├─ experiments/
│  └─ .gitkeep                  # per-run artifacts live here
├─ tests/
│  ├─ test_cvar.py
│  ├─ test_quota_feasibility.py
│  └─ test_reproducibility.py
└─ third_party/
└─ urgent2026_baseline/      # git submodule (no edits)

````

**One-liners:**
- `scoring/`: Reads manifests, computes proxies/embeddings → `scores.parquet`.  
- `selector/`: Pluggable selection strategies, outputs `selection.csv` (+ meta).  
- `datasets/`: One CLI to **download/prepare/manifest** any dataset; **standardized** JSONL manifests so training code never changes.  
- `trainers/`: Thin wrappers that call the **official** baseline trainer and route artifacts to `experiments/{exp_name}`.  
- `eval/`: Evaluation + slice/tail summaries.  
- `tools/`: Small helpers for manifests, hashing, name generation.  
- `configs/`: Everything configurable (paths, quotas, α, K hours, dataset choice).

---

## Dataflow & Where Each Part Is Used

1) **datasets** produces **manifests** (`data/manifests/<dataset>/{train,dev,test}.jsonl`).  
   → Inputs for `scoring/score.py`.

2) **scoring** reads **train** manifest → writes `data/scores/train_scores.parquet`.  
   → `selector/` consumes this parquet (plus `quotas.yaml`).

3) **selector** reads `scores.parquet` + quotas → writes `data/curated/curated_train_*.csv` (+ meta JSON with hashes).  
   → `trainers/` use the curated list to filter data loading.

4) **trainers** train the baseline model on the curated set → write logs/checkpoints under `experiments/{exp_name}`.  
   → `eval/` loads checkpoint to compute official metrics.

---

## Public Schemas (authoritative)

### Manifest JSONL row (produced by `datasets/`)
```json
{
  "utt_id": "unique_string",
  "path": "/abs/path/to/file.wav",
  "duration_sec": 3.21,
  "sr": 16000,
  "language": "en",
  "distortion": "noise",
  "split": "train"
}
````

### `scores.parquet` columns (produced by `scoring/score.py`)

Required:

* `utt_id: str`
* `path: str`
* `duration_sec: float`
* `sr: int`
* `language: str` (optional but recommended)
* `distortion: str` (optional but recommended)
* `loss_proxy: float` (tail surrogate; higher = “harder”)
* `ssl_embed: list[float]` (vector; JSON/bytes if needed)

Optional:

* `mos_dns`, `mos_nisqa`, `uncert`, `asr_consistency`, …

### `selection.csv` (produced by `selector/`)

* CSV with at least `utt_id`.
* Sidecar `${out}_meta.json` with: {config snapshot, scores path, n_selected, input hashes}.

---

## CLI Contracts (verbatim)

**Datasets:**

```bash
python -m datasets.cli \
  --dataset_cfg configs/datasets/urgent2026.yaml \
  --data_cfg configs/data.yaml \
  --stage all \
  --split all
# stages: download | prepare | verify | manifest | all
# split:  train | dev | test | all
```

**Scoring:**

```bash
python -m scoring.score \
  --config configs/scoring.yaml \
  data.manifest=data/manifests/urgent2026/train.jsonl \
  --out data/scores/train_scores.parquet
```

**Selection:**

```bash
python -m selector \
  --config configs/select/cvar.yaml \
  --scores data/scores/train_scores.parquet \
  --quotas configs/quotas.yaml \
  --out data/curated/curated_train_K=700h.csv
```

**Training:**

```bash
python -m trainers.bsrrn \
  --config configs/base.yaml \
  --curated_list data/curated/curated_train_K=700h.csv \
  --exp.name 251007_bsrrn_cvar_K700h_a0.10_s7
```

**Evaluation:**

```bash
python -m eval.eval \
  --config configs/eval.yaml \
  --ckpt experiments/251007_bsrrn_cvar_K700h_a0.10_s7/train/checkpoints/best.ckpt
```

---

## Configuration (key files & flexibility knobs)

### `configs/data.yaml` (global data routing & quick filters)

```yaml
data:
  dataset: "urgent2026"
  root: "data"
  manifests:
    train: "${data.root}/manifests/${data.dataset}/train.jsonl"
    dev:   "${data.root}/manifests/${data.dataset}/dev.jsonl"
    test:  "${data.root}/manifests/${data.dataset}/test.jsonl"
  dirs:
    raw:       "${data.root}/raw/${data.dataset}"
    external:  "${data.root}/external/${data.dataset}"
    processed: "${data.root}/processed/${data.dataset}"
  force_rebuild: false
  verify_checksums: true

  # Global *filters* used by scoring/selection loaders for fast, focused runs
  filters:
    languages: ["en"]          # set [] or null for all languages
    sample_rates: [16000]      # e.g., focus on 16 kHz
    distortions: []            # e.g., ["noise","reverb"]; empty = all
    max_hours: null            # cap size for quick experiments
```

> **Why filters?** Enable English-only, 16 kHz-only, or small-hour caps **without code changes**.

### `configs/select/cvar.yaml`

```yaml
select:
  name: "cvar"
  K_hours: 700
  alpha: 0.10
  diversity:
    min_cosine: 0.15
  penalties:
    quota_violation: 10.0
```

### `configs/quotas.yaml` (example)

```yaml
distortion:
  reverb: 0.12
  noise: 0.14
  clip: 0.14
  codec: 0.14
  packetloss: 0.14
  bandlimit: 0.16
  device: 0.16
```

---

## Datasets Module (first-class support)

**Goal:** a single CLI (`datasets.cli`) manages **download → prepare → verify → manifest** for any dataset. Training never changes; only manifests change.

**Key files**

* `datasets/cli.py` — runs stages with `--dataset_cfg` + `--data_cfg`.
* `datasets/base.py` — `BaseDataset` with `_write_manifest()`.
* `datasets/urgent2026.py` — adapter for Track-1 (pre-simulated **or** run simulation via the official repo).
* `datasets/registry.py` — registers datasets by name.
* `configs/datasets/urgent2026.yaml` — where to find pre-simulated pack or how to simulate.
* `configs/datasets/TEMPLATE.yaml` — copy to add a new dataset.

**Reasoning:** Centralizing data setup eliminates “where is the data?” and “what do I change to train on it?”; manifests unify formats so the rest of the pipeline is untouched.

---

## Naming, Artifacts, and Checkpoints

* Experiment names from `configs/base.yaml`, e.g.:

  ```
  ${now:%y%m%d}_${train.model}_${select.name}_K${select.K_hours}h_a${select.alpha}_s${seed}
  ```
* Artifacts under `experiments/{exp_name}/`:

  * `config.yaml` (frozen merged config)
  * `selection.csv` and `selection_meta.json` (hashes: `scores.parquet`, `quotas.yaml`, code SHA)
  * `train/checkpoints/{best.ckpt,last.ckpt,epoch-*.ckpt}`
  * `train/logs/{console.txt,events.jsonl}`
  * `eval/{metrics_*.json,slice_reports/*.csv}`
* Maintain a `experiments/runs.csv` registry:
  `exp_name, git_sha, config_sha, selection_sha, model, epochs, train_minutes, best_val_metric, path`.

---

## Requirements, Makefile, and Ignore

**requirements.txt**

```
numpy>=1.26
scipy>=1.11
pandas>=2.1
pyyaml>=6.0.1
omegaconf>=2.3.0
tqdm>=4.66
rich>=13.7
soundfile>=0.12.1
librosa>=0.10.1
pyarrow>=15.0.0
torch
torchaudio
s3prl
transformers
requests>=2.32
pytest>=8.0
black>=24.4
ruff>=0.4
mypy>=1.8
```

**Makefile**

```make
.PHONY: setup lint fmt test run_all

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt
	git submodule update --init --recursive

lint:
	ruff check .

fmt:
	black .

test:
	pytest -q

run_all:
	bash scripts/run_all.sh
```

**.gitignore**

```
__pycache__/
*.pyc
.venv/
.env
data/scores/
data/curated/
experiments/
logs/
.vscode/
.idea/
third_party/urgent2026_baseline/.venv/
```

---

## Critical Code Snippets (paste exactly unless improving compatibly)

> These are minimum working versions used in early batches. You may refine internals later, but **do not** change public CLIs or schemas.

```python
// FILE: selector/registry.py
from typing import Dict, Type

_REGISTRY: Dict[str, Type] = {}

def register_selector(name: str):
    def deco(cls):
        if name in _REGISTRY:
            raise KeyError(f"Selector '{name}' already registered")
        _REGISTRY[name] = cls
        return cls
    return deco

def build(name: str, **kwargs):
    try:
        cls = _REGISTRY[name]
    except KeyError as e:
        raise KeyError(f"Unknown selector '{name}'. Known: {list(_REGISTRY)}") from e
    return cls(**kwargs)
```

```python
// FILE: selector/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd

class BaseSelector(ABC):
    """Base class for data selection strategies."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def select(self, df_scores: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with at least ['utt_id'] for selected rows."""
        raise NotImplementedError
```

```python
// FILE: selector/__main__.py
import argparse, json, pandas as pd, logging
from omegaconf import OmegaConf
from .registry import build

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML for selector params")
    ap.add_argument("--scores", required=True, help="Path to scores.parquet")
    ap.add_argument("--quotas", required=False, help="YAML of coverage quotas")
    ap.add_argument("--out", required=True, help="Output curated CSV path")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.quotas:
        cfg.select.quotas = args.quotas
    df = pd.read_parquet(args.scores)

    selector = build(cfg.select.name, **OmegaConf.to_container(cfg.select, resolve=True))
    df_sel = selector.select(df)

    df_sel[["utt_id"]].to_csv(args.out, index=False)
    meta = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "scores_path": args.scores,
        "n_selected": int(df_sel.shape[0]),
    }
    with open(args.out.replace(".csv", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
```

```python
// FILE: selector/cvar.py
from __future__ import annotations
import numpy as np, pandas as pd
from .base import BaseSelector
from .registry import register_selector

@register_selector("cvar")
class CvarSelector(BaseSelector):
    """
    Select K hours minimizing CVaR_alpha of a proxy loss L(u),
    while meeting (approximate) quotas and enforcing diversity.
    Required columns: ['utt_id','duration_sec','loss_proxy','distortion','language','sr','ssl_embed']
    """
    def __init__(self, K_hours: float, alpha: float, quotas=None, diversity=None, penalties=None, **_):
        super().__init__()
        self.K_hours = float(K_hours)
        self.alpha = float(alpha)
        self.quotas = quotas
        self.diversity = diversity or {"min_cosine": 0.15}
        self.penalties = penalties or {"quota_violation": 10.0}

    def select(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Optional upstream filters (see configs/data.yaml)
        for col, key in (("language","languages"),("sr","sample_rates"),("distortion","distortions")):
            flt = self.kwargs.get("filters", {}).get(key)
            if flt: df = df[df[col].isin(flt)]
        df["hours"] = df["duration_sec"] / 3600.0

        # Tail focus (greedy; replace with true CVaR solver later)
        df = df.sort_values("loss_proxy", ascending=False)
        picked, hours = [], 0.0

        def ok_diversity(row, pool):
            if not pool: return True
            v = np.asarray(row["ssl_embed"], dtype=float)
            for p in pool[-50:]:
                w = np.asarray(p["ssl_embed"], dtype=float)
                denom = (np.linalg.norm(v)*np.linalg.norm(w) + 1e-8)
                cos = float(np.dot(v,w) / denom)
                if cos > 1 - self.diversity["min_cosine"]:
                    return False
            return True

        for _, row in df.iterrows():
            if hours >= self.K_hours: break
            if ok_diversity(row, picked):
                picked.append(row)
                hours += row["hours"]
        return pd.DataFrame(picked)
```

```python
// FILE: scoring/score.py
import argparse, pandas as pd, logging, json
from omegaconf import OmegaConf

def load_manifest_iter(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("data.manifest", help="JSONL (utt_id,path,distortion,language,sr,duration_sec)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    rows = []
    for rec in load_manifest_iter(getattr(args, "data.manifest")):
        rows.append({
            "utt_id": rec["utt_id"],
            "path": rec["path"],
            "distortion": rec.get("distortion","unknown"),
            "language": rec.get("language","unknown"),
            "sr": rec.get("sr",16000),
            "duration_sec": rec["duration_sec"],
            "loss_proxy": 0.0,
            "ssl_embed": [0.0, 0.0, 0.0],
        })
    pd.DataFrame(rows).to_parquet(args.out, index=False)
    logging.info(f"Wrote {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
```

```python
// FILE: trainers/bsrrn.py
import argparse, os, subprocess
from omegaconf import OmegaConf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--curated_list", required=True)
    ap.add_argument("--exp.name", dest="exp_name", required=False, default=None)
    args, unknown = ap.parse_known_args()

    cfg = OmegaConf.load(args.config)
    if args.exp_name:
        cfg.exp.name = args.exp_name
    exp_dir = os.path.join("experiments", cfg.exp.name)
    os.makedirs(os.path.join(exp_dir, "train", "checkpoints"), exist_ok=True)
    OmegaConf.save(cfg, os.path.join(exp_dir, "config.yaml"))

    cmd = [
        "python",
        os.path.join(cfg.paths.baseline_repo, "baseline_code/train_se.py"),
        "--config_file", cfg.train.bsrrn_config,
        "--curated_list", args.curated_list,
        "--exp_dir", os.path.join(exp_dir, "train"),
    ]
    cmd += unknown
    subprocess.check_call(cmd)

if __name__ == "__main__":
    main()
```

```python
// FILE: tools/namegen.py
from datetime import datetime
def build_exp_name(model:str, selector:str, K:int, alpha:float, seed:int)->str:
    ts = datetime.now().strftime("%y%m%d")
    return f"{ts}_{model}_{selector}_K{K}h_a{alpha:.2f}_s{seed}"
```

```python
// FILE: tools/manifest_utils.py
import pandas as pd
def filter_wav_scp(wav_scp_in:str, selection_csv:str, wav_scp_out:str):
    keep = set(pd.read_csv(selection_csv)["utt_id"].tolist())
    with open(wav_scp_in) as fi, open(wav_scp_out, "w") as fo:
        for line in fi:
            uid = line.split()[0]
            if uid in keep:
                fo.write(line)
```

```python
// FILE: datasets/cli.py
import argparse
from omegaconf import OmegaConf
from .registry import build_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_cfg", default="configs/data.yaml")
    ap.add_argument("--dataset_cfg", required=True)
    ap.add_argument("--stage", choices=["download","prepare","verify","manifest","all"], default="all")
    ap.add_argument("--split", choices=["train","dev","test","all"], default="all")
    args = ap.parse_args()

    dcfg = OmegaConf.load(args.dataset_cfg)
    gcfg = OmegaConf.load(args.data_cfg)
    ds = build_dataset(dcfg, gcfg)

    if args.stage in ("download","all"): ds.download()
    if args.stage in ("prepare","all"):  ds.prepare()
    if args.stage in ("verify","all"):   ds.verify()
    if args.stage in ("manifest","all"): ds.build_manifests(split=args.split)

if __name__ == "__main__":
    main()
```

```python
// FILE: datasets/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from omegaconf import DictConfig
import json, os

class BaseDataset(ABC):
    """Unified interface so training never changes when switching datasets."""

    def __init__(self, dataset_cfg: DictConfig, global_cfg: DictConfig):
        self.dcfg = dataset_cfg
        self.gcfg = global_cfg
        self.root = Path(os.getenv("DATA_ROOT", self.gcfg.data.root)).resolve()
        self.dirs = {
            "raw": Path(self.gcfg.data.dirs.raw.format(dataset=self.dcfg.dataset.name)).resolve(),
            "external": Path(self.gcfg.data.dirs.external.format(dataset=self.dcfg.dataset.name)).resolve(),
            "processed": Path(self.gcfg.data.dirs.processed.format(dataset=self.dcfg.dataset.name)).resolve(),
        }
        for p in self.dirs.values(): p.mkdir(parents=True, exist_ok=True)

    def download(self): pass
    def prepare(self): pass
    def verify(self): pass

    @abstractmethod
    def build_manifests(self, split: str = "all"): ...

    def _write_manifest(self, rows, split: str):
        out_map = self.gcfg.data.manifests
        out_path = Path(getattr(out_map, split))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return str(out_path)
```

```python
// FILE: datasets/registry.py
from typing import Dict, Type
from .base import BaseDataset
from .urgent2026 import Urgent2026Dataset

_REGISTRY: Dict[str, Type[BaseDataset]] = {
    "urgent2026": Urgent2026Dataset
}

def register_dataset(name: str):
    def deco(cls):
        if name in _REGISTRY: raise KeyError(f"{name} already registered")
        _REGISTRY[name] = cls
        return cls
    return deco

def build_dataset(dataset_cfg, global_cfg) -> BaseDataset:
    name = dataset_cfg.dataset.name
    cls = _REGISTRY.get(name)
    if not cls:
        raise KeyError(f"Unknown dataset '{name}'. Known: {list(_REGISTRY)}")
    return cls(dataset_cfg, global_cfg)
```

```python
// FILE: datasets/urgent2026.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Dict, Any
import subprocess, glob, soundfile as sf, os
from .base import BaseDataset

def _iter_wavs(pattern: str):
    for p in glob.glob(pattern, recursive=True):
        if p.lower().endswith(".wav"): yield p

def _infer_label(path: str, keys=("language","distortion")) -> Dict[str,str]:
    low = path.lower()
    lab = {}
    if "language" in keys:
        for lg in ("en","zh","ja","de","fr","es","it","ko","ru"):
            if f"/{lg}/" in low or f"_{lg}_" in low:
                lab["language"] = lg; break
    if "distortion" in keys:
        for d in ("reverb","noise","clip","codec","packetloss","bandlimit","device","clean"):
            if f"/{d}/" in low or f"_{d}_" in low:
                lab["distortion"] = d; break
    return lab

class Urgent2026Dataset(BaseDataset):
    """Adapter for URGENT Track-1 simulated data (pre-sim or run simulation)."""

    def download(self): pass  # organizers typically share packs; path is configured

    def prepare(self):
        sim = self.dcfg.dataset.simulation
        if bool(sim.use):
            repo = Path(sim.baseline_repo)
            out = Path(sim.out_dir); out.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            if "ffmpeg_path" in sim and sim.ffmpeg_path:
                env["FFMPEG_BIN"] = sim.ffmpeg_path
            subprocess.check_call([
                "python",
                str(Path(repo, "simulation/simulate_data_from_param.py")),
                "--param_json", str(Path(sim.params_json)),
                "--out_dir", str(out)
            ], env=env)

    def verify(self): pass

    def build_manifests(self, split: str = "all"):
        cfg = self.dcfg
        base = Path(cfg.dataset.pre_simulated.path if bool(cfg.dataset.pre_simulated.use)
                    else cfg.dataset.simulation.out_dir)
        globs = cfg.dataset.globs
        split_list = ["train","dev","test"] if split == "all" else [split]
        for sp in split_list:
            pat = getattr(globs, sp)
            rows = []
            for wav in _iter_wavs(pat):
                try:
                    info = sf.info(wav)
                    dur = float(info.duration)
                    sr = int(info.samplerate)
                except Exception:
                    continue
                row: Dict[str, Any] = {
                    "utt_id": Path(wav).stem,
                    "path": str(Path(wav).resolve()),
                    "duration_sec": dur,
                    "sr": sr,
                    "split": sp
                }
                inf = []
                if bool(cfg.dataset.infer.language_from_path): inf.append("language")
                if bool(cfg.dataset.infer.distortion_from_path): inf.append("distortion")
                row.update(_infer_label(wav, keys=tuple(inf)))
                rows.append(row)
            outp = self._write_manifest(rows, sp)
            print(f"[urgent2026] Wrote {len(rows)} rows → {outp}")
```

---

## Acceptance Criteria (definition of done)

* Repo tree exactly as in **Repository Layout** (stubs acceptable where marked TODO).
* `python -m datasets.cli ... --stage all` creates non-empty manifests when `.wav` files exist.
* `python -m scoring.score ...` writes a valid `scores.parquet` with **required columns**.
* `python -m selector ...` outputs a CSV with `utt_id` and a meta JSON; **cvar** works naively.
* `python -m trainers.bsrrn ...` creates `experiments/{exp_name}/train/checkpoints/` and saves `config.yaml`.
* `README.md` includes Quickstart, repo map, Add-a-selector guide, datasets instructions, reproducibility notes.
* `requirements.txt`, `pyproject.toml`, `.gitignore`, `Makefile`, and scripts under `scripts/` exist and run.
* Tests (can be smoke/TODO): CVaR monotonicity vs `K_hours`, seed reproducibility, quota feasibility (skip allowed with TODO).

---

## Batched Prompts (for GPT-5-Codex)

> Use the **Master Command**: **`Execute: BATCH N`**.
> Codex must output one `// FILE:` block per file, with full contents.

* **BATCH 0 — Datasets**: Create/implement dataset CLI, base, registry, urgent2026 adapter; add `configs/data.yaml`, `configs/datasets/*`.
* **BATCH 1 — Scaffold**: Create remaining tree, stubs, `requirements.txt`, `pyproject.toml`, `.gitignore`, `Makefile`, minimal `README.md`.
* **BATCH 2 — Selector plumbing**: Implement registry/base/CLI/CVAR + `configs/select/cvar.yaml`, `configs/quotas.yaml`.
* **BATCH 3 — Scoring**: Implement `scoring/score.py` and `configs/scoring.yaml`.
* **BATCH 4 — Trainers**: Implement `trainers/bsrrn.py` and ensure training configs.
* **BATCH 5 — Eval, scripts, tests**: Add `eval/` stubs, `scripts/run_*.sh`, basic tests.
* **BATCH 6 — README polish**: Replace `README.md` with full onboarding.

---

## Quickstart (for humans)

```bash
make setup
# 0) Datasets → manifests
python -m datasets.cli --dataset_cfg configs/datasets/urgent2026.yaml --data_cfg configs/data.yaml --stage all
# 1) Score
python -m scoring.score --config configs/scoring.yaml data.manifest=data/manifests/urgent2026/train.jsonl --out data/scores/train_scores.parquet
# 2) Select (CVaR)
python -m selector --config configs/select/cvar.yaml --scores data/scores/train_scores.parquet --quotas configs/quotas.yaml --out data/curated/curated_train_K=700h.csv
# 3) Train (BSRNN)
python -m trainers.bsrrn --config configs/base.yaml --curated_list data/curated/curated_train_K=700h.csv --exp.name 251007_bsrrn_cvar_K700h_a0.10_s7
# 4) Eval
python -m eval.eval --config configs/eval.yaml --ckpt experiments/251007_bsrrn_cvar_K700h_a0.10_s7/train/checkpoints/best.ckpt
```

---

## Non-Goals / Constraints

* **Do not edit** the official baseline submodule in `third_party/urgent2026_baseline`.
* No absolute paths; respect `DATA_ROOT` and YAML overrides.
* Keep CLIs stable; add new features/configs without breaking existing commands.

---

## Change Log

* **1.1 (2025-10-07):** Added “Execute: BATCH N” controller and strict `// FILE:` output-format rules.
* **1.0 (2025-10-07):** Initial end-to-end spec with datasets module, pluggable selector, training wrappers, batched prompts.

