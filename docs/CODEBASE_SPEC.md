# URGENT 2026 – Curator Codebase Spec
**Spec-Version:** 1.2 (2025-10-07)

<!-- SOURCE_OF_TRUTH_ANCHOR_V1_2 -->
> **AI implementers (GPT-5-Codex in VS Code):**  
> This file is the **source of truth**. Always re-open and parse this file before making changes.  
> - **Do not** invent paths or rename files.  
> - All parameters must come from YAML (no magic numbers).  
> - Keep CLIs stable; if you improve internals, **preserve public interfaces**.  
> - If something is underspecified, emit a TODO stub **without changing the public API**.

## How to invoke me (simple, robust)
- Keep this file **open** in VS Code when you prompt.
- To build any step, say: **`Execute: BATCH <N><LETTER>`**, e.g., `Execute: BATCH 0A`.  
  You can also run full batches (`BATCH 0`, `BATCH 1`, …), or the entire pipeline (`BATCH 0-6`), but small sub-batches are safer.
- If output would exceed limits, **stop** and reply only with:  
  `READY FOR CONTINUE: BATCH <N><LETTER> PART <k>`  
  I will reply with `CONTINUE: BATCH <N><LETTER> PART <k>`.

## OUTPUT FORMAT (must follow exactly)
When you cannot auto-apply edits to the workspace, output **one file per fenced block**:

```

// FILE: path/to/file.ext <full file content with no omissions>

```

Rules: one file per block; full content (no ellipses); exact paths from this spec; **overwrite** existing files.

## Does smaller sub-batches hurt performance?
No—breaking tasks into smaller, labeled sub-batches usually **improves reliability** (less truncation, fewer hallucinations). To maintain global coherence, **each sub-batch starts with a “What & Why”** summary reminding you how outputs are used later. If context feels lost, **re-sync** to this spec (anchor `SOURCE_OF_TRUTH_ANCHOR_V1_2`).

---

## Executive Summary (≤200 words)
We’re building a clean, extensible pipeline for **Universal Speech Enhancement (URGENT 2026 – Track-1)** where the contribution is **data curation, not brute-scale**. The pipeline is:

**score → select → train → eval**  
1) **Scoring** computes per-utterance proxies (MOS, SSL embeddings, SDR/eSTOI proxies).  
2) **Selection** chooses a subset via pluggable strategies (CVaR, coverage-only, MOS-threshold) with **coverage quotas** and **diversity**.  
3) **Training** wraps official baselines (BSRNN/FlowSE) to train on the curated subset.  
4) **Evaluation** runs official metrics and tail/slice analyses.

Everything is **config-driven**, **reproducible**, and easy to extend: add a selector by dropping a file + decorator; switch datasets via a unified **datasets** CLI that standardizes manifests, so training doesn’t change.

---

## North-Star Principles
- **One folder = one purpose** (`scoring/`, `selector/`, `datasets/`, `trainers/`, `eval/`, `tools/`).
- **Config > code** (paths, quotas, α, K hours, dataset choice).
- **Pluggable selection** via registry and base class.
- **Experiment hygiene** (frozen config, curated CSV, logs, metrics, checkpoints).
- **Flexibility knobs** (English-only, 16 kHz-only, cap hours) in `configs/data.yaml`.

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
│  ├─ data.yaml                 # global dataset routing + filters
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
│  ├─ mos_backends/
│  ├─ ssl_backends/
│  ├─ proxies/
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

---

## Dataflow (who consumes what)
1) **datasets** → writes **manifests** (`data/manifests/<dataset>/{train,dev,test}.jsonl`).  
   → Input for **scoring**.

2) **scoring** → reads manifest → writes `data/scores/train_scores.parquet`.  
   → Input for **selector**.

3) **selector** → reads `scores.parquet` (+ `quotas.yaml`) → writes `data/curated/curated_train_*.csv` (+ meta).  
   → Input for **trainers**.

4) **trainers** → train baseline on curated set → write artifacts under `experiments/{exp_name}`.  
   → Input for **eval**.

---

## Public Schemas (authoritative)

### Manifest JSONL row (from `datasets/`)
```json
{"utt_id":"unique","path":"/abs/path.wav","duration_sec":3.21,"sr":16000,"language":"en","distortion":"noise","split":"train"}
````

### `scores.parquet` columns (from `scoring/score.py`)

Required: `utt_id:str`, `path:str`, `duration_sec:float`, `sr:int`, `language:str?`, `distortion:str?`, `loss_proxy:float`, `ssl_embed:list[float]`.
Optional: `mos_dns`, `mos_nisqa`, `uncert`, `asr_consistency`, …

### `selection.csv` (from `selector/`)

CSV with at least `utt_id`; sidecar `${out}_meta.json` with config snapshot, `scores_path`, `n_selected`, input hashes.

---

## CLI Contracts (verbatim)

**Datasets**

```bash
python -m datasets.cli \
  --dataset_cfg configs/datasets/urgent2026.yaml \
  --data_cfg configs/data.yaml \
  --stage all \
  --split all
# stages: download | prepare | verify | manifest | all
# split:  train | dev | test | all
```

**Scoring**

```bash
python -m scoring.score \
  --config configs/scoring.yaml \
  data.manifest=data/manifests/urgent2026/train.jsonl \
  --out data/scores/train_scores.parquet
```

**Selection**

```bash
python -m selector \
  --config configs/select/cvar.yaml \
  --scores data/scores/train_scores.parquet \
  --quotas configs/quotas.yaml \
  --out data/curated/curated_train_K=700h.csv
```

**Training**

```bash
python -m trainers.bsrrn \
  --config configs/base.yaml \
  --curated_list data/curated/curated_train_K=700h.csv \
  --exp.name 251007_bsrrn_cvar_K700h_a0.10_s7
```

**Evaluation**

```bash
python -m eval.eval \
  --config configs/eval.yaml \
  --ckpt experiments/251007_bsrrn_cvar_K700h_a0.10_s7/train/checkpoints/best.ckpt
```

---

## Configuration (key files & flexibility knobs)

### `configs/data.yaml`

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

  # Global filters for quick, focused runs (honored by scoring & selection)
  filters:
    languages: ["en"]          # [] or null = all
    sample_rates: [16000]      # e.g., focus on 16k
    distortions: []            # e.g., ["noise","reverb"]; empty = all
    max_hours: null            # cap size for fast experiments
```

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

## Critical Code Snippets (minimum working — keep CLIs stable)

```python
// FILE: selector/registry.py
from typing import Dict, Type
_REGISTRY: Dict[str, Type] = {}
def register_selector(name: str):
    def deco(cls):
        if name in _REGISTRY: raise KeyError(f"Selector '{name}' already registered")
        _REGISTRY[name] = cls
        return cls
    return deco
def build(name: str, **kwargs):
    try: cls = _REGISTRY[name]
    except KeyError as e: raise KeyError(f"Unknown selector '{name}'. Known: {list(_REGISTRY)}") from e
    return cls(**kwargs)
```

```python
// FILE: selector/base.py
from abc import ABC, abstractmethod
import pandas as pd
class BaseSelector(ABC):
    def __init__(self, **kwargs): self.kwargs = kwargs
    @abstractmethod
    def select(self, df_scores: pd.DataFrame) -> pd.DataFrame: ...
```

```python
// FILE: selector/__main__.py
import argparse, json, pandas as pd, logging
from omegaconf import OmegaConf
from .registry import build
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--quotas")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    if args.quotas: cfg.select.quotas = args.quotas
    df = pd.read_parquet(args.scores)
    selector = build(cfg.select.name, **OmegaConf.to_container(cfg.select, resolve=True))
    df_sel = selector.select(df)
    df_sel[["utt_id"]].to_csv(args.out, index=False)
    meta = {"config": OmegaConf.to_container(cfg, resolve=True), "scores_path": args.scores, "n_selected": int(df_sel.shape[0])}
    with open(args.out.replace(".csv","_meta.json"), "w") as f: json.dump(meta, f, indent=2)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO); main()
```

```python
// FILE: selector/cvar.py
import numpy as np, pandas as pd
from .base import BaseSelector
from .registry import register_selector
@register_selector("cvar")
class CvarSelector(BaseSelector):
    def __init__(self, K_hours: float, alpha: float, quotas=None, diversity=None, penalties=None, **_):
        super().__init__(); self.K_hours=float(K_hours); self.alpha=float(alpha)
        self.quotas=quotas; self.diversity=diversity or {"min_cosine":0.15}; self.penalties=penalties or {"quota_violation":10.0}
    def select(self, df: pd.DataFrame) -> pd.DataFrame:
        df=df.copy()
        for col,key in (("language","languages"),("sr","sample_rates"),("distortion","distortions")):
            flt=self.kwargs.get("filters",{}).get(key); 
            if flt: df=df[df[col].isin(flt)]
        df["hours"]=df["duration_sec"]/3600.0
        df=df.sort_values("loss_proxy", ascending=False)
        picked, hours=[], 0.0
        def ok_div(row,pool):
            if not pool: return True
            v=np.asarray(row["ssl_embed"],dtype=float)
            for p in pool[-50:]:
                w=np.asarray(p["ssl_embed"],dtype=float)
                cos=float(np.dot(v,w)/(np.linalg.norm(v)*np.linalg.norm(w)+1e-8))
                if cos>1-self.diversity["min_cosine"]: return False
            return True
        for _,r in df.iterrows():
            if hours>=self.K_hours: break
            if ok_div(r,picked):
                picked.append(r); hours+=r["hours"]
        return pd.DataFrame(picked)
```

```python
// FILE: scoring/score.py
import argparse, pandas as pd, logging, json
from omegaconf import OmegaConf
def load_manifest_iter(path):
    with open(path,"r",encoding="utf-8") as f:
        for line in f: yield json.loads(line)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("data.manifest")
    ap.add_argument("--out", required=True)
    args=ap.parse_args()
    _=OmegaConf.load(args.config)  # reserved
    rows=[]
    for rec in load_manifest_iter(getattr(args,"data.manifest")):
        rows.append({"utt_id":rec["utt_id"],"path":rec["path"],"distortion":rec.get("distortion","unknown"),
                     "language":rec.get("language","unknown"),"sr":rec.get("sr",16000),
                     "duration_sec":rec["duration_sec"],"loss_proxy":0.0,"ssl_embed":[0.0,0.0,0.0]})
    pd.DataFrame(rows).to_parquet(args.out,index=False)
    logging.info(f"Wrote {len(rows)} rows to {args.out}")
if __name__=="__main__": logging.basicConfig(level=logging.INFO); main()
```

```python
// FILE: trainers/bsrrn.py
import argparse, os, subprocess
from omegaconf import OmegaConf
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--curated_list", required=True)
    ap.add_argument("--exp.name", dest="exp_name", default=None)
    args, unknown=ap.parse_known_args()
    cfg=OmegaConf.load(args.config)
    if args.exp_name: cfg.exp.name=args.exp_name
    exp_dir=os.path.join("experiments", cfg.exp.name)
    os.makedirs(os.path.join(exp_dir,"train","checkpoints"), exist_ok=True)
    OmegaConf.save(cfg, os.path.join(exp_dir,"config.yaml"))
    cmd=["python", os.path.join(cfg.paths.baseline_repo,"baseline_code/train_se.py"),
         "--config_file", cfg.train.bsrrn_config, "--curated_list", args.curated_list,
         "--exp_dir", os.path.join(exp_dir,"train")]
    cmd+=unknown; subprocess.check_call(cmd)
if __name__=="__main__": main()
```

```python
// FILE: datasets/cli.py
import argparse
from omegaconf import OmegaConf
from .registry import build_dataset
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_cfg", default="configs/data.yaml")
    ap.add_argument("--dataset_cfg", required=True)
    ap.add_argument("--stage", choices=["download","prepare","verify","manifest","all"], default="all")
    ap.add_argument("--split", choices=["train","dev","test","all"], default="all")
    args=ap.parse_args()
    dcfg=OmegaConf.load(args.dataset_cfg); gcfg=OmegaConf.load(args.data_cfg)
    ds=build_dataset(dcfg,gcfg)
    if args.stage in ("download","all"): ds.download()
    if args.stage in ("prepare","all"):  ds.prepare()
    if args.stage in ("verify","all"):   ds.verify()
    if args.stage in ("manifest","all"): ds.build_manifests(split=args.split)
if __name__=="__main__": main()
```

```python
// FILE: datasets/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from omegaconf import DictConfig
import json, os
class BaseDataset(ABC):
    def __init__(self, dataset_cfg: DictConfig, global_cfg: DictConfig):
        self.dcfg=dataset_cfg; self.gcfg=global_cfg
        self.root=Path(os.getenv("DATA_ROOT", self.gcfg.data.root)).resolve()
        self.dirs={"raw":Path(self.gcfg.data.dirs.raw.format(dataset=self.dcfg.dataset.name)).resolve(),
                   "external":Path(self.gcfg.data.dirs.external.format(dataset=self.dcfg.dataset.name)).resolve(),
                   "processed":Path(self.gcfg.data.dirs.processed.format(dataset=self.dcfg.dataset.name)).resolve()}
        for p in self.dirs.values(): p.mkdir(parents=True, exist_ok=True)
    def download(self): pass
    def prepare(self): pass
    def verify(self): pass
    @abstractmethod
    def build_manifests(self, split: str = "all"): ...
    def _write_manifest(self, rows, split: str):
        out_map=self.gcfg.data.manifests; out_path=Path(getattr(out_map, split))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w",encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")
        return str(out_path)
```

```python
// FILE: datasets/registry.py
from typing import Dict, Type
from .base import BaseDataset
from .urgent2026 import Urgent2026Dataset
_REGISTRY: Dict[str, Type[BaseDataset]]={"urgent2026": Urgent2026Dataset}
def register_dataset(name: str):
    def deco(cls):
        if name in _REGISTRY: raise KeyError(f"{name} already registered")
        _REGISTRY[name]=cls; return cls
    return deco
def build_dataset(dataset_cfg, global_cfg)->BaseDataset:
    name=dataset_cfg.dataset.name; cls=_REGISTRY.get(name)
    if not cls: raise KeyError(f"Unknown dataset '{name}'. Known: {list(_REGISTRY)}")
    return cls(dataset_cfg, global_cfg)
```

```python
// FILE: datasets/urgent2026.py
from pathlib import Path
from typing import Iterable, Dict, Any
import subprocess, glob, soundfile as sf, os
from .base import BaseDataset
def _iter_wavs(pattern: str):
    for p in glob.glob(pattern, recursive=True):
        if p.lower().endswith(".wav"): yield p
def _infer_label(path: str, keys=("language","distortion"))->Dict[str,str]:
    low=path.lower(); lab={}
    if "language" in keys:
        for lg in ("en","zh","ja","de","fr","es","it","ko","ru"):
            if f"/{lg}/" in low or f"_{lg}_" in low: lab["language"]=lg; break
    if "distortion" in keys:
        for d in ("reverb","noise","clip","codec","packetloss","bandlimit","device","clean"):
            if f"/{d}/" in low or f"_{d}_" in low: lab["distortion"]=d; break
    return lab
class Urgent2026Dataset(BaseDataset):
    def download(self): pass  # path configured in YAML
    def prepare(self):
        sim=self.dcfg.dataset.simulation
        if bool(sim.use):
            repo=Path(sim.baseline_repo); out=Path(sim.out_dir); out.mkdir(parents=True, exist_ok=True)
            env=os.environ.copy()
            if "ffmpeg_path" in sim and sim.ffmpeg_path: env["FFMPEG_BIN"]=sim.ffmpeg_path
            subprocess.check_call(["python", str(Path(repo,"simulation/simulate_data_from_param.py")),
                                   "--param_json", str(Path(sim.params_json)), "--out_dir", str(out)], env=env)
    def verify(self): pass
    def build_manifests(self, split: str="all"):
        cfg=self.dcfg; globs=cfg.dataset.globs
        split_list=["train","dev","test"] if split=="all" else [split]
        for sp in split_list:
            pat=getattr(globs, sp); rows=[]
            for wav in _iter_wavs(pat):
                try: info=sf.info(wav); dur=float(info.duration); sr=int(info.samplerate)
                except Exception: continue
                row={"utt_id":Path(wav).stem,"path":str(Path(wav).resolve()),
                     "duration_sec":dur,"sr":sr,"split":sp}
                inf=[]; 
                if bool(cfg.dataset.infer.language_from_path): inf.append("language")
                if bool(cfg.dataset.infer.distortion_from_path): inf.append("distortion")
                row.update(_infer_label(wav, keys=tuple(inf))); rows.append(row)
            outp=self._write_manifest(rows, sp); print(f"[urgent2026] Wrote {len(rows)} rows → {outp}")
```

---

## Requirements, Makefile, Ignore (runtime + dev)

**`requirements.txt`**

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

**`Makefile`**

```make
.PHONY: setup lint fmt test run_all
setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt
	git submodule update --init --recursive
lint:  ; ruff check .
fmt:   ; black .
test:  ; pytest -q
run_all: ; bash scripts/run_all.sh
```

**`.gitignore`**

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

## Acceptance Criteria (definition of done)

* Tree matches **Repository Layout** (stubs OK where TODO).
* `datasets.cli` creates non-empty manifests when WAVs exist.
* `scoring.score` writes a valid `scores.parquet` with required columns.
* `selector` outputs `selection.csv` + `_meta.json` using `cvar` stub.
* `trainers.bsrrn` creates `experiments/{exp}/train/checkpoints` and writes `config.yaml`.
* `README.md` has Quickstart, repo map, datasets usage, add-a-selector guide, reproducibility.
* `requirements.txt`, `pyproject.toml`, `.gitignore`, `Makefile`, and `scripts/` exist and run.
* Tests (smoke/TODO): CVaR monotonicity vs `K_hours`, seed reproducibility, quota feasibility (can be skip with TODO).

---

# Batched Prompts (smaller, with “What & Why”)

> **Use the Master Command:** `Execute: BATCH <N><LETTER>`
> If too long, reply: `READY FOR CONTINUE: BATCH <N><LETTER> PART <k>`.

### BATCH 0A — Datasets: folders & cards

**What & Why:** Create the **datasets scaffold** so future steps have a home. Cards help humans find sources/licensing.
**Do:** Create `datasets/` tree (incl. `cards/` + `README.md`) and empty `__init__.py`.
**End check:** folders exist; `datasets/cards/README.md` explains card purpose.

### BATCH 0B — Dataset configs

**What & Why:** Config drives where data lives and how to build manifests.
**Do:** Add `configs/data.yaml`, `configs/datasets/urgent2026.yaml`, `configs/datasets/TEMPLATE.yaml` with fields shown above.
**End check:** YAMLs load (no syntax errors), contain keys `data.manifests`, `dataset.pre_simulated`, `dataset.simulation`, `globs`, `infer`.

### BATCH 0C — Base & Registry

**What & Why:** Standard interface so training never changes when switching datasets.
**Do:** Implement `datasets/base.py`, `datasets/registry.py` (exact CLIs).
**End check:** `python -c "import datasets; from datasets.registry import build_dataset"` succeeds.

### BATCH 0D — URGENT2026 adapter

**What & Why:** Concrete adapter that knows how to simulate or use pre-sim packs and infer labels from paths.
**Do:** Implement `datasets/urgent2026.py` as above.
**End check:** `build_dataset` returns `Urgent2026Dataset`.

### BATCH 0E — Datasets utils

**What & Why:** Room for downloader/extractor/checksums; keep lightweight now.
**Do:** Create `datasets/utils.py` with `sha256_file`, `download`, `extract` stubs or minimal implementations.
**End check:** Import works.

### BATCH 0F — Datasets CLI

**What & Why:** Single entrypoint for **download → prepare → verify → manifest**.
**Do:** Implement `datasets/cli.py` as above.
**End check:** Running `python -m datasets.cli --help` shows stages; manifests path in config is honored.

### BATCH 0G — Manifests smoke test (optional)

**What & Why:** Validate globs produce JSONL even if small.
**Do:** With WAVs present, run `--stage manifest`; verify JSONLs under `data/manifests/urgent2026/`.

---

### BATCH 1A — Root scaffold

**What & Why:** Provide repo hygiene & dev tooling.
**Do:** Add `requirements.txt`, `pyproject.toml`, `.gitignore`, `Makefile`, minimal `README.md`.
**End check:** `pip install -r requirements.txt` and `make fmt && make lint` pass.

### BATCH 1B — Package inits & stubs

**What & Why:** Prepare folders for upcoming code.
**Do:** Add `__init__.py` stubs and empty module files per layout (`scoring/`, `selector/`, `trainers/`, `eval/`, `tools/`).
**End check:** `pytest -q` runs (even if no tests yet).

### BATCH 1C — Scripts stubs

**What & Why:** One-click commands improve usability.
**Do:** Add `scripts/get_data.sh`, `prepare_manifests.sh` with CLI calls (no-op if not runnable).
**End check:** `bash scripts/get_data.sh` prints usage or runs.

### BATCH 1D — Tests stubs

**What & Why:** Ensure we can hook CI later.
**Do:** Add placeholder tests with TODO/xfail.
**End check:** `pytest -q` passes (ok with xfail/skip).

---

### BATCH 2A — Selector registry & base

**What & Why:** Pluggable selection strategies need a registry + ABC.
**Do:** Implement `selector/registry.py`, `selector/base.py`.
**End check:** Import and `register_selector` works.

### BATCH 2B — Selector CLI dispatcher

**What & Why:** Users pick strategies via config; CLI wires it.
**Do:** Implement `selector/__main__.py`.
**End check:** `python -m selector --help` shows args.

### BATCH 2C — CVaR (naïve) strategy

**What & Why:** Provide a working baseline focusing tail difficulty + diversity gate.
**Do:** Implement `selector/cvar.py` minimal version above.
**End check:** It accepts DataFrame with required columns and returns selected subset.

### BATCH 2D — Selector configs

**What & Why:** Strategy knobs live in YAML.
**Do:** Add `configs/select/cvar.yaml` and ensure `configs/quotas.yaml` exists.
**End check:** YAML loads; keys match CLI expectations.

### BATCH 2E — Quick dry-run

**What & Why:** Sanity before full scoring.
**Do:** Create a tiny fake parquet with required columns and run the selector CLI.
**End check:** Produces `selection.csv`.

---

### BATCH 3A — Scoring stub

**What & Why:** Enable end-to-end wiring **now**, even before MOS/SSL are real.
**Do:** Implement `scoring/score.py` that reads manifest JSONL → writes required columns.
**End check:** Parquet exists with correct schema.

### BATCH 3B — Scoring config

**What & Why:** Batch sizes, device, and output path in YAML.
**Do:** Add `configs/scoring.yaml`.
**End check:** YAML loads; tool runs with `--config`.

### BATCH 3C — Score real manifests

**What & Why:** Produce inputs for selector.
**Do:** Run scoring on `data/manifests/.../train.jsonl`.
**End check:** `data/scores/train_scores.parquet` exists.

---

### BATCH 4A — Trainer wrapper (BSRNN)

**What & Why:** Centralize training orchestration (paths, checkpoints, logs).
**Do:** Implement `trainers/bsrrn.py` (above).
**End check:** Command builds subprocess call and writes `experiments/<exp>/config.yaml`.

### BATCH 4B — Training configs

**What & Why:** Control model/backbone, epochs, etc.
**Do:** Add/adjust `configs/base.yaml` and `configs/train/bsrrn.yaml` fields referenced by wrapper.
**End check:** Running the wrapper creates `experiments/<exp>/train/checkpoints` (even if baseline is a dry run or fails later).

---

### BATCH 5A — Eval stubs

**What & Why:** Close the loop; provide a place for official metrics.
**Do:** Add `eval/eval.py` and `eval/tail_metrics.py` stubs.
**End check:** CLI runs with `--help`.

### BATCH 5B — Pipeline scripts

**What & Why:** End-to-end convenience.
**Do:** Add `scripts/run_score.sh`, `run_select.sh`, `run_train.sh`, `run_eval.sh`, `run_all.sh`.
**End check:** Scripts are executable and echo the right commands.

### BATCH 5C — Minimal tests

**What & Why:** Lock basic behavior.
**Do:** Add tests for CVaR monotonicity, reproducibility, quota feasibility (skip allowed).
**End check:** `pytest -q` runs.

---

### BATCH 6 — README polish

**What & Why:** Humans need a 60-second onboarding.
**Do:** Replace `README.md` with Quickstart, repo map, datasets usage, add-a-selector guide, reproducibility.
**End check:** README contains commands that match CLIs; copy-paste works.

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
* No absolute paths (use `DATA_ROOT` and YAML).
* Keep CLIs stable; new features should be additive.

---

## Change Log

* **1.2 (2025-10-07):** Added fine-grained sub-batches with “What & Why”, resilient CONTINUE protocol, re-sync anchor v1.2.
* **1.1 (2025-10-07):** Added “Execute: BATCH N” controller and strict `// FILE:` output rules.
* **1.0 (2025-10-07):** Initial end-to-end spec with datasets module, pluggable selector, training wrappers, batched prompts.
