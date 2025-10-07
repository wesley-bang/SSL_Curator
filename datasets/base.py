from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Mapping
import json
import os

from omegaconf import DictConfig, OmegaConf


class BaseDataset(ABC):
    """Common utilities for dataset adapters."""

    def __init__(self, dataset_cfg: DictConfig, global_cfg: DictConfig) -> None:
        self.dcfg = dataset_cfg
        self.gcfg = global_cfg

        data_cfg = OmegaConf.to_container(self.gcfg.data, resolve=True)
        dataset_name = self.dcfg.dataset.name

        root_value = data_cfg.get("root", "data")
        self.root = Path(os.getenv("DATA_ROOT", str(root_value))).resolve()

        dirs_cfg: Mapping[str, str] = data_cfg.get("dirs", {})
        self.dirs = {
            key: self._resolve_path(template, dataset_name)
            for key, template in dirs_cfg.items()
        }
        for directory in self.dirs.values():
            directory.mkdir(parents=True, exist_ok=True)

    def download(self) -> None:  # pragma: no cover - to be overridden
        pass

    def prepare(self) -> None:  # pragma: no cover - to be overridden
        pass

    def verify(self) -> None:  # pragma: no cover - to be overridden
        pass

    @abstractmethod
    def build_manifests(self, split: str = "all") -> None:
        """Create manifest files for the requested split(s)."""

    def _resolve_path(self, template: str, dataset_name: str) -> Path:
        value = str(template)
        replacements = {
            "${data.root}": str(self.root),
            "${data.dataset}": dataset_name,
            "${dataset}": dataset_name,
        }
        for token, replacement in replacements.items():
            value = value.replace(token, replacement)
        try:
            value = value.format(dataset=dataset_name)
        except KeyError:
            pass
        return Path(value).resolve()

    def _write_manifest(self, rows: Iterable[Mapping[str, object]], split: str) -> str:
        manifests_cfg = OmegaConf.to_container(self.gcfg.data.manifests, resolve=True)
        target = manifests_cfg[split]
        out_path = Path(target)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return str(out_path)
