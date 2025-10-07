from __future__ import annotations

from typing import Dict, Type

from omegaconf import DictConfig

from .base import BaseDataset

_REGISTRY: Dict[str, Type[BaseDataset]] = {}


def register_dataset(name: str):
    def decorator(cls: Type[BaseDataset]) -> Type[BaseDataset]:
        if not issubclass(cls, BaseDataset):
            raise TypeError(f"Dataset '{name}' must inherit from BaseDataset")
        if name in _REGISTRY:
            raise KeyError(f"Dataset '{name}' already registered")
        _REGISTRY[name] = cls
        return cls

    return decorator


def _ensure_builtin(name: str) -> None:
    if name == "urgent2026" and "urgent2026" not in _REGISTRY:
        from .urgent2026 import Urgent2026Dataset  # local import to avoid cycles

        _REGISTRY["urgent2026"] = Urgent2026Dataset


def build_dataset(dataset_cfg: DictConfig, global_cfg: DictConfig) -> BaseDataset:
    name = dataset_cfg.dataset.name
    if name not in _REGISTRY:
        _ensure_builtin(name)
    try:
        dataset_cls = _REGISTRY[name]
    except KeyError as exc:  # pragma: no cover - defensive
        known = ", ".join(sorted(_REGISTRY)) or "<empty>"
        raise KeyError(f"Unknown dataset '{name}'. Known: {known}") from exc
    return dataset_cls(dataset_cfg, global_cfg)
