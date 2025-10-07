"""Selector registry utilities."""

from typing import Any, Dict, Type

from .base import BaseSelector

_REGISTRY: Dict[str, Type[BaseSelector]] = {}


def register_selector(name: str):
    """Decorator used by selector implementations to register themselves."""

    def deco(cls: Type[BaseSelector]) -> Type[BaseSelector]:
        if not issubclass(cls, BaseSelector):
            raise TypeError(f"Selector '{name}' must inherit from BaseSelector")
        if name in _REGISTRY:
            raise KeyError(f"Selector '{name}' already registered")
        _REGISTRY[name] = cls
        return cls

    return deco


def build(name: str, **kwargs: Any) -> BaseSelector:
    """Instantiate a selector by name with the provided kwargs."""
    try:
        cls = _REGISTRY[name]
    except KeyError as error:
        known = ", ".join(sorted(_REGISTRY)) or "<empty>"
        raise KeyError(f"Unknown selector '{name}'. Known: {known}") from error
    return cls(**kwargs)


__all__ = ["register_selector", "build"]
