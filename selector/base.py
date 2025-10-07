"""Base selector abstract class."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd


class BaseSelector(ABC):
    """Defines the interface selectors must implement."""

    def __init__(self, **kwargs: Any) -> None:
        # Store aux configuration for use by subclasses.
        self.kwargs: Dict[str, Any] = kwargs

    @abstractmethod
    def select(self, df_scores: pd.DataFrame) -> pd.DataFrame:
        """Return the curated subset from the input scores DataFrame."""
        raise NotImplementedError


__all__ = ["BaseSelector"]
