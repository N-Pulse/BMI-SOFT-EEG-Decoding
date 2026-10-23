# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import ClassVar
from dataclasses import dataclass
from collections.abc import Callable

from .download_strategy import DownloadStrategy



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Registry:
    _strategies: ClassVar[dict[str, type[DownloadStrategy]]] = {}

    @classmethod
    def register(cls, source: str) -> Callable[[type[DownloadStrategy]], type[DownloadStrategy]]:
        def decorator(strategy_cls: type[DownloadStrategy]) -> type[DownloadStrategy]:
            if source in cls._strategies:
                raise ValueError(f"Source {source!r} already registered")
            cls._strategies[source] = strategy_cls
            return strategy_cls
        return decorator

    @classmethod
    def get(cls, source: str) -> DownloadStrategy:
        try:
            strategy_cls = cls._strategies[source]
        except KeyError:
            raise ValueError(
                f"Unknown source {source!r}. Registered: {sorted(cls._strategies)}"
            ) from None
        return strategy_cls()
