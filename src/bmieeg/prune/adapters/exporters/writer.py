# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path
from typing import Any, ClassVar, Self
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ...domain.actors import DataActor


# ================================================================
# 1. Section: Class definition
# ================================================================
@dataclass
class Writer[A: DataActor](ABC):
    name: ClassVar[str]
    actor: ClassVar[str]
    timestamp: str

    @classmethod
    def from_config(cls, data: dict[str, Any], timestamp: str) -> Self:
        return cls(timestamp=timestamp, **data)

    @abstractmethod
    def write(self, actor: A, dest_dir: Path) -> None:
        raise NotImplementedError
