# ================================================================
# 0. Section: IMPORTS
# ================================================================
from abc import ABC, abstractmethod
from pathlib import Path

from ...domain import DataActor


# ================================================================
# 1. Section: Class definition
# ================================================================
class FileLoader(ABC):
    @abstractmethod
    def load_actors(self, path: Path) -> list[DataActor]:
        raise NotImplementedError
