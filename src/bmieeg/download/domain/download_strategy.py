# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .download_spec import DownloadSpec



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class DownloadStrategy[SpecT: DownloadSpec](ABC):
    @abstractmethod
    def validate(self, spec: SpecT) -> None:
        raise NotImplementedError

    @abstractmethod
    def fetch(self, spec: SpecT) -> None:
        raise NotImplementedError
