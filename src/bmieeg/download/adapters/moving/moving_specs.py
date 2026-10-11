# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import ClassVar
from dataclasses import dataclass

from ...domain import DownloadSpec


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class MOVINGSpecs(DownloadSpec):
    source: ClassVar[str] = "moving"

    record_id: int = 12804784
    filename: str | None = None