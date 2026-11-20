# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from typing import ClassVar

from .data_actor import DataActor
from .dataset_partition import DatasetPartition


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Dataset(DataActor):
    name: ClassVar[str] = "dataset"
    client: list[DatasetPartition]
    server: list[DatasetPartition]
