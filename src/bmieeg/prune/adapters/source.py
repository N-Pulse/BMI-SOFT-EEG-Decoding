# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from pathlib import Path


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Source:
    data_dir: Path

    @property
    def dataset_dir(self) -> Path:
        return self.data_dir / "dataset"

    def raw_data_dir(self, dataset_path: Path) -> Path:
        return self.data_dir / dataset_path

    def out_dir(self, protocol_name: str) -> Path:
        return self.dataset_dir / protocol_name
