# ================================================================
# 0. Section: IMPORTS
# ================================================================
from tqdm import tqdm
from typing import Any
from pathlib import Path
from datetime import UTC, datetime
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..adapters import Source, Loader, Exporter, ProtocolLoader
from ..domain import PruneStep, DataActor


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class DatasetPruner:
    source_name: list[str]

    custom_steps: list[PruneStep] = field(default_factory=list)

    _data_dir: Path = Path("data/")
    _config_dir: Path = Path("configs/prune/")

    def __post_init__(self):
        self._source = Source(data_dir=self._data_dir)
        self._loader = Loader(source=self._source)
        self._exporter = Exporter()
        self._protocol_loader = ProtocolLoader(config_dir=self._config_dir)

    # ================================================================
    # 2. Section: MAIN FUNCTIONS
    # ================================================================
    def run(self, protocol_name: str | None) -> list[DataActor]:
        # 1. Load all the actors into the scene
        actors = self._loader.load(self.source_name)

        # 2. Load all the steps into the scene
        steps = self._load_steps(protocol_name)

        # 3. Apply the steps to the actors
        pruned_actors = self.apply(actors, steps)

        return pruned_actors

    def export(self, actors: list[DataActor], protocol_name: str) -> None:
        # One timestamp per run so a dataset's files share a suffix.
        timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d_%H-%M-%S")
        writers = self._protocol_loader.load_writers(protocol_name, timestamp)
        self._exporter.export(actors, writers, self._source.out_dir(protocol_name))

    def apply(
        self,
        actors: list[DataActor],
        steps: Sequence[PruneStep[Any, Any]],
    ) -> list[DataActor]:
        for step in tqdm(steps, total=len(steps), desc="Applying steps"):
            actors = step.apply(actors)

        return actors

    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def _load_steps(self, protocol_name: str | None) -> Sequence[PruneStep[Any, Any]]:
        if self.custom_steps:
            steps = self.custom_steps
        elif protocol_name is not None:
            steps = self._protocol_loader.load(protocol_name).steps
        else:
            raise ValueError("No protocol name provided and no custom steps defined")

        return steps
