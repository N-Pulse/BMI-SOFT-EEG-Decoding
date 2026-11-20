# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import ClassVar
from numpy.typing import NDArray
from dataclasses import dataclass

from .prune_step import PruneStep
from .step_registry import StepRegistry
from ..actors import DataActor, Dataset, DatasetPartition, TrialData


# ================================================================
# 1. Section: Functions
# ================================================================
@StepRegistry.register("build_epochs")
@dataclass
class BuildEpochs(PruneStep[DataActor, DataActor]):
    name: ClassVar[str] = "build_epochs"
    # Window around each movement onset, in seconds (mne.Epochs convention).
    # Defaults match the legacy SignalPartitioner / build_dataset.py.
    tmin: float = -0.5
    tmax: float = 5.0

    def __post_init__(self) -> None:
        if self.tmax <= self.tmin:
            raise ValueError(
                f"tmax must be greater than tmin, got tmin={self.tmin}, tmax={self.tmax}"
            )

    def apply(self, actors: list[DataActor]) -> list[DataActor]:
        segmented: list[DataActor] = []
        for actor in actors:
            if isinstance(actor, Dataset):
                segmented.append(self._segment_dataset(actor))
            elif isinstance(actor, TrialData):
                segmented.extend(self._segment_trial(actor))
            else:
                raise TypeError(
                    f"build_epochs expects Dataset or TrialData, got "
                    f"{type(actor).__name__}"
                )
        return segmented

    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def _segment_dataset(self, dataset: Dataset) -> Dataset:
        return dataset.copy_with(
            client=[self._segment_partition(p) for p in dataset.client],
            server=[self._segment_partition(p) for p in dataset.server],
        )

    def _segment_partition(self, partition: DatasetPartition) -> DatasetPartition:
        trials = [
            seg for trial in partition.trials for seg in self._segment_trial(trial)
        ]
        return partition.copy_with(trials=trials)

    def _segment_trial(self, trial: TrialData) -> list[TrialData]:
        # Movements live in the last two digits of the canonical XXDD code.
        movements = trial.markers[0] % 100
        n = movements.size

        lo_offset = round(self.tmin * trial.sfreq)
        hi_offset = round(self.tmax * trial.sfreq)

        segments: list[TrialData] = []
        for onset, _run_end in _movement_runs(movements):
            lo = onset + lo_offset
            hi = onset + hi_offset
            # Drop windows that run off either edge, like mne.Epochs does.
            if lo < 0 or hi > n:
                continue
            segments.append(
                trial.copy_with(
                    signal_ch=trial.signal_ch[:, lo:hi],
                    markers=trial.markers[:, lo:hi],
                )
            )
        return segments


# ──────────────────────────────────────────────────────
# 1.2 Subsection: Module Helpers
# ──────────────────────────────────────────────────────
def _movement_runs(movements: NDArray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    n = movements.size

    start = 0
    while start < n:
        value = movements[start]
        end = start + 1
        while end < n and movements[end] == value:
            end += 1
        if value != 0:
            runs.append((start, end))
        start = end

    return runs
