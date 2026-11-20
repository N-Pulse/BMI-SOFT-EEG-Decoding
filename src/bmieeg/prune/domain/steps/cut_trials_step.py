# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from dataclasses import dataclass
from typing import ClassVar

from ..actors import TrialData
from .prune_step import PruneStep
from .step_registry import StepRegistry


# ================================================================
# 1. Section: Functions
# ================================================================
@StepRegistry.register("cut_trials")
@dataclass
class CutTrialsStep(PruneStep[TrialData, TrialData]):
    """Cuts every trial out of a whole recording, after the filters have worked on it.

    The markers of a trial hold the code of its movement all the way through.
    """

    name: ClassVar[str] = "cut_trials"
    movement_codes: dict[str, int]  # name of the movement -> code from 1 to 99

    def __post_init__(self) -> None:
        codes = list(self.movement_codes.values())
        if any(not 1 <= code <= 99 for code in codes):
            raise ValueError("the code of a movement must be between 1 and 99")
        if len(set(codes)) != len(codes):
            raise ValueError("two movements have the same code")

    def apply(self, actors: list[TrialData]) -> list[TrialData]:
        return [trial for actor in actors for trial in self._cut_actor(actor)]

    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def _cut_actor(self, actor: TrialData) -> list[TrialData]:
        n_samples = actor.signal_ch.shape[1]
        trials = []
        for window in actor.windows:
            if window.movement not in self.movement_codes:
                raise ValueError(f"{actor.file_name}: no code for the movement {window.movement!r}")

            first, last = round(window.start * actor.sfreq), round(window.end * actor.sfreq)
            if first < 0 or last > n_samples:
                raise ValueError(f"{actor.file_name}: the trial {window} is outside of the recording")

            code = self.movement_codes[window.movement]
            trials.append(actor.copy_with(
                signal_ch=actor.signal_ch[:, first:last],
                markers=np.full((1, last - first), code, dtype=np.int64),
                windows=[window],
                file_name=f"{actor.file_name}_{window.movement}_{window.repetition:02d}",
            ))
        return trials
