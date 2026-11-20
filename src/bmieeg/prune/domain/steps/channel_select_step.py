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
@StepRegistry.register("channel_select")
@dataclass
class ChannelSelectStep(PruneStep[TrialData, TrialData]):
    name: ClassVar[str] = "channel_select"
    channels: list[str]
    # A channel that moves less than this (microvolts) is flat: the electrode has no contact
    flat_microvolts: float = 1e-6

    def apply(self, actors: list[TrialData]) -> list[TrialData]:
        return [self._select_actor(actor) for actor in actors]

    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def _select_actor(self, actor: TrialData) -> TrialData:
        # The recordings write the names in capital letters ("CZ"), the board writes "Cz"
        position = {name.upper(): i for i, name in enumerate(self.channels)}

        kept = [
            (position[name.upper()], row)
            for row, name in enumerate(actor.channel_names)
            if name.upper() in position and np.ptp(actor.signal_ch[row]) > self.flat_microvolts
        ]
        if not kept:
            raise ValueError(f"{actor.file_name}: none of the channels {self.channels} has a signal")

        # In the order of the list, with the names of the list
        kept.sort()
        rows = [row for _, row in kept]
        names = [self.channels[place] for place, _ in kept]
        return actor.copy_with(signal_ch=actor.signal_ch[rows], channel_names=names)
