# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from typing import ClassVar
from datetime import date

from ..actors import XdfData
from .prune_step import PruneStep
from .step_registry import StepRegistry


# ================================================================
# 1. Section: Functions
# ================================================================
@StepRegistry.register("protocol_definer")
@dataclass
class ProtocolDefinerStep(PruneStep[XdfData, XdfData]):
    name: ClassVar[str] = "protocol_definer"
    cuttoff_dates: list[date]

    def apply(self, actors: list[XdfData]) -> list[XdfData]:
        cutoffs = sorted(self.cuttoff_dates)

        for actor in actors:
            if actor.date is None:
                continue  # leave protocol_version at its default (undefined)
            actor.protocol_version = 1 + sum(
                1 for cutoff in cutoffs if actor.date >= cutoff
            )
        return actors
