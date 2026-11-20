# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from typing import ClassVar

from ..actors import TrialData
from .prune_step import PruneStep
from .step_registry import StepRegistry


# ================================================================
# 1. Section: Functions
# ================================================================
@StepRegistry.register("to_full_scale")
@dataclass
class ToFullScaleStep(PruneStep[TrialData, TrialData]):
    """Writes the signal (microvolts) as a part of the full scale of the converter, from -1 to 1.

    `adc_quantize` needs this scale. The number of bits says how many counts one unit is worth.
    """

    name: ClassVar[str] = "to_full_scale"
    full_scale_microvolts: float  # the biggest signal the converter can show, plus or minus

    def __post_init__(self) -> None:
        if self.full_scale_microvolts <= 0:
            raise ValueError(f"full_scale_microvolts must be positive, got {self.full_scale_microvolts}")

    def apply(self, actors: list[TrialData]) -> list[TrialData]:
        return [
            actor.copy_with(signal_ch=actor.signal_ch / self.full_scale_microvolts)
            for actor in actors
        ]
