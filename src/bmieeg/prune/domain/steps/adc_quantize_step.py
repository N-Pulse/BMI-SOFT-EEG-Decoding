# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from typing import ClassVar
from dataclasses import dataclass

from ..actors import TrialData
from .prune_step import PruneStep
from .step_registry import StepRegistry


# ================================================================
# 1. Section: Functions
# ================================================================
@StepRegistry.register("adc_quantize")
@dataclass
class AdcQuantizeStep(PruneStep[TrialData, TrialData]):
    name: ClassVar[str] = "adc_quantize"
    bits: int

    def __post_init__(self) -> None:
        if not 2 <= self.bits <= 32:
            raise ValueError(f"bits must be between 2 and 32, got {self.bits}")

    def apply(self, actors: list[TrialData]) -> list[TrialData]:
        return [self._quantize_actor(actor) for actor in actors]

    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def _quantize_actor(self, actor: TrialData) -> TrialData:
        half_range = 1 << (self.bits - 1)
        dtype = np.int16 if self.bits <= 16 else np.int32

        # Saturate like the ADC does instead of wrapping around.
        counts = np.clip(
            np.rint(actor.signal_ch * half_range), -half_range, half_range - 1
        )
        return actor.copy_with(signal_ch=counts.astype(dtype))
