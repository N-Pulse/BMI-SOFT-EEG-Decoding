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
@StepRegistry.register("adc_normalize")
@dataclass
class AdcNormalizeStep(PruneStep[TrialData, TrialData]):
    name: ClassVar[str] = "adc_normalize"
    bits: int

    def __post_init__(self) -> None:
        if not 2 <= self.bits <= 32:
            raise ValueError(f"bits must be between 2 and 32, got {self.bits}")

    def apply(self, actors: list[TrialData]) -> list[TrialData]:
        return [self._normalize_actor(actor) for actor in actors]

    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def _normalize_actor(self, actor: TrialData) -> TrialData:
        half_range = 1 << (self.bits - 1)
        signal = np.asarray(actor.signal_ch, dtype=np.float64)

        # Out-of-range counts mean the bits setting is wrong for this source.
        if signal.size and (signal.min() < -half_range or signal.max() >= half_range):
            raise ValueError(
                f"{actor.file_name}: counts [{signal.min()}, {signal.max()}] exceed "
                f"signed {self.bits}-bit limits [{-half_range}, {half_range - 1}]"
            )
        return actor.copy_with(signal_ch=signal / half_range)
