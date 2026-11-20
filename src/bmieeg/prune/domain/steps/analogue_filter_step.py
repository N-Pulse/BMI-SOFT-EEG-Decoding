# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Self

from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt

from ..actors import TrialData
from .prune_step import PruneStep
from .step_registry import StepRegistry

FilterType = Literal["bandpass", "bandstop", "lowpass", "highpass"]


# ================================================================
# 1. Section: Functions
# ================================================================
@StepRegistry.register("analogue_filter")
@dataclass
class AnalogueFilterStep(PruneStep):
    name: ClassVar[str] = "analogue_filter"
    btype: FilterType
    cutoff: float | tuple[float, float]
    order: int = 1
    causal: bool = False

    @classmethod
    def from_config(cls, data: dict[str, Any]) -> Self:
        cutoff = data["cutoff"]

        if isinstance(cutoff, list):
            data = {**data, "cutoff": tuple(cutoff)}
        return cls(**data)

    def __post_init__(self) -> None:
        pair = isinstance(self.cutoff, tuple)
        if self.btype in ("bandpass", "bandstop") and not pair:
            raise ValueError(f"{self.btype} needs a (low, high) cutoff pair")
        if self.btype in ("lowpass", "highpass") and pair:
            raise ValueError(f"{self.btype} needs a single cutoff value")

    def apply(self, actors: list[TrialData]) -> list[TrialData]:
        return [self._filter_actor(actor) for actor in actors]

    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def _filter_actor(self, actor: TrialData) -> TrialData:
        cutoff = self._clamp_cutoff(actor.sfreq / 2)
        sos = butter(
            self.order,
            cutoff,
            btype=self.btype,
            fs=actor.sfreq,
            output="sos",
        )

        if self.causal:
            # Start as if the signal had always been at its first value: an electrode sits at an
            # offset of millivolts, a filter that starts from zero would ring for seconds
            start = sosfilt_zi(sos)[:, None, :] * actor.signal_ch[:, :1]
            filtered, _ = sosfilt(sos, actor.signal_ch, axis=1, zi=start)
        else:
            # Zero-phase filtering keeps the signal aligned with the markers.
            filtered = sosfiltfilt(sos, actor.signal_ch, axis=1)
        return actor.copy_with(signal_ch=filtered)

    def _clamp_cutoff(self, nyquist: float) -> float | tuple[float, float]:
        low = nyquist * 1e-6
        high = nyquist * (1 - 1e-6)

        def clamp(freq: float) -> float:
            return min(max(freq, low), high)

        cutoff = self.cutoff
        clamped = (
            (clamp(cutoff[0]), clamp(cutoff[1]))
            if isinstance(cutoff, tuple)
            else clamp(cutoff)
        )
        return clamped
