# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from typing import ClassVar
from dataclasses import dataclass, replace

from ..actors import XdfData
from .prune_step import PruneStep
from .step_registry import StepRegistry


# ================================================================
# 1. Section: Functions
# ================================================================
@StepRegistry.register("marker_deduplicate")
@dataclass
class MarkerDeduplicateStep(PruneStep[XdfData, XdfData]):
    """Keeps only the first of two markers that come less than `max_gap_seconds` apart.

    The newer sessions (protocol version 2, from 2025-11-21) write every marker twice. The second
    one is the same marker with a duration, and for return and iti it even carries the code of
    the movement. A signal that holds every marker until the next one would lose those phases.
    """

    name: ClassVar[str] = "marker_deduplicate"
    max_gap_seconds: float = 0.05

    def apply(self, actors: list[XdfData]) -> list[XdfData]:
        # Only XDF recordings have markers to clean; trials that are already cut (MOVING) pass through
        return [self._clean_actor(a) if isinstance(a, XdfData) else a for a in actors]

    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def _clean_actor(self, actor: XdfData) -> XdfData:
        return replace(actor, streams=[self._clean_stream(s) for s in actor.streams])

    def _clean_stream(self, stream: dict) -> dict:
        info_type = stream.get("info", {}).get("type")
        if not info_type or str(info_type[0]).lower() != "markers":
            return stream

        times = np.asarray(stream["time_stamps"], dtype=float)
        order = np.argsort(times, kind="stable")
        keep = np.ones(order.size, dtype=bool)
        keep[1:] = np.diff(times[order]) >= self.max_gap_seconds
        kept = order[keep]

        cleaned = dict(stream)
        cleaned["time_stamps"] = times[kept]
        cleaned["time_series"] = [stream["time_series"][i] for i in kept]
        return cleaned
