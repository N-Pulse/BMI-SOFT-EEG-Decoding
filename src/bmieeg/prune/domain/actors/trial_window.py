# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass


# ================================================================
# 1. Section: Class definition
# ================================================================
@dataclass(frozen=True)
class TrialWindow:
    """One trial of a recording: one repetition of one movement."""

    start: float  # seconds from the start of the recording
    end: float
    movement: str  # "openhand_slow_3sec"
    repetition: int  # 0, 1, 2... of this movement in this recording
    orientation: str  # "Up", "Side" or "none"
