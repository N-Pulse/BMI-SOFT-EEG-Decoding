# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import ClassVar
from numpy.typing import NDArray
from dataclasses import dataclass, field

from .metadata_actor import MetadataActor
from .trial_window import TrialWindow


# ================================================================
# 1. Section: Class definition
# ================================================================
@dataclass(kw_only=True)
class TrialData(MetadataActor):
    name: ClassVar[str] = "trial_data"
    signal_ch: NDArray  # shape (channels, time)
    markers: NDArray  # shape (1, time)
    channel_names: list[str]
    sfreq: float
    # The trials of a whole recording, in seconds. `cut_trials` cuts them out.
    windows: list[TrialWindow] = field(default_factory=list)
