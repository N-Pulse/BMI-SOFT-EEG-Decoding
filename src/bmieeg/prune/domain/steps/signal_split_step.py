# ================================================================
# 0. Section: IMPORTS
# ================================================================
import copy
from dataclasses import dataclass, replace
from typing import Any, ClassVar, get_args

import numpy as np

from ..actors import XdfData, Signal
from .prune_step import PruneStep
from .step_registry import StepRegistry

_ACCEPTED_TYPES: dict[Signal, set[str]] = {
    "eeg": {"eeg"},
    "emg": {"aux", "emg"},
}


# ================================================================
# 2. Section: Functions
# ================================================================
@StepRegistry.register("signal_split")
@dataclass
class SignalSplitStep(PruneStep[XdfData, XdfData]):
    name: ClassVar[str] = "signal_split"
    signal: Signal

    def __post_init__(self) -> None:
        if self.signal not in get_args(Signal):
            raise ValueError(
                f"signal must be one of {get_args(Signal)}, got {self.signal!r}"
            )

    def apply(self, actors: list[XdfData]) -> list[XdfData]:
        # Only XDF recordings are split; trials that are already cut (MOVING) pass through
        return [self._split_actor(a) if isinstance(a, XdfData) else a for a in actors]

    # ──────────────────────────────────────────────────────
    # 2.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def _split_actor(self, actor: XdfData) -> XdfData:
        accepted = _ACCEPTED_TYPES[self.signal]

        kept_streams: list[dict] = []
        for stream in actor.streams:
            # Markers carry the task events and are kept for every signal.
            if _stream_type(stream) == "markers":
                kept_streams.append(stream)
                continue

            new_stream = self._keep_signal(stream, accepted)
            if new_stream is not None:
                kept_streams.append(new_stream)

        return replace(actor, streams=kept_streams, signal_type=self.signal)

    def _keep_signal(self, stream: dict, accepted: set[str]) -> dict | None:
        channels = _channels(stream)
        if channels is None:
            # No per-channel metadata: decide at the stream level.
            return stream if _stream_type(stream) in accepted else None

        keep_idx = [
            i for i, chan in enumerate(channels) if _channel_type(chan) in accepted
        ]
        if not keep_idx:
            return None

        return _select_channels(stream, keep_idx, channels, self.signal)


# ──────────────────────────────────────────────────────
# 2.2 Subsection: Module Helpers
# ──────────────────────────────────────────────────────
def _select_channels(
    stream: dict, keep_idx: list[int], channels: list[dict], signal: Signal
) -> dict:
    new_stream = dict(stream)
    new_stream["time_series"] = np.asarray(stream["time_series"])[:, keep_idx]

    info = copy.deepcopy(stream["info"])
    info["type"] = [signal.upper()]
    info["channel_count"] = [str(len(keep_idx))]
    info["desc"][0]["channels"][0]["channel"] = [channels[i] for i in keep_idx]
    new_stream["info"] = info

    return new_stream


def _channels(stream: dict) -> list[dict] | None:
    desc = stream.get("info", {}).get("desc")
    if not desc or not desc[0]:
        return None
    try:
        return desc[0]["channels"][0]["channel"]
    except (KeyError, IndexError, TypeError):
        return None


def _channel_type(channel: dict) -> str | None:
    return _first_lower(channel.get("type"))


def _stream_type(stream: dict) -> str | None:
    return _first_lower(stream.get("info", {}).get("type"))


def _first_lower(value: Any) -> str | None:
    if not value:
        return None
    return str(value[0]).lower()
