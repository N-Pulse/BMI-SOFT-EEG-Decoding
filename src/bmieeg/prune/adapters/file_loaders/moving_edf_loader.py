# ================================================================
# 0. Section: IMPORTS
# ================================================================
import mne
import numpy as np

from pathlib import Path
from collections import defaultdict

from ...domain import DataActor
from .file_loader import FileLoader
from ...domain.actors import TrialData, TrialWindow
from .file_loader_registry import FileLoaderRegistry


# ================================================================
# 1. Section: INPUTS
# ================================================================
# The channels that are not EEG (sensors of the movement of the head)
SKIP_CHANNELS = ["X", "Y", "Z"]

# In the EDF files the triggers are saved as "Trigger#1" ... "Trigger#17"
EVENT_ID = {f"Trigger#{i}": i for i in range(1, 18)}

# The triggers of the three blocks of every gesture: (rest, imagined, executed)
GESTURES = {
    "open_close": (1, 3, 5),
    "wrist_rotation": (7, 9, 11),
    "finger_tapping": (13, 15, 17),
}

# Every block starts at its trigger and lasts 6 seconds, the cross lasts 2 seconds
BLOCK_SECONDS = 6.0
CROSS_SECONDS = 2.0

# MOVING has no raw markers: its own protocol version has an empty marker schema in the protocol
MOVING_PROTOCOL_VERSION = 3

# The people of MOVING get a number from 1001, so they never have the number of one of ours
SUBJECT_OFFSET = 1000


# ================================================================
# 2. Section: Class definition
# ================================================================
@FileLoaderRegistry.register(".edf")
class MovingEdfLoader(FileLoader):
    """MOVING: one EDF file per person, in volts.

    In MOVING a person does every gesture in three blocks of 6 seconds: rest, imagined movement
    and executed movement, with a cross of 2 seconds in between. The three blocks of one gesture
    are one trial. Gives one actor for the whole recording, with the list of its trials.
    """

    def load_actors(self, path: Path) -> list[DataActor]:
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        windows = _cut_subject(raw)

        raw.drop_channels(SKIP_CHANNELS)
        signal = raw.get_data() * 1e6

        return [
            TrialData(
                signal_ch=signal,
                markers=np.zeros((1, signal.shape[1]), dtype=np.int64),
                channel_names=list(raw.ch_names),
                sfreq=float(raw.info["sfreq"]),
                windows=windows,
                file_header={"source": "MOVING", "sfreq": float(raw.info["sfreq"])},
                file_name=path.stem,
                file_path=path,
                directory_structure="moving",
                protocol_version=MOVING_PROTOCOL_VERSION,
                signal_type="eeg",
                subject_id=SUBJECT_OFFSET + int(path.stem.split("_")[2]),
            )
        ]


# ──────────────────────────────────────────────────────
# 2.1 Subsection: Module Helpers
# ──────────────────────────────────────────────────────
def _cut_subject(raw: mne.io.BaseRaw) -> list[TrialWindow]:
    """List the trials of one person."""
    # 1. When every trigger happened, in seconds
    events, _ = mne.events_from_annotations(raw, event_id=EVENT_ID, verbose="ERROR")
    times: dict[int, list[float]] = defaultdict(list)
    for sample, _, number in events:
        times[int(number)].append(float(sample) / raw.info["sfreq"])

    # 2. The k-th rest, imagined and executed block of a gesture make the k-th trial.
    # The trial starts with the cross before the rest block and ends with the executed block.
    windows = []
    for movement, (rest, imagined, executed) in GESTURES.items():
        for k in range(min(len(times[rest]), len(times[imagined]), len(times[executed]))):
            start = times[rest][k] - CROSS_SECONDS
            end = times[executed][k] + BLOCK_SECONDS

            # 3. A trial that was cut short (the recording stopped) is left out
            if end > raw.times[-1]:
                continue
            windows.append(TrialWindow(
                start=start, end=end, movement=movement, repetition=k, orientation="none",
            ))
    return windows
