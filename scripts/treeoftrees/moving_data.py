"""
Reads the MOVING EEG recordings (one EDF file per person) and cuts them into
1-second windows, each labeled with what the person was doing.

A window is a table of numbers: one row per EEG channel, one column per time
point. The labels are: noGesture, open_close, wrist_rotation and finger_tapping.
"""
# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path

import mne
import numpy as np


# ================================================================
# 1. Section: INPUTS
# ================================================================
# Paths
ROOT: Path = Path(__file__).resolve().parents[2]
DATA: Path = ROOT / "data" / "raw" / "moving" / "edf"

# In the EDF files the triggers are saved as "Trigger#1" ... "Trigger#17"
EVENT_ID: dict[str, int] = {f"Trigger#{i}": i for i in range(1, 18)}

# Triggers of the 6-second periods: rest and executed movements.
TRIGGER_TO_CLASS: dict[int,str] = {
    1: "noGesture", 7: "noGesture", 13: "noGesture",
    5: "open_close",
    11: "wrist_rotation",
    17: "finger_tapping",
}

# Windows
PERIOD_SECONDS: int = 6
WINDOW_SECONDS: int = 1

# Seconds skipped at the start of a period, the person does not react instantly
SKIP_SECONDS: dict[str, int] = {
    "noGesture": 2,
    "open_close": 1,
    "wrist_rotation": 1,
    "finger_tapping": 1,
}

# Band-pass filter (Hz)
LOW_HZ: float = 1.0
HIGH_HZ: float = 45.0


# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def load_subject(path: Path) -> mne.io.BaseRaw:
    """Read one EDF file, filter it and subtract the average of the channels."""
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")

    # 1. Keep the 32 EEG channels (X, Y and Z are head-movement sensors)
    raw.drop_channels(["X", "Y", "Z"])

    # 2. Find the channels without signal: an electrode without contact never changes
    spread = np.ptp(raw.get_data(), axis=1)
    flat = [name for name, value in zip(raw.ch_names, spread) if value == 0]

    # 3. Remove slow drifts and fast noise before cutting, to avoid edge effects
    raw.filter(LOW_HZ, HIGH_HZ, verbose="ERROR")

    # 4. Subtract the average of the working channels from every channel
    raw.info["bads"] = flat
    raw.set_eeg_reference("average", verbose="ERROR")

    return raw


def cut_windows(raw: mne.io.BaseRaw) -> tuple[np.ndarray, np.ndarray]:
    """Cut a recording into labeled windows.

    Returns the windows (windows x channels x time points, in microvolts) and
    their labels.
    """
    # 1. Find where every trigger happened (sample number and trigger number)
    events, _ = mne.events_from_annotations(raw, event_id=EVENT_ID, verbose="ERROR")
    data = raw.get_data() * 1e6  # MNE stores volts, microvolts are easier to read
    sfreq = raw.info["sfreq"]
    size = int(WINDOW_SECONDS * sfreq)

    # 2. Cut the windows that follow each rest or executed movement trigger
    windows, labels = [], []
    for sample, _, number in events:
        name = TRIGGER_TO_CLASS.get(number)
        if name is None:
            continue

        skip = SKIP_SECONDS[name]
        for k in range((PERIOD_SECONDS - skip) // WINDOW_SECONDS):
            start = sample + int((skip + k * WINDOW_SECONDS) * sfreq)
            if start + size <= data.shape[1]:
                windows.append(data[:, start : start + size])
                labels.append(name)

    return np.stack(windows), np.array(labels)

# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    # Prints how many windows of each class every person gives
    classes = sorted(set(TRIGGER_TO_CLASS.values()))
    for path in sorted(DATA.glob("*.edf")):
        # 1. Load and cut the recording
        subject = path.stem.split("_")[2]
        windows, labels = cut_windows(load_subject(path))

        # 2. Count the windows of each class
        counts = {name: int((labels == name).sum()) for name in classes}
        print(f"Subject {subject}", windows.shape, counts)
