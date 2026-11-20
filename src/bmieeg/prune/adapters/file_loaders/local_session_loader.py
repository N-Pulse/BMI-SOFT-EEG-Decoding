# ================================================================
# 0. Section: IMPORTS
# ================================================================
import csv
import statistics

import mne
import numpy as np

from pathlib import Path
from collections import Counter

from ...domain import DataActor
from .file_loader import FileLoader
from ...domain.actors import TrialData, TrialWindow
from .file_loader_registry import FileLoaderRegistry


# ================================================================
# 1. Section: INPUTS
# ================================================================
# The channels that are not EEG
SKIP_CHANNELS = ["M1", "M2", "EOG", "TRIGGER"]

# The four parts of a trial, in the order they happen
PHASES = ("prep", "move", "return", "iti")

# A marker that lasts less than this (seconds) is only a point in time
INSTANT_SECONDS = 0.05

# Test recordings have 1 to 3 trials, a real session has many more
MIN_TRIALS = 12


# ================================================================
# 2. Section: Class definition
# ================================================================
@FileLoaderRegistry.register(".fif")
class LocalSessionLoader(FileLoader):
    """Our own recordings: `<date>/<sub>/<ses>/eeg/<name>_eeg.fif` with its `_events.tsv` next to it.

    Gives one actor for the whole recording, with the list of its trials. The numbers in the
    file are already microvolts, even if MNE thinks they are volts.
    """

    def load_actors(self, path: Path) -> list[DataActor]:
        events_path = path.with_name(path.name.replace("_eeg.fif", "_events.tsv"))
        if not path.name.endswith("_eeg.fif") or not events_path.exists():
            return []

        windows = _cut_session(events_path)
        if len(windows) < MIN_TRIALS:
            print(f"skipped {path.name}: {len(windows)} trials, a test recording")
            return []

        raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
        raw.drop_channels(SKIP_CHANNELS, on_missing="ignore")
        signal = raw.get_data()

        return [
            TrialData(
                signal_ch=signal,
                markers=np.zeros((1, signal.shape[1]), dtype=np.int64),
                channel_names=list(raw.ch_names),
                sfreq=float(raw.info["sfreq"]),
                windows=windows,
                file_header={"source": "local", "sfreq": float(raw.info["sfreq"])},
                file_name=path.stem,
                file_path=path,
                signal_type="eeg",
            )
        ]


# ──────────────────────────────────────────────────────
# 2.1 Subsection: Module Helpers
# ──────────────────────────────────────────────────────
def _read_events(events_path: Path) -> list[dict]:
    with open(events_path, newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def _find_markers(rows: list[dict]) -> list[dict]:
    """Keep one row for every marker of the four phases, in time order.

    Newer sessions (for example 2025-11-27) write every marker twice: a row of 0.01 s
    and a row with a duration. The durations of return and iti are even written into
    rows called "move". In those sessions only the 0.01 s rows are right.
    """
    markers = [row for row in rows if row["trial_type"] in PHASES]
    doubled = any(
        row["trial_type"] == "prep" and float(row["duration"]) < INSTANT_SECONDS
        for row in markers
    )
    if doubled:
        markers = [row for row in markers if float(row["duration"]) < INSTANT_SECONDS]
    return sorted(markers, key=lambda row: float(row["onset"]))


def _typical_iti(markers: list[dict]) -> float:
    """How long the iti normally lasts (seconds).

    The iti before a break between two blocks lasts longer, so the middle value of
    all the iti is used. Markers that are only a point in time last until the next marker.
    """
    lengths = []
    for before, after in zip(markers, markers[1:]):
        if before["trial_type"] != "iti":
            continue
        length = float(before["duration"])
        if length < INSTANT_SECONDS:
            length = float(after["onset"]) - float(before["onset"])
        lengths.append(length)
    return statistics.median(lengths) if lengths else 0.0


def _cut_session(events_path: Path) -> list[TrialWindow]:
    """List the trials of one session: from the prep signal to the end of a normal iti."""
    markers = _find_markers(_read_events(events_path))
    iti_seconds = _typical_iti(markers)

    # 1. Every prep starts a trial, the markers up to the next prep belong to it
    preps = [i for i, row in enumerate(markers) if row["trial_type"] == "prep"]
    windows, repetitions = [], Counter()
    for n, first in enumerate(preps):
        last = preps[n + 1] if n + 1 < len(preps) else len(markers)
        phases: dict[str, dict] = {}
        for row in markers[first:last]:
            phases.setdefault(row["trial_type"], row)

        # 2. A trial that was cut short (the recording stopped) is left out
        if any(name not in phases for name in PHASES):
            continue

        # 3. The trial ends when a normal iti ends
        movement = phases["prep"]["movement_label"]
        windows.append(TrialWindow(
            start=float(phases["prep"]["onset"]),
            end=float(phases["iti"]["onset"]) + iti_seconds,
            movement=movement,
            repetition=repetitions[movement],
            orientation=phases["prep"]["baseline_label"],
        ))
        repetitions[movement] += 1
    return windows
