"""Load one XDF recording and make labeled EMG windows for offline evaluation."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bmiemg.data.convert import session_load
from bmiemg.data.epoch import TriggerMap, V1_TRIGGER_MAP, V2_TRIGGER_MAP


CHANNELS = ("FC1", "FC2", "FC3", "FC4", "FCZ", "C1", "C2", "C3", "C4", "CZ",
            "CP1", "CP2", "CP3", "CP4")


@dataclass(frozen=True)
class Trial:
    number: int
    code: int
    label: str
    prep_start: float
    start: float
    end: float


@dataclass(frozen=True)
class BatchInfo:
    time: float
    trial: int
    phase: str
    truth: str 
    clean: bool


# ================================================================
# 1. Map an XDF movement code to its gesture label
# ================================================================
def phase_digit(trigger_map: TriggerMap, name: str) -> str:
    return next(str(k) for k, v in trigger_map.phase_code.code_dict.items() if v == name)
def detect_trigger_map(marker_stream) -> TriggerMap:
    markers = [str(value[0]).strip() for value in marker_stream.time_series]
    leading = {m[0] for m in markers if len(m) == 5 and m.isdigit()}
    # V1 uses phases 1-5 and V2 uses 1,3,5,7,9, so the leading digits identify the map
    for trigger_map in (V2_TRIGGER_MAP, V1_TRIGGER_MAP):
        if leading <= {str(code) for code in trigger_map.phase_code.code_dict}:
            return trigger_map
    raise ValueError(f"Marker phases {sorted(leading)} match neither V1 nor V2")

# Search the existing trigger map for the code and raise an error if the code has no gesture label.
def movement_label(code: int, trigger_map: TriggerMap) -> str:
    for label, codes in trigger_map.target_code.items():
        if code in codes:
            return label
    raise ValueError(f"Movement code {code} is not mapped by this trigger map")


# ================================================================
# 2. Rebuild trials from preparation, movement, and return markers
# ================================================================
# Read the XDF marker stream in chronological order.
# Match markers prep(3), move(5) and return(7) for each movement.
# Store the gesture and its preparation and movement times.
def extract_trials(marker_stream, trigger_map: TriggerMap) -> list[Trial]:
    prep_digit = phase_digit(trigger_map, "prep")
    move_digit = phase_digit(trigger_map, "move")
    return_digit = phase_digit(trigger_map, "return")

    trials = []
    prepared = None
    moving = None

    for timestamp, value in zip(marker_stream.time_stamps, marker_stream.time_series):
        marker = str(value[0]).strip()
        if len(marker) != 5 or not marker.isdigit():
            continue
        phase, identity = marker[0], marker[1:]

        if phase == prep_digit and moving is None:
            prepared = (float(timestamp), identity)
        elif phase == move_digit and prepared is not None and prepared[1] == identity and moving is None:
            moving = (float(timestamp), prepared[0], identity)
            prepared = None
        elif phase == return_digit and moving is not None:
            start, prep_start, current_identity = moving
            if identity != current_identity:
                raise ValueError(f"Return marker {marker} does not match movement")
            code = int(identity[-2:])
            trials.append(
                Trial(len(trials) + 1, code, movement_label(code, trigger_map), prep_start,
                      start, float(timestamp))
            )
            moving = None

    if moving is not None or not trials:
        raise ValueError("Incomplete or missing movement trials in XDF markers")
    return trials


# ================================================================
# 3. Split each code's trials into training and testing
# ================================================================
# Group trials by movement code.
# Use the first train_per_code trials for training.
# Put later trials in the held-out test set.
def split_trials(trials: list[Trial], train_per_code: int) -> tuple[set[int], set[int]]:
    by_code = defaultdict(list)
    for trial in trials:
        by_code[trial.code].append(trial.number)

    train, test = set(), set()
    for code, numbers in by_code.items():
        if len(numbers) <= train_per_code:
            raise ValueError(
                f"Code {code:02d} has {len(numbers)} trials; need more than "
                f"{train_per_code} for a held-out test"
            )
        train.update(numbers[:train_per_code])
        test.update(numbers[train_per_code:])
    return train, test


# ================================================================
# 4. Find each trial's end and the following rest period
# ================================================================
# Find marker ITI after each movement's return marker.
# Label rest from marker ITI until the next preparation marker.
# Leave the final rest period unlabeled because its end is unknown.
def trial_bounds(marker_stream, trials: list[Trial], trigger_map: TriggerMap) -> dict:
    iti_digit = phase_digit(trigger_map, "iti")
    events = [
        (float(time), str(value[0]).strip())
        for time, value in zip(marker_stream.time_stamps, marker_stream.time_series)
    ]
    bounds = {}
    for index, trial in enumerate(trials):
        next_prep = trials[index + 1].prep_start if index + 1 < len(trials) else None
        iti = next(
            (
                time
                for time, marker in events
                if time > trial.end
                and (next_prep is None or time < next_prep)
                and len(marker) == 5
                and marker[0] == iti_digit
                and marker.isdigit()
                and int(marker[-2:]) == trial.code
            ),
            None,
        )
        if iti is None:
            raise ValueError(f"Missing ITI marker after trial {trial.number}")
        # The last ITI has no known end, so it is not labeled as rest.
        bounds[trial.number] = (next_prep or iti, (iti, next_prep) if next_prep else None)
    return bounds


# ================================================================
# 5. Load the XDF and convert selected EMG channels to volts
# ================================================================
# Load the recording once and extract its trials and rest bounds.
# Keep the five model channels in their expected order.
# Return EMG, timestamps, sampling rate, trials, and bounds.
def load_recording(path: Path):
    session = session_load(path)
    trigger_map = detect_trigger_map(session.marker_stream)
    trials = extract_trials(session.marker_stream, trigger_map)
    bounds = trial_bounds(session.marker_stream, trials, trigger_map)
    stream = session.signal_stream
    names = list(stream.channel_names)
    missing = set(CHANNELS) - set(names)
    if missing:
        raise ValueError(f"Missing EMG channels: {sorted(missing)}")
    indices = [names.index(name) for name in CHANNELS]
    # Match SignalStream.to_raw(): recorded microvolts -> volts.
    emg_volts = stream.time_series[:, indices] / 2 * 1e-6
    return emg_volts, stream.time_stamps, float(stream.sfreq), trials, bounds


# ================================================================
# 6. Make fixed-size batches with or without overlap
# ================================================================
# Convert window and step from milliseconds to samples; a full-window step means no overlap.
# Label each batch by the phase covering most of it (prep = noGesture, return = gesture).
# Mark batches fully inside movement or rest as clean.
def make_batches(
    emg_volts: np.ndarray,
    timestamps: np.ndarray,
    sfreq: float,
    trials: list[Trial],
    bounds: dict,
    selected_trials: set[int],
    window_ms: int,
    step_ms: int,
) -> tuple[np.ndarray, list[BatchInfo]]:
    window_samples = round(sfreq * window_ms / 1000)
    step_samples = round(sfreq * step_ms / 1000)
    if window_samples < 3 or not 0 < step_samples <= window_samples:
        raise ValueError("Use a positive step no larger than the window")
    if not np.isclose(window_samples / sfreq * 1000, window_ms, atol=0.5):
        raise ValueError("Window duration cannot be represented at this sample rate")
    if not np.isclose(step_samples / sfreq * 1000, step_ms, atol=0.5):
        raise ValueError("Step duration cannot be represented at this sample rate")

    batches, infos = [], []
    for trial in trials:
        if trial.number not in selected_trials:
            continue
        trial_end, rest = bounds[trial.number]
        iti_start = rest[0] if rest else trial_end
        spans = [
            ("prep", trial.prep_start, trial.start, "noGesture"),
            ("movement", trial.start, trial.end, trial.label),
            ("return", trial.end, iti_start, trial.label),
            ]
        if rest:
            spans.append(("rest", rest[0], rest[1], "noGesture"))
        
        first = int(np.searchsorted(timestamps, trial.prep_start, side="left"))
        last = int(np.searchsorted(timestamps, trial_end, side="left"))
        for offset in range(first, last - window_samples + 1, step_samples):
            batch_start = float(timestamps[offset])
            batch_end = float(timestamps[offset + window_samples - 1] + 1 / sfreq)
            overlaps = [min(batch_end, end) - max(batch_start, start)
                        for _, start, end, _ in spans]
            phase, _, _, truth = spans[int(np.argmax(overlaps))]
            clean = (batch_start >= trial.start and batch_end <= trial.end) or bool(
                rest and batch_start >= rest[0] and batch_end <= rest[1]
            )
            batches.append(emg_volts[offset : offset + window_samples].T)
            infos.append(BatchInfo(batch_start, trial.number, phase, truth, clean))

    if not batches:
        raise ValueError("No complete batches were found")
    return np.stack(batches), infos
