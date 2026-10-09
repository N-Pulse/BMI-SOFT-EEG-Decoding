"""Load one MOVING recording and make labeled EEG windows for offline evaluation."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

from scripts.treeoftrees.moving_data import (
    EVENT_ID,
    PERIOD_SECONDS,
    SKIP_SECONDS,
    TRIGGER_TO_CLASS,
    load_subject,
)

# Seconds of the fixation cross that is shown before every period
CROSS_SECONDS = 2


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
# 1. Rebuild trials from the MOVING triggers
# ================================================================
# A trial is one executed movement (6 s) and the fixation cross before it.
# Skip the first second of the movement: the person does not react instantly.
# Drop a movement that is cut by the end of the recording.
def extract_trials(events: np.ndarray, sfreq: float, duration: float) -> list[Trial]:
    trials = []
    for sample, _, number in events:
        label = TRIGGER_TO_CLASS.get(number)
        onset = sample / sfreq
        if label in (None, "noGesture") or onset + PERIOD_SECONDS > duration:
            continue
        trials.append(
            Trial(len(trials) + 1, int(number), label, onset - CROSS_SECONDS,
                  onset + SKIP_SECONDS[label], onset + PERIOD_SECONDS)
        )

    if not trials:
        raise ValueError("No complete executed movement found in the triggers")
    return trials


# ================================================================
# 2. Split each code's trials into training and testing
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
# 3. Find each trial's end and the following rest period
# ================================================================
# After a movement come the fixation cross and a 6 s rest period.
# Label everything from the end of the movement to the end of that rest as rest.
def trial_bounds(trials: list[Trial]) -> dict:
    bounds = {}
    for trial in trials:
        rest_end = trial.end + CROSS_SECONDS + PERIOD_SECONDS
        bounds[trial.number] = (rest_end, (trial.end, rest_end))
    return bounds


# ================================================================
# 4. Load the EDF and convert the EEG to microvolts
# ================================================================
# Load the recording once: band-pass filtered, without the channels that have no signal.
# Extract its trials and rest bounds.
# Return EEG, timestamps, sampling rate, trials, and bounds.
def load_recording(path: Path):
    raw = load_subject(path)
    raw.drop_channels(raw.info["bads"])
    events, _ = mne.events_from_annotations(raw, event_id=EVENT_ID, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    trials = extract_trials(events, sfreq, raw.times[-1])
    # Microvolts, one row per time point and one column per channel.
    return raw.get_data().T * 1e6, raw.times, sfreq, trials, trial_bounds(trials)


# ================================================================
# 5. Make fixed-size batches with or without overlap
# ================================================================
# Convert window and step from milliseconds to samples; a full-window step means no overlap.
# Label each batch by the phase covering most of it (prep = noGesture, return = gesture).
# Mark batches fully inside movement or rest as clean.
def make_batches(
    signal: np.ndarray,
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
            batches.append(signal[offset : offset + window_samples].T)
            infos.append(BatchInfo(batch_start, trial.number, phase, truth, clean))

    if not batches:
        raise ValueError("No complete batches were found")
    return np.stack(batches), infos
