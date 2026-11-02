"""
Lists the trials of the MOVING recordings.

In MOVING a person does every gesture in three blocks of 6 seconds: rest, imagined
movement and executed movement. A cross of 2 seconds separates the blocks. We call
the three blocks of one gesture one trial, so a trial is the MOVING version of one
repetition of a movement in our own recordings.
"""
# ================================================================
# 0. Section: IMPORTS
# ================================================================
from collections import Counter, defaultdict
from pathlib import Path

import mne

from scripts.ttv.local_trials import Trial
from scripts.treeoftrees.moving_data import DATA, EVENT_ID


# ================================================================
# 1. Section: INPUTS
# ================================================================
# The triggers of the three blocks of every gesture: (rest, imagined, executed)
GESTURES: dict[str, tuple[int, int, int]] = {
    "open_close": (1, 3, 5),
    "wrist_rotation": (7, 9, 11),
    "finger_tapping": (13, 15, 17),
}

# Every block starts at its trigger and lasts 6 seconds, the cross lasts 2 seconds
BLOCK_SECONDS: float = 6.0
CROSS_SECONDS: float = 2.0


# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def cut_subject(path: Path) -> list[Trial]:
    """List the trials of one person."""
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    user = f"moving-{path.stem.split('_')[2]}"

    # 1. When every trigger happened, in seconds
    events, _ = mne.events_from_annotations(raw, event_id=EVENT_ID, verbose="ERROR")
    times: dict[int, list[float]] = defaultdict(list)
    for sample, _, number in events:
        times[int(number)].append(float(sample) / raw.info["sfreq"])

    # 2. The k-th rest, imagined and executed block of a gesture make the k-th trial.
    # The trial starts with the cross before the rest block and ends with the executed block.
    trials = []
    for movement, (rest, imagined, executed) in GESTURES.items():
        for k in range(min(len(times[rest]), len(times[imagined]), len(times[executed]))):
            start = times[rest][k] - CROSS_SECONDS
            end = times[executed][k] + BLOCK_SECONDS

            # 3. A trial that was cut short (the recording stopped) is left out
            if end > raw.times[-1]:
                continue
            trials.append(Trial(
                user=user,
                session="moving",
                orientation="none",
                movement=movement,
                repetition=k,
                start=start,
                end=end,
                folder=path,
            ))
    return trials


def load_trials(data: Path = DATA) -> list[Trial]:
    """List the trials of all the people."""
    trials = []
    for path in sorted(data.glob("*.edf")):
        trials += cut_subject(path)
    return trials


# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    trials = load_trials()
    movements = sorted({trial.movement for trial in trials})
    print(f"{'user':<12}{'trials':>7}" + "".join(f"{name:>16}" for name in movements))
    for user in sorted({trial.user for trial in trials}):
        counts = Counter(trial.movement for trial in trials if trial.user == user)
        print(f"{user:<12}{sum(counts.values()):>7}" + "".join(f"{counts[name]:>16}" for name in movements))
    lengths = [trial.end - trial.start for trial in trials]
    print(f"\nTrials used: {len(trials)}, length of a trial: {min(lengths):.1f} to {max(lengths):.1f} s")
    
