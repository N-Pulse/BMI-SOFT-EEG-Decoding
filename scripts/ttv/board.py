"""
Keeps only the electrodes of the board.

The model will get the 14 electrodes of the board and nothing else, so every recording
is cut down to the electrodes of the board that really have a signal. Electrodes that the
cap does not have, or that are flat, are not rebuilt here: a recording can have fewer than 14.
The names of the electrodes always go with the signal, because they are not the same in every recording.
"""
# ================================================================
# 0. Section: IMPORTS
# ================================================================
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.ttv import local_trials, moving_trials
from scripts.ttv.adapt import adapt_recording
from scripts.ttv.electrodes import BOARD_ELECTRODES


# ================================================================
# 1. Section: INPUTS
# ================================================================
# The names in our recordings are in capital letters ("CZ"), the board uses "Cz"
BOARD_BY_CAPITALS: dict[str, str] = {name.upper(): name for name in BOARD_ELECTRODES}

# A channel that moves less than this (microvolts) is flat. After the filter a flat channel
# is not exactly constant, the rounding of the computer leaves about 0.000000000001
FLAT_MICROVOLTS: float = 1e-6


# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def board_recording(path: Path) -> tuple[np.ndarray, list[str]]:
    """Adapt a recording and keep the electrodes of the board that have a signal, in the order of the board."""
    data, channels = adapt_recording(path)
    keep = [
        i for i, channel in enumerate(channels)
        if channel.upper() in BOARD_BY_CAPITALS and np.ptp(data[i]) > FLAT_MICROVOLTS
    ]
    names = [BOARD_BY_CAPITALS[channels[i].upper()] for i in keep]
    order = sorted(range(len(keep)), key=lambda k: BOARD_ELECTRODES.index(names[k]))
    return data[[keep[k] for k in order]], [names[k] for k in order]


# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    recordings = {
        "MOVING": sorted({trial.folder for trial in moving_trials.load_trials()}),
        "local": sorted({trial.folder for trial in local_trials.load_trials()}),
    }
    for dataset, paths in recordings.items():
        found = Counter()
        in_all = set(BOARD_ELECTRODES)
        for path in paths:
            data, names = board_recording(path)
            found[tuple(names)] += 1
            in_all &= set(names)
        print(f"{dataset}: the electrodes of the board that every recording has: {' '.join(e for e in BOARD_ELECTRODES if e in in_all) or 'none'}")
        for names, number in found.most_common():
            print(f"   {number:>2} recordings with {len(names):>2} electrodes: {' '.join(names)}")
        print()