"""
Runs the protocol eeg_board_v1 of the prune package on our own recordings and on MOVING,
and writes the dataset (one file for every trial) into data/dataset/eeg_board_v1/.

Run it from the root of the repository:
    PYTHONPATH=src python scripts/01_prune_dataset/prune_eeg.py
"""
# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from collections import Counter

from bmieeg.prune import DatasetPruner
from bmieeg.prune.domain.actors import Dataset, DatasetPartition, TrialData


# ================================================================
# 1. Section: INPUTS
# ================================================================
# Folders of data/: our own sessions and the EDF files of MOVING
SOURCE_NAME: list[str] = ["bids", "raw/moving/edf"]

PROTOCOL_NAME = "eeg_board_v1"

# The people of MOVING have a number above this
MOVING_ABOVE = 1000

PARTITIONS = ("train", "val", "test")


# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def source_of(trial: TrialData) -> str:
    return "MOVING" if trial.subject_number > MOVING_ABOVE else "local"


def summarize(dataset: Dataset) -> None:
    # The test part of the client has no labels, the one of the server has them
    parts: dict[str, DatasetPartition] = {p.partition_name: p for p in dataset.client}
    parts["test"] = next(p for p in dataset.server if p.partition_name == "test")

    print(f"\n{'part':<8}{'trials':>7}{'local':>7}{'MOVING':>8}{'people':>8}")
    for name in PARTITIONS:
        trials = parts[name].trials
        sources = Counter(source_of(trial) for trial in trials)
        people = len({trial.subject_number for trial in trials})
        print(f"{name:<8}{len(trials):>7}{sources['local']:>7}{sources['MOVING']:>8}{people:>8}")

    trials = [trial for part in parts.values() for trial in part.trials]
    counts = Counter(len(trial.channel_names) for trial in trials)
    print(f"\nElectrodes of a trial (number of trials): "
          + ", ".join(f"{n} electrodes: {c}" for n, c in sorted(counts.items())))
    print(f"Rate: {sorted({trial.sfreq for trial in trials})} Hz, "
          f"numbers: {sorted({str(trial.signal_ch.dtype) for trial in trials})}, "
          f"length of a trial: {min(t.signal_ch.shape[1] for t in trials) / 160:.1f} to "
          f"{max(t.signal_ch.shape[1] for t in trials) / 160:.1f} s")
    biggest = max(int(np.abs(trial.signal_ch).max()) for trial in trials)
    print(f"Biggest count: {biggest:,} (the converter stops at 8,388,607)")


# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    pruner = DatasetPruner(source_name=SOURCE_NAME)

    actors = pruner.run(PROTOCOL_NAME)
    summarize(actors[0])

    pruner.export(actors, PROTOCOL_NAME)
    print(f"\nSaved: data/dataset/{PROTOCOL_NAME}/")
