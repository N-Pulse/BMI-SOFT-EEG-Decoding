"""
Reads the events of our own recordings (the sessions downloaded from the archive)
and lists every trial: one repetition of one movement.

A trial starts with the prep signal and ends when the rest after the movement (iti)
is over. All the windows of a trial must stay together in the same part of the
train/validation/test split.
"""
# ================================================================
# 0. Section: IMPORTS
# ================================================================
import csv
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


# ================================================================
# 1. Section: INPUTS
# ================================================================
# Paths: data/bids/<date>/sub-XX/ses-XX
ROOT: Path = Path(__file__).resolve().parents[2]
DATA: Path = ROOT / "data" / "bids"

# The four parts of a trial, in the order they happen
PHASES: tuple[str, ...] = ("prep", "move", "return", "iti")

# A marker that lasts less than this (seconds) is only a point in time
INSTANT_SECONDS: float = 0.05

# Test recordings have 1 to 3 trials, a real session has many more
MIN_TRIALS: int = 12


# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
@dataclass
class Trial:
    """One repetition of one movement."""

    user: str         # "sub-05"
    session: str      # "2025-11-27_ses-02", the date is needed: sub-05 has a ses-02 on two days
    orientation: str  # "Up" or "Side"
    movement: str     # "openhand_slow_3sec"
    repetition: int   # 0, 1, 2... of this movement in this session
    start: float      # seconds from the start of the recording
    end: float
    folder: Path      # the session folder


def read_events(folder: Path) -> list[dict]:
    """Read the events file of a session: one dictionary per row."""
    path = next((folder / "eeg").glob("*_events.tsv"))
    with open(path, newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def find_markers(rows: list[dict]) -> list[dict]:
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


def typical_iti(markers: list[dict]) -> float:
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


def cut_session(folder: Path) -> list[Trial]:
    """List the trials of one session."""
    user, session = folder.parent.name, f"{folder.parents[1].name}_{folder.name}"
    markers = find_markers(read_events(folder))
    iti_seconds = typical_iti(markers)

    # 1. Every prep starts a trial, the markers up to the next prep belong to it
    preps = [i for i, row in enumerate(markers) if row["trial_type"] == "prep"]
    trials, repetitions = [], Counter()
    for n, first in enumerate(preps):
        last = preps[n + 1] if n + 1 < len(preps) else len(markers)
        phases: dict[str, dict] = {}
        for row in markers[first:last]:
            phases.setdefault(row["trial_type"], row)

        # 2. A trial that was cut short (the recording stopped) is left out
        if any(name not in phases for name in PHASES):
            continue

        # 3. The trial ends when a normal iti ends
        iti = phases["iti"]
        movement = phases["prep"]["movement_label"]
        trials.append(Trial(
            user=user,
            session=session,
            orientation=phases["prep"]["baseline_label"],
            movement=movement,
            repetition=repetitions[movement],
            start=float(phases["prep"]["onset"]),
            end=float(iti["onset"]) + iti_seconds,
            folder=folder,
        ))
        repetitions[movement] += 1
    return trials


def load_trials(data: Path = DATA) -> list[Trial]:
    """List the trials of all real sessions (the test recordings are left out)."""
    trials = []
    for folder in sorted(data.glob("*/sub-*/ses-*")):
        found = cut_session(folder)
        if len(found) >= MIN_TRIALS:
            trials += found
    return trials


# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    # 1. One line per session
    print(f"{'session':<20}{'user':<8}{'palm':<6}{'trials':>7}{'movements':>10}  length of a trial")
    for folder in sorted(DATA.glob("*/sub-*/ses-*")):
        found = cut_session(folder)
        movements = len({trial.movement for trial in found})
        lengths = [trial.end - trial.start for trial in found]
        note = f"{min(lengths):.1f} to {max(lengths):.1f} s" if found else "-"
        left_out = "" if len(found) >= MIN_TRIALS else "   (test recording, left out)"
        print(f"{folder.parents[1].name + '_' + folder.name:<20}{folder.parent.name:<8}"
              f"{found[0].orientation if found else '-':<6}{len(found):>7}{movements:>10}  {note}{left_out}")

    # 2. How many trials of each movement every user has
    trials = load_trials()
    print(f"\nTrials used: {len(trials)}")
    for user in sorted({trial.user for trial in trials}):
        counts = Counter(trial.movement for trial in trials if trial.user == user)
        print(f"\n{user}: {sum(counts.values())} trials, {len(counts)} movements")
        for movement, number in sorted(counts.items()):
            print(f"   {movement:<48}{number:>3}")
            
