"""
Builds the train/validation/test split (TTV) of the trials of MOVING and of local recordings.

A whole trial goes to one part, so the windows of one movement are never in two parts.
The shares follow the EEG V1 design proposal: 70% train, 15% validation and 15% test.
The train has 80% MOVING and 20% local, validation and test the other way round.
There are not enough MOVING trials to use all of local at these shares. The trials that
do not fit are kept in the file as "spare".

Inside a source the trials are given to the parts one by one, always to the part that is
furthest behind for the user, the movement and the session of the trial (minimization).
This way every part gets the same mix of users, movements and sessions.
"""
# ================================================================
# 0. Section: IMPORTS
# ================================================================
import csv
from collections import Counter, defaultdict
from pathlib import Path

import mne
import numpy as np

from scripts.ttv import local_trials, moving_trials
from scripts.ttv.adapt import HIGH_PASS_HZ, LOW_PASS_HZ, TARGET_HZ
from scripts.ttv.local_trials import ROOT, Trial


# ================================================================
# 1. Section: INPUTS
# ================================================================
SEED: int = 42

# The share of all the trials in every part, and how much of every part is MOVING
PART_SHARE: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}
OPEN_SOURCE_SHARE: dict[str, float] = {"train": 0.80, "val": 0.20, "test": 0.20}

# The trials that do not fit are the spare
PARTS: tuple[str, ...] = ("train", "val", "test", "spare")

# How much every kind of balance counts. The task asks for user and movement,
# the session and the pair of both only help
WEIGHT: dict[str, float] = {"user": 3.0, "movement": 2.0, "session": 1.0, "user and movement": 1.0}

# The largest accepted difference (in trials) from a perfectly even mix
TOLERANCE: float = 3.0

# Two trials in a row may overlap by this many seconds: the end of a trial uses the typical iti
MAX_OVERLAP_SECONDS: float = 0.05

# Where the split file is saved
OUTPUT: Path = ROOT / "artifacts" / "ttv_split" / "ttv_split.csv"


# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def plan_rooms(n_moving: int, n_local: int) -> dict[str, dict[str, int]]:
    """How many trials of each source go to each part.

    The total is the biggest number of trials for which there are enough trials of both
    sources. It is a multiple of 100, so that all the counts are whole numbers.
    """
    for total in range(100 * ((n_moving + n_local) // 100), 0, -100):
        size = {part: round(PART_SHARE[part] * total) for part in PART_SHARE}
        moving = {part: round(OPEN_SOURCE_SHARE[part] * size[part]) for part in size}
        local = {part: size[part] - moving[part] for part in size}
        if sum(moving.values()) <= n_moving and sum(local.values()) <= n_local:
            moving["spare"] = n_moving - sum(moving.values())
            local["spare"] = n_local - sum(local.values())
            return {"moving": moving, "local": local}
    raise ValueError("There are not enough trials for a split")


def levels(trial: Trial) -> dict[str, str]:
    """What every part should have the same mix of."""
    return {
        "user": trial.user,
        "movement": trial.movement,
        "session": trial.session,
        "user and movement": f"{trial.user}|{trial.movement}",
    }


def assign_parts(trials: list[Trial], room: dict[str, int], rng: np.random.Generator) -> list[str]:
    """Give every trial of one source a part, exactly as many as there is room for."""
    total = len(trials)
    labels = [levels(trial) for trial in trials]
    size = {name: Counter(label[name] for label in labels) for name in labels[0]}
    taken: dict[tuple[str, str, str], int] = defaultdict(int)
    left = dict(room)
    parts = [""] * total

    for i in rng.permutation(total):
        # 1. How far behind is every part with a user, a movement... like this one?
        # The part should have a share of them as big as its share of the room
        behind = {}
        for part in PARTS:
            if left[part] == 0:
                continue
            gap = 0.0
            for name, level in labels[i].items():
                wanted = size[name][level] * room[part] / total
                gap += WEIGHT[name] * (wanted - taken[(name, level, part)]) / size[name][level]
            behind[part] = gap + rng.uniform(0, 1e-9)

        # 2. The trial goes to the part that is furthest behind
        part = max(behind, key=behind.get)
        parts[i] = part
        left[part] -= 1
        for name, level in labels[i].items():
            taken[(name, level, part)] += 1
    return parts


def largest_gaps(trials: list[Trial], parts: list[str], room: dict[str, int]) -> dict[str, float]:
    """For every factor: the largest difference (in trials) from a perfectly even mix."""
    total = len(trials)
    labels = [levels(trial) for trial in trials]
    gaps = {}
    for name in labels[0]:
        size = Counter(label[name] for label in labels)
        count = Counter((label[name], part) for label, part in zip(labels, parts))
        gaps[name] = max(
            abs(count[(level, part)] - size[level] * room[part] / total)
            for level in size
            for part in PARTS
        )
    return gaps


def read_rate(folder: Path) -> float:
    """The sampling rate of a recording, read from its header."""
    if folder.suffix == ".edf":
        return mne.io.read_raw_edf(folder, preload=False, verbose="ERROR").info["sfreq"]
    return mne.io.read_info(next((folder / "eeg").glob("*_eeg.fif")), verbose="ERROR")["sfreq"]


def build_split() -> tuple[list[dict], dict[str, dict[str, int]], dict[str, dict[str, float]]]:
    """Split the trials. Returns the rows of the file, the room of every part and the gaps."""
    rng = np.random.default_rng(SEED)
    sources = {"moving": moving_trials.load_trials(), "local": local_trials.load_trials()}
    rooms = plan_rooms(len(sources["moving"]), len(sources["local"]))

    rows, gaps = [], {}
    for source, trials in sources.items():
        parts = assign_parts(trials, rooms[source], rng)
        gaps[source] = largest_gaps(trials, parts, rooms[source])
        rows += [{"source": source, "trial": trial, "split": part} for trial, part in zip(trials, parts)]

    # The same order every time, so that the ids do not change
    rows.sort(key=lambda row: (row["source"], row["trial"].user, row["trial"].session, row["trial"].start))
    for number, row in enumerate(rows):
        row["trial_id"] = f"{row['source']}-{number:04d}"
    return rows, rooms, gaps


def check_split(rows: list[dict], rooms: dict[str, dict[str, int]], gaps: dict[str, dict[str, float]]) -> None:
    """Stop with an error if the split is not what it should be."""
    # 1. No trial twice
    keys = Counter(
        (row["source"], row["trial"].session, row["trial"].user, row["trial"].movement, row["trial"].repetition)
        for row in rows
    )
    if max(keys.values()) > 1:
        raise ValueError("A trial is in the file twice")

    # 2. Exactly as many trials in every part as planned
    for source in rooms:
        counted = Counter(row["split"] for row in rows if row["source"] == source)
        for part in PARTS:
            if counted[part] != rooms[source][part]:
                raise ValueError(f"{source} {part}: {counted[part]} trials instead of {rooms[source][part]}")
    used = sum(1 for row in rows if row["split"] != "spare")
    for part, share in PART_SHARE.items():
        if sum(1 for row in rows if row["split"] == part) != round(share * used):
            raise ValueError(f"{part} is not {share:.0%} of the trials that are used")

    # 3. The trials of one recording never overlap in time
    by_recording = defaultdict(list)
    for row in rows:
        by_recording[row["trial"].folder].append(row["trial"])
    for folder, trials in by_recording.items():
        trials.sort(key=lambda trial: trial.start)
        for before, after in zip(trials, trials[1:]):
            if before.end > after.start + MAX_OVERLAP_SECONDS:
                raise ValueError(f"Two trials overlap in {folder}")

    # 4. The parts have the same mix of users, movements and sessions
    for source, source_gaps in gaps.items():
        for name, gap in source_gaps.items():
            if gap > TOLERANCE:
                raise ValueError(f"{source}, {name}: the mix is {gap:.1f} trials away from an even one")


def write_split(rows: list[dict], path: Path = OUTPUT) -> Path:
    """Save one row per trial."""
    rates: dict[Path, float] = {}
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "trial_id", "source", "user", "session", "palm", "movement", "repetition", "start_s", "end_s",
        "split", "source_path", "source_sampling_rate_hz", "target_sampling_rate_hz", "analog_filter",
    ]
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            trial = row["trial"]
            if trial.folder not in rates:
                rates[trial.folder] = read_rate(trial.folder)
            writer.writerow({
                "trial_id": row["trial_id"],
                "source": row["source"],
                "user": trial.user,
                "session": trial.session,
                "palm": trial.orientation,
                "movement": trial.movement,
                "repetition": trial.repetition,
                "start_s": f"{trial.start:.3f}",
                "end_s": f"{trial.end:.3f}",
                "split": row["split"],
                "source_path": trial.folder.relative_to(ROOT),
                "source_sampling_rate_hz": rates[trial.folder],
                "target_sampling_rate_hz": TARGET_HZ,
                "analog_filter": f"causal first-order {HIGH_PASS_HZ}-{LOW_PASS_HZ} Hz band-pass",
            })
    return path


# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    rows, rooms, gaps = build_split()
    check_split(rows, rooms, gaps)
    path = write_split(rows)

    # 1. The parts: how many trials, and how many of them are MOVING
    used = sum(1 for row in rows if row["split"] != "spare")
    print(f"{len(rows)} trials, {used} used and {len(rows) - used} spare\n")
    print(f"{'part':<7}{'trials':>7}{'share':>8}{'MOVING':>8}{'local':>7}")
    for part in PARTS:
        moving = rooms["moving"][part]
        local = rooms["local"][part]
        share = "" if part == "spare" else f"{(moving + local) / used:.1%}"
        print(f"{part:<7}{moving + local:>7}{share:>8}{moving:>8}{local:>7}")

    # 2. Does every part have every user?
    print("\nUsers in every part (trials per user, smallest to largest):")
    for source in ("moving", "local"):
        users = sorted({row["trial"].user for row in rows if row["source"] == source})
        text = []
        for part in PARTS:
            counts = Counter(row["trial"].user for row in rows if row["source"] == source and row["split"] == part)
            text.append(f"{part} {len(counts)}/{len(users)} ({min(counts.values())}-{max(counts.values())})")
        print(f"   {source:<7}" + "   ".join(text))

    # 3. Local sessions in every part
    print("\nLocal trials per session:")
    print(f"   {'session':<22}{'user':<8}" + "".join(f"{part:>7}" for part in PARTS))
    sessions = sorted({(row["trial"].session, row["trial"].user) for row in rows if row["source"] == "local"})
    for session, user in sessions:
        counts = Counter(row["split"] for row in rows if row["trial"].session == session and row["trial"].user == user)
        print(f"   {session:<22}{user:<8}" + "".join(f"{counts[part]:>7}" for part in PARTS))

    # 4. How even is the mix? Zero would be perfect
    print("\nLargest difference from a perfectly even mix (trials):")
    for source, source_gaps in gaps.items():
        print(f"   {source:<7}" + "   ".join(f"{name} {gap:.1f}" for name, gap in source_gaps.items()))
    print(f"\nSaved: {path.relative_to(ROOT)}")
