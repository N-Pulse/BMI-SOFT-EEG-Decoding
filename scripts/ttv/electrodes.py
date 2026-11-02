"""
Which electrodes the board has, and which of them every dataset really has.

The board measures 14 electrodes. A recording has an electrode when the name is in its
cap and the signal is not flat (an electrode without contact never changes).
The maps show the electrodes before the pruning (the whole cap) and after it (the 14 of
the board). The numbers are saved in a table as well.
"""
# ================================================================
# 0. Section: IMPORTS
# ================================================================
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw into a file, no window
import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.patches import Circle, Polygon

from scripts.ttv import local_trials, moving_trials
from scripts.ttv.adapt import open_recording
from scripts.ttv.local_trials import ROOT
from scripts.ttv.split import OUTPUT


# ================================================================
# 1. Section: INPUTS
# ================================================================
# The electrodes of the board (EEG V1 design proposal, 2.2)
BOARD_ELECTRODES: list[str] = ["C1", "C2", "C3", "C4", "C5", "C6", "Cz", "FC1", "FC2", "FC3", "FC4", "CP1", "CP2", "CPz"]
REFERENCE: str = "FCz"
GROUND: str = "AFz"

# Where the electrodes of the 10-05 system are on the head
MONTAGE: str = "colin27_1005"

# The part of the map that is shown for the 14 electrodes of the board
ZOOM_X: tuple[float, float] = (-0.95, 0.95)
ZOOM_Y: tuple[float, float] = (-0.5, 0.95)

# Where the results are saved
FIGURE: Path = OUTPUT.with_name("electrode_maps.png")
TABLE: Path = OUTPUT.with_name("electrode_inventory.csv")

INK: str = "#0b0b0b"
MUTED: str = "#52514e"
SURFACE: str = "#fcfcfb"
REBUILT: str = "#eda100"


# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def head_positions() -> dict[str, np.ndarray]:
    """Where every electrode of the 10-05 system is on the head (x to the right, y to the front, z up)."""
    return mne.channels.make_standard_montage(MONTAGE).get_positions()["ch_pos"]


def to_map(position: np.ndarray) -> tuple[float, float]:
    """Flatten the head like a map: the top is the middle, the ears are on the circle of radius 1."""
    x, y, z = position
    distance = np.arccos(z / np.linalg.norm(position)) / (np.pi / 2)
    angle = np.arctan2(y, x)
    return distance * np.cos(angle), distance * np.sin(angle)


def signals_of(path: Path, names: dict[str, str]) -> dict[str, bool]:
    """Every electrode of a recording, and whether it has a signal."""
    raw, _ = open_recording(path)
    return {
        names.get(channel.upper(), channel): bool(np.ptp(raw.get_data(picks=[channel])) > 0)
        for channel in raw.ch_names
    }


def count_signals(paths: list[Path], names: dict[str, str]) -> Counter:
    """For every electrode: in how many recordings it has a signal (an electrode that is in the cap but flat counts 0)."""
    counts: Counter = Counter()
    for path in paths:
        for electrode, works in signals_of(path, names).items():
            counts[electrode] += int(works)
    return counts


def draw_head(ax: plt.Axes) -> None:
    """The outline of the head: circle, nose and ears."""
    ax.add_patch(Circle((0, 0), 1, fill=False, edgecolor=MUTED, linewidth=1.5))
    ax.add_patch(Polygon([(-0.12, 0.99), (0, 1.16), (0.12, 0.99)], closed=False, fill=False, edgecolor=MUTED, linewidth=1.5))
    for side in (-1, 1):
        ear = np.array([(1.0, 0.18), (1.07, 0.14), (1.09, 0.0), (1.07, -0.14), (1.0, -0.18)])
        ax.add_patch(Polygon(ear * [side, 1], closed=False, fill=False, edgecolor=MUTED, linewidth=1.5))
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.25, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_map(ax: plt.Axes, title: str, electrodes: list[str], counts: Counter, total: int, where: dict, after: bool) -> None:
    """Draw the electrodes. The colour is the share of recordings in which the electrode has a signal."""
    if after:
        # The 14 electrodes of the board are all in the middle of the head: zoom in
        ax.set_xlim(*ZOOM_X)
        ax.set_ylim(*ZOOM_Y)
        ax.set_aspect("equal")
        ax.axis("off")
    else:
        draw_head(ax)

    cmap = plt.get_cmap("Blues")
    for electrode in electrodes:
        x, y = to_map(where[electrode])
        share = counts[electrode] / total
        board = electrode in BOARD_ELECTRODES
        if share > 0:
            fill = cmap(0.15 + 0.85 * share)
        else:
            fill = REBUILT if after else SURFACE
        ax.scatter(x, y, s=800 if after else 150, color=fill, edgecolor=INK if board else MUTED,
                   linewidth=2.4 if board else 0.8, zorder=3)
        if after:
            ax.text(x, y, electrode, ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white" if share > 0.5 else INK, zorder=4)
            ax.text(x, y - 0.115, f"{counts[electrode]}/{total}", ha="center", va="top", fontsize=9, color=MUTED, zorder=4)
        else:
            ax.text(x, y + 0.055, electrode, ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold" if board else "normal", color=INK if board else MUTED, zorder=4)
    if after:
        for name, label, above in ((REFERENCE, "ref", False), (GROUND, "ground", True)):
            x, y = to_map(where[name])
            ax.scatter(x, y, s=130, marker="D", color=SURFACE, edgecolor=MUTED, linewidth=1.5, zorder=3)
            ax.text(x, y + (0.06 if above else -0.07), f"{name} ({label})", ha="center", va="bottom" if above else "top",
                    fontsize=8.5, color=MUTED)
    ax.set_title(title, loc="left", color=INK, fontsize=12.5)


def draw_figure(inventory: dict[str, tuple[Counter, int]], where: dict, path: Path = FIGURE) -> Path:
    """Four maps: MOVING and local, before and after the pruning."""
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK})
    figure, axes = plt.subplots(2, 2, figsize=(15, 14))
    for row, (dataset, (counts, total)) in enumerate(inventory.items()):
        cap = sorted(counts)
        draw_map(axes[row, 0], f"{dataset} before: {len(cap)} electrodes in the cap", cap, counts, total, where, after=False)
        draw_map(axes[row, 1], f"{dataset} after: the {len(BOARD_ELECTRODES)} electrodes of the board (zoom)", BOARD_ELECTRODES, counts, total, where, after=True)
    figure.text(0.5, 0.025, "Colour: in how many recordings the electrode has a signal (dark blue = in all, white = in none). "
                "Amber: no recording has it, it would have to be rebuilt.\nThick ring: an electrode of the board. "
                "The numbers are recordings with a signal / recordings of the dataset.", ha="center", color=MUTED, fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=105, bbox_inches="tight", facecolor=SURFACE)
    return path


def write_table(inventory: dict[str, tuple[Counter, int]], path: Path = TABLE) -> Path:
    """One row for every electrode of every dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["dataset", "electrode", "board", "recordings_with_signal", "recordings"])
        for dataset, (counts, total) in inventory.items():
            for electrode in sorted(set(counts) | set(BOARD_ELECTRODES)):
                writer.writerow([dataset, electrode, electrode in BOARD_ELECTRODES, counts[electrode], total])
    return path


# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    where = head_positions()
    names = {name.upper(): name for name in where}
    recordings = {
        "MOVING": sorted({trial.folder for trial in moving_trials.load_trials()}),
        "local": sorted({trial.folder for trial in local_trials.load_trials()}),
    }
    inventory = {dataset: (count_signals(paths, names), len(paths)) for dataset, paths in recordings.items()}

    # 1. The electrodes of the board in every dataset
    print(f"{'electrode':<10}" + "".join(f"{dataset + ' (of ' + str(total) + ')':>16}" for dataset, (_, total) in inventory.items()))
    for electrode in BOARD_ELECTRODES:
        print(f"{electrode:<10}" + "".join(f"{counts[electrode]:>16}" for counts, _ in inventory.values()))
    print()
    for dataset, (counts, total) in inventory.items():
        real = sum(1 for electrode in BOARD_ELECTRODES if counts[electrode] == total)
        none = sum(1 for electrode in BOARD_ELECTRODES if counts[electrode] == 0)
        print(f"{dataset}: {len(counts)} electrodes in the cap, {real} board electrodes in every recording, "
              f"{none} in no recording (to be rebuilt)")

    # 2. The files
    print(f"\nSaved: {draw_figure(inventory, where).relative_to(ROOT)}")
    print(f"Saved: {write_table(inventory).relative_to(ROOT)}")

