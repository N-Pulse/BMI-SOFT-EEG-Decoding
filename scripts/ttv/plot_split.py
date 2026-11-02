"""
Draws how the split file is made up: how many trials of each source, user, session
and movement are in every part (train, val, test and spare).
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
import numpy as np

from scripts.ttv.local_trials import ROOT
from scripts.ttv.split import OUTPUT, PARTS


# ================================================================
# 1. Section: INPUTS
# ================================================================
FIGURE: Path = OUTPUT.with_name("split_composition.png")

PART_COLORS: dict[str, str] = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a", "spare": "#c9c8c1"}
SOURCE_COLORS: dict[str, str] = {"moving": "#4a3aa7", "local": "#e87ba4"}
TEXT_ON_SOURCE: dict[str, str] = {"moving": "white", "local": "#0b0b0b"}
INK: str = "#0b0b0b"
MUTED: str = "#52514e"
SURFACE: str = "#fcfcfb"


# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def read_split(path: Path = OUTPUT) -> list[dict]:
    """Read the split file: one dictionary per trial."""
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def draw_parts(ax: plt.Axes, rows: list[dict]) -> None:
    """How many trials are in every part, and how many of them are MOVING."""
    counts = Counter((row["split"], row["source"]) for row in rows)
    for x, part in enumerate(PARTS):
        bottom = 0
        for source in ("moving", "local"):
            number = counts[(part, source)]
            ax.bar(x, number, bottom=bottom, width=0.62, color=SOURCE_COLORS[source], edgecolor=SURFACE, linewidth=2)
            if number >= 25:
                ax.text(x, bottom + number / 2, str(number), ha="center", va="center", color=TEXT_ON_SOURCE[source])
            elif number > 0:
                # a thin segment: its number goes to the side
                ax.text(x + 0.38, bottom + number / 2 + 9, f"{number} MOVING", ha="left", va="center", color=MUTED, fontsize=9)
            bottom += number
        ax.text(x, bottom + 4, str(bottom), ha="center", va="bottom", color=INK, fontweight="bold")
    ax.set_xticks(range(len(PARTS)), PARTS)
    ax.set_xlim(-0.5, len(PARTS) - 0.2)
    ax.set_ylim(0, 330)
    ax.set_ylabel("trials")
    ax.set_title("Trials in every part", loc="left", color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SOURCE_COLORS[s]) for s in SOURCE_COLORS]
    ax.legend(handles, ["MOVING", "local"], frameon=False, loc="upper right")


def draw_who(ax: plt.Axes, rows: list[dict]) -> None:
    """How many trials of every user and every local session are in every part."""
    groups = {
        "MOVING users": sorted({row["user"] for row in rows if row["source"] == "moving"}),
        "local users": sorted({row["user"] for row in rows if row["source"] == "local"}),
    }
    sessions = sorted({(row["session"], row["user"], row["palm"]) for row in rows if row["source"] == "local"})
    labels, table, breaks = [], [], []
    for users in groups.values():
        for user in users:
            counts = Counter(row["split"] for row in rows if row["user"] == user)
            labels.append(user)
            table.append([counts[part] for part in PARTS])
        breaks.append(len(labels))
    for session, user, palm in sessions:
        counts = Counter(row["split"] for row in rows if row["session"] == session and row["user"] == user)
        labels.append(f"{session} ({user}, {palm})")
        table.append([counts[part] for part in PARTS])

    table = np.array(table)
    share = table / table.sum(axis=1, keepdims=True)
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1.15, aspect="auto")
    for y in range(table.shape[0]):
        for x in range(table.shape[1]):
            ax.text(x, y, str(table[y, x]), ha="center", va="center", color="white" if share[y, x] > 0.55 else INK)
    for y in breaks:
        ax.axhline(y - 0.5, color=SURFACE, linewidth=5)
    ax.set_xticks(range(len(PARTS)), PARTS)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title("Trials of every user and every local session\n(the colour is the share of the row)", loc="left", color=INK, pad=28)


def draw_movements(ax: plt.Axes, rows: list[dict]) -> None:
    """How many trials of every movement are in every part."""
    total = Counter((row["source"], row["movement"]) for row in rows)
    movements = sorted(
        total,
        key=lambda key: (key[0] != "moving", -total[key], key[1]),
    )
    for y, (source, movement) in enumerate(movements):
        left = 0
        for part in PARTS:
            number = sum(1 for row in rows if row["source"] == source and row["movement"] == movement and row["split"] == part)
            ax.barh(y, number, left=left, height=0.7, color=PART_COLORS[part], edgecolor=SURFACE, linewidth=1.5)
            if number >= 2:
                ax.text(left + number / 2, y, str(number), ha="center", va="center", color="white" if part != "spare" else INK, fontsize=8.5)
            left += number
        ax.text(left + 2, y, str(left), va="center", color=MUTED, fontsize=9)
    ax.set_yticks(range(len(movements)), [f"{movement}  [{'MOVING' if source == 'moving' else 'local'}]" for source, movement in movements])
    ax.invert_yaxis()
    ax.set_xlabel("trials")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Trials of every movement", loc="left", color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PART_COLORS[p]) for p in PARTS]
    ax.legend(handles, PARTS, frameon=False, ncol=4, loc="lower right")


def draw_figure(rows: list[dict], path: Path = FIGURE) -> Path:
    """Draw the three pictures into one file."""
    plt.rcParams.update({
        "font.size": 11, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED,
        "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    })
    figure = plt.figure(figsize=(19, 10.5))
    grid = figure.add_gridspec(2, 2, width_ratios=[1, 1.2], height_ratios=[1, 2.4], wspace=0.95, hspace=0.35)
    draw_parts(figure.add_subplot(grid[0, 0]), rows)
    draw_who(figure.add_subplot(grid[1, 0]), rows)
    draw_movements(figure.add_subplot(grid[:, 1]), rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110, bbox_inches="tight", facecolor=SURFACE)
    return path


# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    print(f"Saved: {draw_figure(read_split()).relative_to(ROOT)}")