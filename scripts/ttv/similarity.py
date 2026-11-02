"""
How similar are the signals of MOVING and of our local recordings, for the same task?

Both datasets are adapted to the board first (microvolts, 500 Hz, first order 0.5-45 Hz).
Then the windows of one second of the same task are cut: rest, open and close the hand,
and rotate the wrist (the movements of our recordings are sorted into these tasks).
The spectrum of every electrode is measured, and the datasets are compared on the
electrodes that both of them have.
"""
# ================================================================
# 0. Section: IMPORTS
# ================================================================
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw into a file, no window
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import periodogram

from scripts.ttv import local_trials, moving_trials
from scripts.ttv.adapt import TARGET_HZ, adapt_recording
from scripts.ttv.electrodes import BOARD_ELECTRODES, head_positions
from scripts.ttv.local_trials import ROOT
from scripts.ttv.split import OUTPUT


# ================================================================
# 1. Section: INPUTS
# ================================================================
TASKS: tuple[str, ...] = ("rest", "open_close", "wrist_rotation")

# Which task a movement of our recordings belongs to (my guess, to be confirmed).
# The grasps have no task in MOVING and are left out.
LOCAL_TASK: dict[str, str] = {
    "openhand_slow_3sec": "open_close",
    "openhand_fast_1sec": "open_close",
    "open_fourfingers_together_3sec": "open_close",
    "close_fourfingers_together_3sec": "open_close",
    "closetofist_allfingerstogether_slow_3sec": "open_close",
    "closetofist_allfingerstogether_fast_1sec": "open_close",
    "closetofist_allfingerstogether_3sec_maxforce": "open_close",
    "wrist_dorsiflexion_normalforce_3sec": "wrist_rotation",
    "wrist_dorsiflexion_maxforce_3sec": "wrist_rotation",
    "wrist_palmarflexion_normalforce_3sec": "wrist_rotation",
    "wrist_palmarflexion_maxforce_3sec": "wrist_rotation",
}

# Which part of a block or of a movement is used (seconds). The first seconds are skipped,
# the person does not react at once (the same rule as in the TreeOfTrees windows)
MOVING_REST = (4.0, 8.0)      # after the start of the trial: 2 s cross + 2 s skipped, then 4 s of the rest block
MOVING_MOVE_FROM_END = (5.0, 0.0)   # the last 5 seconds of the executed block, seen from the end of the trial
LOCAL_REST_SKIP = 5.0         # seconds skipped at the start of the eyes-open rest
LOCAL_MOVE = (1.0, 5.0)       # after the start of the move phase

# The bands of the spectrum (Hz), the frequencies that count and the mains frequency
BANDS: dict[str, tuple[float, float]] = {
    "delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45),
}
FREQUENCIES: tuple[float, float] = (1, 45)
LINE: tuple[float, float] = (49, 51)
LINE_RANGE: tuple[float, float] = (1, 100)

# An electrode is compared when it has a signal in at least this share of the recordings
MIN_SHARE: float = 0.5

FIGURE: Path = OUTPUT.with_name("signal_similarity.png")
COLORS: dict[str, str] = {"MOVING": "#4a3aa7", "local": "#e87ba4"}
BAND_COLORS: list[str] = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK: str = "#0b0b0b"
MUTED: str = "#52514e"
SURFACE: str = "#fcfcfb"


# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def cut_windows(data: np.ndarray, start: float, end: float) -> list[np.ndarray]:
    """Cut the signal between two moments (seconds) into windows of one second."""
    windows = []
    second = start
    while second + 1 <= end + 1e-6:
        first = round(second * TARGET_HZ)
        if first + TARGET_HZ <= data.shape[1]:
            windows.append(data[:, first:first + TARGET_HZ])
        second += 1
    return windows


def mean_spectrum(windows: list[np.ndarray]) -> np.ndarray:
    """The spectrum of every channel (microvolts squared per Hz), averaged over the windows."""
    _, power = periodogram(np.stack(windows), fs=TARGET_HZ, window="hann", axis=-1)
    return power.mean(axis=0)


def moving_windows(data: np.ndarray, trials: list) -> dict[str, list[np.ndarray]]:
    """The windows of every task in one recording of MOVING."""
    found = defaultdict(list)
    for trial in trials:
        found["rest"] += cut_windows(data, trial.start + MOVING_REST[0], trial.start + MOVING_REST[1])
        if trial.movement in TASKS:
            found[trial.movement] += cut_windows(data, trial.end - MOVING_MOVE_FROM_END[0], trial.end)
    return found


def local_windows(data: np.ndarray, folder: Path, trials: list) -> dict[str, list[np.ndarray]]:
    """The windows of every task in one recording of ours."""
    events = local_trials.read_events(folder)
    moves = sorted(float(row["onset"]) for row in local_trials.find_markers(events) if row["trial_type"] == "move")
    rest = next(row for row in events if row["trial_type"] == "rest_open")
    found = defaultdict(list)
    found["rest"] += cut_windows(data, float(rest["onset"]) + LOCAL_REST_SKIP, float(rest["onset"]) + float(rest["duration"]))
    for trial in trials:
        task = LOCAL_TASK.get(trial.movement)
        if task:
            move = next(second for second in moves if second >= trial.start - 1e-6)
            found[task] += cut_windows(data, move + LOCAL_MOVE[0], move + LOCAL_MOVE[1])
    return found


def measure(dataset: str, trials: list, names: dict[str, str], average_reference: bool) -> list[dict]:
    """The spectra of every recording of a dataset: for every task, the electrodes that have a signal."""
    by_recording = defaultdict(list)
    for trial in trials:
        by_recording[trial.folder].append(trial)

    recordings = []
    for folder, group in by_recording.items():
        data, channels = adapt_recording(folder)
        channels = [names.get(channel.upper(), channel) for channel in channels]
        working = np.ptp(data, axis=1) > 1e-9
        if average_reference:
            data = data - data[working].mean(axis=0, keepdims=True)
        found = moving_windows(data, group) if dataset == "MOVING" else local_windows(data, folder, group)
        spectra = {}
        for task, windows in found.items():
            power = mean_spectrum(windows)
            spectra[task] = {channel: power[i] for i, channel in enumerate(channels) if working[i]}
        recordings.append(spectra)
    return recordings


def per_electrode(recordings: list[dict], task: str) -> dict[str, np.ndarray]:
    """The spectrum of every electrode, averaged over the recordings where it has a signal."""
    found = defaultdict(list)
    for spectra in recordings:
        for electrode, power in spectra[task].items():
            found[electrode].append(power)
    return {electrode: np.mean(powers, axis=0) for electrode, powers in found.items()}


def common_electrodes(measured: dict[str, list[dict]]) -> list[str]:
    """The electrodes that have a signal in enough recordings of both datasets."""
    keep = None
    for recordings in measured.values():
        count = defaultdict(int)
        for spectra in recordings:
            for electrode in spectra["rest"]:
                count[electrode] += 1
        enough = {electrode for electrode, n in count.items() if n >= MIN_SHARE * len(recordings)}
        keep = enough if keep is None else keep & enough
    return sorted(keep)


def summary(spectra: dict[str, np.ndarray], electrodes: list[str]) -> np.ndarray:
    """One spectrum for a dataset and a task: the middle value of the electrodes."""
    return np.median([spectra[electrode] for electrode in electrodes], axis=0)


def share_in(power: np.ndarray, freqs: np.ndarray, low: float, high: float, whole: tuple[float, float]) -> float:
    """The share of the power of a range of frequencies in the power of a wider range."""
    inside = (freqs >= low) & (freqs <= high)
    outside = (freqs >= whole[0]) & (freqs <= whole[1])
    return float(power[inside].sum() / power[outside].sum())


def rms(power: np.ndarray, freqs: np.ndarray) -> float:
    """How big the signal is (microvolts) between the frequencies that count."""
    band = (freqs >= FREQUENCIES[0]) & (freqs <= FREQUENCIES[1])
    return float(np.sqrt(power[band].sum() * (freqs[1] - freqs[0])))


def similarity(first: np.ndarray, second: np.ndarray, freqs: np.ndarray) -> float:
    """How much the shapes of two spectra overlap: 1 is the same shape, 0 nothing in common."""
    band = (freqs >= FREQUENCIES[0]) & (freqs <= FREQUENCIES[1])
    a, b = first[band] / first[band].sum(), second[band] / second[band].sum()
    middle = (a + b) / 2
    divergence = 0.5 * np.sum(a * np.log2(a / middle)) + 0.5 * np.sum(b * np.log2(b / middle))
    return float(1 - divergence)


def draw_figure(measured, electrodes, freqs, path: Path = FIGURE) -> Path:
    """Spectra of every task, size of the signal of every electrode, bands and the 50 Hz of every recording."""
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    figure, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    figure.subplots_adjust(hspace=0.38, wspace=0.28)
    figure.suptitle("MOVING and local, both adapted to the board (microvolts, 500 Hz, first order 0.5-45 Hz). "
                    "Spectra: the line is the middle value of the electrodes, the band is the middle half of them.",
                    fontsize=12.5, y=0.97)
    spectra = {dataset: {task: per_electrode(recordings, task) for task in TASKS} for dataset, recordings in measured.items()}

    # 1. The spectra
    band = (freqs >= 1) & (freqs <= 125)
    for ax, task in zip(axes[0], TASKS):
        for dataset in measured:
            curves = np.array([spectra[dataset][task][electrode] for electrode in electrodes])
            ax.fill_between(freqs[band], np.percentile(curves, 25, axis=0)[band], np.percentile(curves, 75, axis=0)[band],
                            color=COLORS[dataset], alpha=0.2, linewidth=0)
            ax.plot(freqs[band], np.median(curves, axis=0)[band], color=COLORS[dataset], linewidth=2, label=dataset)
        ax.axvline(LINE[0] + 0.5, color=MUTED, linestyle=":", linewidth=1)
        ax.text(LINE[0] + 0.5, ax.get_ylim()[1], " 50 Hz", color=MUTED, va="top", fontsize=9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("power (microvolts squared per Hz)")
        ax.set_title(f"{task}: the spectrum", loc="left", fontsize=11.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0, 0].legend(frameon=False)

    # 2. How big is the signal of every electrode, compared with MOVING
    ax = axes[1, 0]
    ratios = {e: rms(spectra["local"]["rest"][e], freqs) / rms(spectra["MOVING"]["rest"][e], freqs) for e in electrodes}
    order = sorted(ratios, key=ratios.get)
    ax.scatter([ratios[e] for e in order], range(len(order)), color=[INK if e in BOARD_ELECTRODES else COLORS["local"] for e in order], s=45, zorder=3)
    ax.axvline(1, color=MUTED, linewidth=1)
    ax.set_xscale("log")
    ax.set_xlim(0.06, 1.6)
    ax.set_yticks(range(len(order)), order)
    for label, e in zip(ax.get_yticklabels(), order):
        label.set_fontweight("bold" if e in BOARD_ELECTRODES else "normal")
        label.set_color(INK if e in BOARD_ELECTRODES else MUTED)
    ax.set_xlabel("size of the signal at rest: local / MOVING (1 = the same)")
    ax.set_title("Size of the signal of every electrode\n(black: electrodes of the board)", loc="left", fontsize=11)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # 3. The bands
    ax = axes[1, 1]
    rows = [(task, dataset) for task in TASKS for dataset in measured]
    for y, (task, dataset) in enumerate(rows):
        curve = summary(spectra[dataset][task], electrodes)
        left = 0.0
        for color, (name, (low, high)) in zip(BAND_COLORS, BANDS.items()):
            width = share_in(curve, freqs, low, high - 1e-9, FREQUENCIES)
            ax.barh(y, width, left=left, height=0.7, color=color, edgecolor=SURFACE, linewidth=1.5, label=name if y == 0 else None)
            if width >= 0.07:
                ax.text(left + width / 2, y, f"{width:.0%}", ha="center", va="center", color="white" if color != "#eda100" else INK, fontsize=9)
            left += width
    ax.set_yticks(range(len(rows)), [f"{task} - {dataset}" for task, dataset in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the power between 1 and 45 Hz")
    ax.set_title("Where the power is", loc="left", fontsize=11)
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # 4. The 50 Hz of every recording
    ax = axes[1, 2]
    values = []
    for dataset, recordings in measured.items():
        for number, recording_spectra in enumerate(recordings):
            curve = np.median(list(recording_spectra["rest"].values()), axis=0)
            values.append((dataset, number, share_in(curve, freqs, LINE[0], LINE[1], LINE_RANGE)))
    ax.bar(range(len(values)), [v[2] for v in values], color=[COLORS[v[0]] for v in values], width=0.7)
    ax.set_xticks(range(len(values)), [f"{v[0][0]}{v[1] + 1}" for v in values], fontsize=9)
    ax.set_ylabel("share of the power (1-100 Hz)")
    ax.set_title("The power at 50 Hz in every recording (rest)\n(M = MOVING person, l = our session)", loc="left", fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=105, bbox_inches="tight", facecolor=SURFACE)
    return path


# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    names = {name.upper(): name for name in head_positions()}
    trials = {"MOVING": moving_trials.load_trials(), "local": local_trials.load_trials()}
    freqs = np.fft.rfftfreq(TARGET_HZ, 1 / TARGET_HZ)

    for average_reference in (False, True):
        measured = {dataset: measure(dataset, group, names, average_reference) for dataset, group in trials.items()}
        electrodes = common_electrodes(measured)
        spectra = {dataset: {task: per_electrode(recordings, task) for task in TASKS} for dataset, recordings in measured.items()}
        curves = {dataset: {task: summary(spectra[dataset][task], electrodes) for task in TASKS} for dataset in measured}

        # 1. The numbers
        print("=== " + ("after an average reference in every recording" if average_reference else "as adapted to the board") + " ===")
        print(f"{len(electrodes)} electrodes that both datasets have: {' '.join(electrodes)}\n")
        print(f"{'task':<16}{'MOVING':>10}{'local':>8}{'size':>8}{'':>3}{'shape':>8}{'':>3}{'50 Hz share':>14}")
        print(f"{'':<16}{'(uV)':>10}{'(uV)':>8}{'ratio':>8}{'':>3}{'overlap':>8}{'':>3}{'MOVING / local':>14}")
        for task in TASKS:
            a, b = curves["MOVING"][task], curves["local"][task]
            print(f"{task:<16}{rms(a, freqs):>10.1f}{rms(b, freqs):>8.1f}{rms(b, freqs) / rms(a, freqs):>8.2f}{'':>3}"
                  f"{similarity(a, b, freqs):>8.2f}{'':>3}"
                  f"{share_in(a, freqs, *LINE, LINE_RANGE):>8.3f} /{share_in(b, freqs, *LINE, LINE_RANGE):>6.3f}")
        print("\nShape overlap of two tasks inside one dataset, and of the same task between the datasets:")
        for dataset in curves:
            print(f"   {dataset:<8} rest and open_close {similarity(curves[dataset]['rest'], curves[dataset]['open_close'], freqs):.2f}"
                  f"   rest and wrist_rotation {similarity(curves[dataset]['rest'], curves[dataset]['wrist_rotation'], freqs):.2f}")
        print("   between  " + "   ".join(f"{task} {similarity(curves['MOVING'][task], curves['local'][task], freqs):.2f}" for task in TASKS))
        print()

        # 2. The picture, of the data as adapted to the board
        if not average_reference:
            sessions = sorted({trial.folder for trial in trials["local"]})
            print("In the picture: l1 to l5 are " + ", ".join(
                f"l{number + 1} = {folder.parents[1].name}_{folder.name}" for number, folder in enumerate(sessions)))
            print(f"Saved: {draw_figure(measured, electrodes, freqs).relative_to(ROOT)}\n")