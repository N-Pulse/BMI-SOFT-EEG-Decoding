"""
A noise model that makes the gel recordings of local look like the dry recordings of MOVING.

MOVING was recorded with dry electrodes and local with gel electrodes. At rest the spectrum of
a dry recording is stronger than the spectrum of local. For the same channel the difference of
the two spectra is the noise that has to be added to local. The dry recordings are very different from each other, so every one of them gives its own profile:
for every frequency a gain (local is weakened where it is stronger than the dry recording)
and a noise (the power that is still missing). Local with the noise of a profile looks like that recording.

The noise is estimated at rest, so it does not depend on the movement and the label is not in it.
It is estimated on the train trials of MOVING and on the eyes-open rest of local.

MOVING has a dry spectrum only for the channels that have a signal in the recording, and only
C3, Cz and C4 of the board electrodes are among them. An electrode of the board without a dry
spectrum gets the average dry spectrum of the other channels of the same recording: in the
hidden-channel test this is the closest guess, and local itself does not help (from 10 Hz on a dry
recording does not depend on what local shows).
"""
# ================================================================
# 0. Section: IMPORTS
# ================================================================
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw into a file, no window
import matplotlib.pyplot as plt
import numpy as np

from scripts.ttv import local_trials, moving_trials
from scripts.ttv.adapt import TARGET_HZ, adapt_recording
from scripts.ttv.board import FLAT_MICROVOLTS
from scripts.ttv.electrodes import BOARD_ELECTRODES, draw_head, head_positions, to_map
from scripts.ttv.local_trials import ROOT
from scripts.ttv.similarity import LOCAL_REST_SKIP, MOVING_REST, cut_windows
from scripts.ttv.split import OUTPUT


# ================================================================
# 1. Section: INPUTS
# ================================================================
# The frequencies (Hz) of the spectrum of a window of one second, and the ones that count
FREQUENCIES: np.ndarray = np.fft.rfftfreq(TARGET_HZ, 1 / TARGET_HZ)
BAND: tuple[float, float] = (1, 45)
MAINS_HZ: float = 50

# A spectrum of a channel needs at least this many windows
MIN_WINDOWS: int = 15

# The hidden-channel test: a channel is hidden only if this many other channels are measured,
# and the random guess is drawn this many times
MIN_OTHERS: int = 5
DRAWS: int = 30

# A channel counts as measured if it has a dry spectrum in at least this many recordings
MIN_RECORDINGS: int = 6

SEED: int = 42

# Where the results are saved. The picture and the profiles are not kept in the repo (see .gitignore)
FIGURE: Path = OUTPUT.with_name("dry_noise.png")
PROFILES: Path = OUTPUT.with_name("dry_noise.npz")

INK: str = "#0b0b0b"
MUTED: str = "#52514e"
SURFACE: str = "#fcfcfb"
GRID: str = "#e6e5e0"
DRY: str = "#4a3aa7"
GEL: str = "#e87ba4"
MADE: str = "#1baf7a"
BLUE: str = "#2a78d6"


# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
@dataclass
class LocalRest:
    """The eyes-open rest of local."""

    spectrum: dict[str, np.ndarray]    # channel -> power, the middle value of the sessions where it has a signal
    board: list[np.ndarray]            # for every session the windows of the 14 electrodes of the board (windows x 14 x samples)
    sessions: int


def closest(target: str, candidates: list[str], where: dict[str, np.ndarray]) -> str:
    """The candidate that is nearest to the target on the head."""
    return min((c for c in candidates if c != target), key=lambda c: np.linalg.norm(where[target] - where[c]))


def working_channels(path: Path, names: dict[str, str]) -> tuple[np.ndarray, dict[str, int]]:
    """Adapt a recording. Returns the signals and the row of every electrode that has a signal."""
    data, channels = adapt_recording(path)
    rows = {}
    for i, channel in enumerate(channels):
        name = names.get(channel.upper())
        if name is not None and np.ptp(data[i]) > FLAT_MICROVOLTS:
            rows[name] = i
    return data, rows


def channel_powers(windows: np.ndarray, rows: dict[str, int]) -> dict[str, np.ndarray]:
    """The power of every frequency of every channel, averaged over the windows.

    A window in which the channel does not move at all is left out, and a channel with too few windows is dropped.
    """
    powers = {}
    for name, i in rows.items():
        moves = windows[:, i].std(axis=1) > 0
        if moves.sum() >= MIN_WINDOWS:
            powers[name] = (np.abs(np.fft.rfft(windows[moves, i], axis=1)) ** 2).mean(axis=0)
    return powers


def train_trials() -> set[tuple[str, str, int]]:
    """The MOVING trials that are in the train part of the split: (user, movement, repetition)."""
    with open(OUTPUT, newline="") as file:
        return {
            (row["user"], row["movement"], int(row["repetition"]))
            for row in csv.DictReader(file)
            if row["source"] == "moving" and row["split"] == "train"
        }


def moving_spectra(names: dict[str, str]) -> list[dict[str, np.ndarray]]:
    """For every dry recording: the spectrum at rest of every channel that has a signal."""
    train = train_trials()
    by_recording: dict[Path, list] = {}
    for trial in moving_trials.load_trials():
        by_recording.setdefault(trial.folder, []).append(trial)

    spectra = []
    for path, trials in sorted(by_recording.items()):
        data, rows = working_channels(path, names)
        windows = []
        for trial in trials:
            if (trial.user, trial.movement, trial.repetition) in train:
                windows += cut_windows(data, trial.start + MOVING_REST[0], trial.start + MOVING_REST[1])
        spectra.append(channel_powers(np.stack(windows), rows))
    return spectra


def local_rest(names: dict[str, str], where: dict[str, np.ndarray]) -> LocalRest:
    """The spectrum at rest of every channel of local, and the rest windows of the 14 electrodes of the board.

    An electrode of the board that a session does not have (flat, or not in the cap) is a copy of the closest one.
    """
    powers, board = [], []
    for path in sorted({trial.folder for trial in local_trials.load_trials()}):
        data, rows = working_channels(path, names)
        rest = next(row for row in local_trials.read_events(path) if row["trial_type"] == "rest_open")
        start = float(rest["onset"]) + LOCAL_REST_SKIP
        windows = np.stack(cut_windows(data, start, float(rest["onset"]) + float(rest["duration"])))
        powers.append(channel_powers(windows, rows))
        board.append(windows[:, [rows[e] if e in rows else rows[closest(e, list(rows), where)] for e in BOARD_ELECTRODES]])
    channels = sorted({c for power in powers for c in power})
    spectrum = {c: np.median([power[c] for power in powers if c in power], axis=0) for c in channels}
    return LocalRest(spectrum, board, len(powers))


def rms(power: np.ndarray) -> np.ndarray:
    """How big the signal is (microvolts) between the frequencies that count."""
    band = (FREQUENCIES >= BAND[0]) & (FREQUENCIES <= BAND[1])
    return np.sqrt(power[..., band].sum(axis=-1)) / TARGET_HZ * np.sqrt(2)


def distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """How far two spectra are from each other: the mean |log10 of the ratio| (0.3 is a factor 2)."""
    band = (FREQUENCIES >= BAND[0]) & (FREQUENCIES <= BAND[1])
    return np.abs(np.log10(first[..., band]) - np.log10(second[..., band])).mean(axis=-1)


def average_spectrum(spectra: list[np.ndarray]) -> np.ndarray:
    """The average of several spectra, frequency by frequency, in the logarithmic scale."""
    return np.exp(np.log(np.maximum(np.stack(spectra), 1e-6)).mean(axis=0))


def noise_profile(dry: np.ndarray, local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """What turns the spectrum of local into the spectrum of a dry recording.

    Where local is stronger it is weakened (gain below 1), then the power that is still missing
    is added as noise. Both are given for every frequency (and every channel).
    """
    gain_power = np.minimum(1.0, dry / local)
    return np.sqrt(gain_power), np.maximum(dry - gain_power * local, 0.0)


def add_noise(windows: np.ndarray, gain: np.ndarray, noise: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Make windows of local (windows x channels x samples) look dry: a gaussian noise with the power of the profile."""
    spectrum = np.fft.rfft(windows, axis=2) * gain[None]
    a, b = rng.standard_normal(spectrum.shape), rng.standard_normal(spectrum.shape)
    noise_spectrum = np.sqrt(noise / 2)[None] * (a + 1j * b)
    # The first and the last frequency have no phase: their noise is real
    noise_spectrum[..., 0] = np.sqrt(noise[:, 0])[None] * a[..., 0]
    noise_spectrum[..., -1] = np.sqrt(noise[:, -1])[None] * a[..., -1]
    return np.fft.irfft(spectrum + noise_spectrum, n=windows.shape[2], axis=2)


def board_gel(local: LocalRest, where: dict[str, np.ndarray]) -> np.ndarray:
    """The spectrum of local for the 14 electrodes of the board (electrodes x frequencies)."""
    return np.stack([
        local.spectrum[e] if e in local.spectrum else local.spectrum[closest(e, list(local.spectrum), where)]
        for e in BOARD_ELECTRODES
    ])


def board_profiles(dry: dict[str, np.ndarray], local: LocalRest, where: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The profile of one dry recording for the 14 electrodes of the board.

    An electrode with a dry spectrum gets its own profile. An electrode without one gets the profile that turns local
    into the average dry spectrum of the channels that are measured in this recording.
    Returns the gain and the noise (electrodes x frequencies) and which electrodes are measured.
    """
    measured = [c for c in dry if c in local.spectrum]
    average = average_spectrum([dry[c] for c in measured])
    gel = board_gel(local, where)
    own = np.array([e in measured for e in BOARD_ELECTRODES])
    target = np.stack([dry[e] if is_own else average for e, is_own in zip(BOARD_ELECTRODES, own)])
    gain, noise = noise_profile(target, gel)
    return gain, noise, own


def hidden_channel_test(dry: list[dict[str, np.ndarray]], local: LocalRest, where: dict[str, np.ndarray],
                        rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Hide a channel that is measured and guess its dry spectrum from the others. How far is the guess from the real one?"""
    results: dict[str, list[float]] = {
        "gel as it is": [], "a random other channel": [], "the closest other channel": [], "the average of the other channels": [],
    }
    for recording in dry:
        measured = [c for c in recording if c in local.spectrum]
        for hidden in measured:
            others = [c for c in measured if c != hidden]
            if len(others) < MIN_OTHERS:
                continue
            truth, gel = recording[hidden], local.spectrum[hidden]

            def guess(channel: str) -> np.ndarray:
                """The profile of another channel, applied to the gel spectrum of the hidden one."""
                gain, noise = noise_profile(recording[channel], local.spectrum[channel])
                return gain ** 2 * gel + noise

            results["gel as it is"].append(distance(gel, truth))
            draws = [others[i] for i in rng.integers(len(others), size=DRAWS)]
            results["a random other channel"].append(np.median([distance(guess(c), truth) for c in draws]))
            results["the closest other channel"].append(distance(guess(closest(hidden, others, where)), truth))
            results["the average of the other channels"].append(distance(average_spectrum([recording[c] for c in others]), truth))
    return {key: np.array(value) for key, value in results.items()}


def draw_figure(dry: list[dict], local: LocalRest, count: dict[str, int], hidden: dict[str, np.ndarray],
                gain: np.ndarray, noise: np.ndarray, own: np.ndarray, gel: np.ndarray, where: dict, path: Path = FIGURE) -> Path:
    """The channels with a measured noise, the hidden-channel test, the level on the board and one electrode without a dry recording."""
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    figure, axes = plt.subplots(2, 2, figsize=(16, 13))
    figure.subplots_adjust(hspace=0.3, wspace=0.22, top=0.92)

    # 1. Which channels of local have a measured dry noise
    ax = axes[0, 0]
    draw_head(ax)
    for channel in sorted(local.spectrum):
        x, y = to_map(where[channel])
        measured, board = count.get(channel, 0) >= MIN_RECORDINGS, channel in BOARD_ELECTRODES
        ax.scatter(x, y, s=420 if board else 190, color=BLUE if measured else "white",
                   edgecolor=INK if board else (BLUE if measured else MUTED), linewidth=2.4 if board else 1, zorder=3)
        ax.text(x, y + (0 if board else 0.05), channel, ha="center", va="center" if board else "bottom", fontsize=8 if board else 6.5,
                fontweight="bold" if board else "normal", color=("white" if measured else INK) if board else MUTED, zorder=4)
    number = sum(1 for c in local.spectrum if count.get(c, 0) >= MIN_RECORDINGS)
    ax.set_title(f"The {len(local.spectrum)} channels of local: the dry noise is measured on {number} (blue),\nassumed on the others (white). Thick ring: the board.", loc="left", fontsize=12)

    # 2. The hidden-channel test
    ax = axes[0, 1]
    order = sorted(hidden, key=lambda key: -np.median(hidden[key]))
    for i, key in enumerate(order):
        values = hidden[key]
        low, middle, high = np.percentile(values, [25, 50, 75])
        ax.barh(i, middle, 0.6, color=MADE if key.startswith("the average") else MUTED, zorder=3)
        ax.plot([low, high], [i, i], color="white", linewidth=5, zorder=4, solid_capstyle="butt")
        ax.plot([low, high], [i, i], color=INK, linewidth=2, zorder=5, solid_capstyle="butt")
        ax.text(high + 0.015, i, f"{middle:.2f}", va="center", fontsize=10.5, fontweight="bold")
    ax.set_yticks(range(len(order)), order)
    ax.set_xlim(0, ax.get_xlim()[1] * 1.1)
    ax.set_xlabel("distance to the real dry spectrum (0.3 = a factor 2); bar: middle, line: the middle half of the cases")
    ax.set_title(f"A measured channel is hidden and guessed from the others\n({len(hidden['gel as it is'])} cases)", loc="left", fontsize=12)

    # 3. The size of the signal on the 14 electrodes after the noise, one dot per dry recording
    ax = axes[1, 0]
    for e in range(len(BOARD_ELECTRODES)):
        for r in range(len(dry)):
            made = rms(gain[r, e] ** 2 * gel[e] + noise[r, e])
            ax.scatter(e + (r - len(dry) / 2) * 0.05, made, s=22, color=DRY if own[r, e] else MADE, alpha=0.85, zorder=3)
        ax.hlines(rms(gel[e]), e - 0.35, e + 0.35, color=GEL, linewidth=3, zorder=4)
    ax.set_yscale("log")
    ax.set_ylim(1.2, 700)
    ax.set_xticks(range(len(BOARD_ELECTRODES)), BOARD_ELECTRODES, fontsize=9)
    ax.set_ylabel("RMS 1-45 Hz at rest (uV)")
    ax.scatter([], [], s=30, color=DRY, label="measured dry noise")
    ax.scatter([], [], s=30, color=MADE, label="average dry spectrum of the recording")
    ax.plot([], [], color=GEL, linewidth=3, label="local (gel)")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", ncol=3)
    ax.set_title("Level on the 14 board electrodes after the noise\n(one dot per dry recording)", loc="left", fontsize=12)

    # 4. One electrode without a dry recording
    ax = axes[1, 1]
    e = BOARD_ELECTRODES.index("FC3")
    show = (FREQUENCIES >= 1) & (FREQUENCIES <= 125)
    for r in range(len(dry)):
        ax.plot(FREQUENCIES[show], (gain[r, e] ** 2 * gel[e] + noise[r, e])[show], color=MADE, alpha=0.5, linewidth=1.2)
    ax.plot(FREQUENCIES[show], gel[e][show], color=GEL, linewidth=2.4)
    ax.plot([], [], color=MADE, label="local with the noise of one dry recording (11 lines)")
    ax.plot([], [], color=GEL, linewidth=2.4, label="local (gel)")
    ax.axvline(MAINS_HZ, color=MUTED, linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("frequency (Hz); dotted: 50 Hz")
    ax.set_ylabel("power of a window of 1 s (uV squared)")
    ax.legend(frameon=False, fontsize=9.5, loc="lower left")
    ax.set_title("FC3 has no dry recording: the 11 lines show\nthe spread between the recordings", loc="left", fontsize=12)

    for ax in axes.ravel()[1:]:
        ax.grid(color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    figure.suptitle("Local made dry, with the noise measured in MOVING", x=0.06, ha="left", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=105, bbox_inches="tight", facecolor=SURFACE)
    return path


# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    where = head_positions()
    names = {name.upper(): name for name in where}
    rng = np.random.default_rng(SEED)
    dry = moving_spectra(names)
    local = local_rest(names, where)

    # 1. What is measured
    count = {c: sum(1 for recording in dry if c in recording) for c in local.spectrum}
    number = sum(1 for c in local.spectrum if count[c] >= MIN_RECORDINGS)
    print(f"Dry recordings (MOVING): {len(dry)}, channels with a dry spectrum in each: {' '.join(str(len(r)) for r in dry)}")
    print(f"Local sessions: {local.sessions}, channels with a signal: {len(local.spectrum)}")
    print(f"Channels with a dry spectrum in at least {MIN_RECORDINGS} of the {len(dry)} recordings: {number}, assumed: {len(local.spectrum) - number}")
    print("Board electrodes, in how many dry recordings the spectrum is measured: "
          + ", ".join(f"{e} {count.get(e, 0)}" for e in BOARD_ELECTRODES))

    # 2. How loud the dry recordings are: they are very different from each other
    print("\nSize of the signal at rest (RMS 1-45 Hz, microvolts), the dry recordings from the quietest to the loudest")
    for channel in ("C3", "Cz", "C4"):
        sizes = np.sort([rms(recording[channel]) for recording in dry if channel in recording])
        print(f"   {channel}: " + " ".join(f"{size:.1f}" for size in sizes)
              + f"   (loudest {sizes[-1] / sizes[0]:.0f} times the quietest, middle {np.median(sizes):.1f}, local {rms(local.spectrum[channel]):.1f})")

    # 3. The profiles of the dry recordings for the 14 electrodes of the board
    profiles = [board_profiles(recording, local, where) for recording in dry]
    gain, noise, own = (np.stack([p[i] for p in profiles]) for i in range(3))
    gel = board_gel(local, where)
    print(f"\nProfiles: {len(dry)} recordings x {len(BOARD_ELECTRODES)} electrodes, {int(own.sum())} measured and {int((~own).sum())} from the average dry spectrum")

    # 4. Do the windows of local with the noise look like the dry recording? (only where the dry spectrum is measured)
    far_gel, far_made = [], []
    for recording, (g, n, is_own) in zip(dry, profiles):
        made = np.median([(np.abs(np.fft.rfft(add_noise(windows, g, n, rng), axis=2)) ** 2).mean(axis=0) for windows in local.board], axis=0)
        for e, electrode in enumerate(BOARD_ELECTRODES):
            if is_own[e]:
                far_gel.append(distance(gel[e], recording[electrode]))
                far_made.append(distance(made[e], recording[electrode]))
    print(f"Distance to the dry recording where its spectrum is measured ({len(far_gel)} cases, middle value): local {np.median(far_gel):.2f}, local + the noise {np.median(far_made):.2f}")

    # 5. Where it is not measured: hide a measured channel and guess it
    hidden = hidden_channel_test(dry, local, where, rng)
    print(f"\nHidden-channel test ({len(hidden['gel as it is'])} cases): distance of the guess to the real dry spectrum, the middle value [middle half of the cases]")
    for key in sorted(hidden, key=lambda k: np.median(hidden[k])):
        low, high = np.percentile(hidden[key], [25, 75])
        print(f"   {key:<36}: {np.median(hidden[key]):.2f} [{low:.2f}, {high:.2f}]")
    better = np.mean(hidden["the average of the other channels"] < hidden["the closest other channel"])
    print(f"   the average is closer than the closest channel in {better:.0%} of the cases")

    # 6. The files
    np.savez(PROFILES, frequencies=FREQUENCIES, electrodes=np.array(BOARD_ELECTRODES), gain=gain, noise=noise, measured=own)
    print(f"\nSaved: {PROFILES.relative_to(ROOT)}")
    print(f"Saved: {draw_figure(dry, local, count, hidden, gain, noise, own, gel, where).relative_to(ROOT)}")
