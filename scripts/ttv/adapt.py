"""
Makes a recording look like the data the board will deliver.

The board filters the signal with a simple analog filter (first order, 0.5 to 45 Hz)
and then measures it at a fixed rate. We do the same with every recording, local and
MOVING, so that the model is trained on data that looks like the data it will see.
"""
# ================================================================
# 0. Section: IMPORTS
# ================================================================
from fractions import Fraction
from pathlib import Path

import mne
import numpy as np
from scipy.signal import butter, resample_poly, sosfilt, sosfilt_zi


# ================================================================
# 1. Section: INPUTS
# ================================================================
# The analog filter of the board (Hz), first order
HIGH_PASS_HZ: float = 0.5
LOW_PASS_HZ: float = 45.0

# The rate after the adaptation (samples per second)
TARGET_HZ: int = 500

# Channels that are not EEG
LOCAL_SKIP: list[str] = ["M1", "M2", "EOG", "TRIGGER"]
MOVING_SKIP: list[str] = ["X", "Y", "Z"]


# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def open_recording(path: Path) -> tuple[mne.io.BaseRaw, float]:
    """Open one recording without the channels that are not EEG.

    Returns the recording and the number that turns its values into microvolts.
    """
    # MOVING: one EDF file, MNE gives volts
    if path.suffix == ".edf":
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        raw.drop_channels(MOVING_SKIP)
        return raw, 1e6

    # Local: a session folder. The numbers in the file are already microvolts,
    # even if MNE thinks they are volts
    raw = mne.io.read_raw_fif(next((path / "eeg").glob("*_eeg.fif")), preload=True, verbose="ERROR")
    raw.drop_channels(LOCAL_SKIP, on_missing="ignore")
    return raw, 1.0


def board_filter(signal: np.ndarray, sfreq: float) -> np.ndarray:
    """Filter one channel like the analog filter of the board: first order, 0.5 to 45 Hz.

    Like a real filter it only looks at the past. It starts as if the signal had always
    been at its first value, otherwise the offset of the electrode makes a big jump.
    """
    sos = np.vstack([
        butter(1, HIGH_PASS_HZ, btype="highpass", fs=sfreq, output="sos"),
        butter(1, LOW_PASS_HZ, btype="lowpass", fs=sfreq, output="sos"),
    ])
    filtered, _ = sosfilt(sos, signal, zi=sosfilt_zi(sos) * signal[0])
    return filtered


def resample(signal: np.ndarray, sfreq: float) -> np.ndarray:
    """Change the rate of one channel to TARGET_HZ (the filter against aliasing is included)."""
    ratio = Fraction(TARGET_HZ, round(sfreq))
    if ratio == 1:
        return signal
    return resample_poly(signal, ratio.numerator, ratio.denominator)


def adapt_recording(path: Path) -> tuple[np.ndarray, list[str]]:
    """Adapt every channel of a recording.

    The filter comes first, like on the board, where the analog filter works before
    the signal is measured. Returns the channels x samples table and the channel names.
    """
    raw, to_microvolts = open_recording(path)
    sfreq = raw.info["sfreq"]
    channels = [
        resample(board_filter(raw.get_data(picks=[name])[0] * to_microvolts, sfreq), sfreq)
        for name in raw.ch_names
    ]
    return np.stack(channels), list(raw.ch_names)


# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    from scripts.ttv import local_trials, moving_trials

    # 1. Does the filter do what the board does? The gain of a first order filter is known
    print("Gain of the filter (dB), a sine wave of every frequency is sent through it")
    print(f"{'Hz':>6}{'board (formula)':>18}{'at 1000 Hz':>12}{'at 500 Hz':>12}")
    for hz in (0.1, 0.5, 1, 10, 45, 100):
        formula = 20 * np.log10(
            hz / np.hypot(hz, HIGH_PASS_HZ) / np.sqrt(1 + (hz / LOW_PASS_HZ) ** 2)
        )
        measured = []
        for rate in (1000, 500):
            t = np.arange(60 * rate) / rate
            wave = np.cos(2 * np.pi * hz * t)
            out = board_filter(wave, rate)
            half = len(t) // 2
            measured.append(20 * np.log10(np.std(out[half:]) / np.std(wave[half:])))
        print(f"{hz:>6}{formula:>18.1f}{measured[0]:>12.1f}{measured[1]:>12.1f}")

    # 2. One recording of each kind, before and after
    local = local_trials.load_trials()[-1].folder
    moving = moving_trials.load_trials()[0].folder
    for name, path in (("local", local), ("MOVING", moving)):
        raw, to_microvolts = open_recording(path)
        before = raw.get_data() * to_microvolts
        after, channels = adapt_recording(path)
        working = np.ptp(before, axis=1) > 0
        print(f"\n{name}: {path.name if name == 'MOVING' else path.parents[1].name + '_' + path.name}")
        print(f"   channels: {len(channels)} ({int((~working).sum())} without signal)")
        print(f"   rate: {raw.info['sfreq']:.0f} -> {TARGET_HZ} Hz, length: "
              f"{before.shape[1] / raw.info['sfreq']:.0f} -> {after.shape[1] / TARGET_HZ:.0f} s")
        print(f"   typical size of the signal (microvolts): "
              f"{np.median(np.std(before[working], axis=1)):.1f} -> {np.median(np.std(after[working], axis=1)):.1f}")
