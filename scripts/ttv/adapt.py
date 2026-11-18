"""
Makes a recording look like the data the board will deliver.

The board filters the signal with a simple analog filter (first order, 0.2 to 80 Hz),
measures it at a fixed rate and turns it into whole numbers with a 16-bit converter.
We do the same with every recording, local and MOVING, so that the model is trained on
data that looks like the data it will see.
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
# The analog filter of the board (Hz), first order (the order is not confirmed yet)
HIGH_PASS_HZ: float = 0.2
LOW_PASS_HZ: float = 80.0

# The rate after the adaptation (samples per second)
# PLACEHOLDER: the real rate must be confirmed with the hardware team (it has to be above 160 Hz)
TARGET_HZ: int = 250

# The converter of the board: 16 bits, so whole numbers from -32768 to 32767
ADC_BITS: int = 16

# The size of one count of the converter (microvolts)
# PLACEHOLDER: the real size must be confirmed with the hardware team
ADC_MICROVOLTS_PER_COUNT: float = 0.2

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
    """Filter one channel like the analog filter of the board: first order, 0.2 to 80 Hz.

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
    the signal is measured. Returns the channels x samples table in microvolts (not yet
    whole numbers, `quantize` does that last) and the channel names.
    """
    raw, to_microvolts = open_recording(path)
    sfreq = raw.info["sfreq"]
    channels = [
        resample(board_filter(raw.get_data(picks=[name])[0] * to_microvolts, sfreq), sfreq)
        for name in raw.ch_names
    ]
    return np.stack(channels), list(raw.ch_names)


def quantize(signal: np.ndarray) -> np.ndarray:
    """Measure a signal (microvolts) like the converter of the board: whole numbers of counts.

    A signal that is too big stays at the limit of the converter, it does not wrap around.
    """
    limit = 2 ** (ADC_BITS - 1)
    counts = np.clip(np.rint(signal / ADC_MICROVOLTS_PER_COUNT), -limit, limit - 1)
    return counts.astype(np.int16 if ADC_BITS <= 16 else np.int32)


def counts_to_microvolts(counts: np.ndarray) -> np.ndarray:
    """Turn the counts of the converter back into microvolts, to check what the converter did."""
    return counts.astype(np.float64) * ADC_MICROVOLTS_PER_COUNT


# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    from scripts.ttv import local_trials, moving_trials

    # 1. Does the filter do what the board does? The gain of a first order filter is known
    print("Gain of the filter (dB), a sine wave of every frequency is sent through it")
    print(f"{'Hz':>6}{'board (formula)':>18}{'at 1000 Hz':>12}{'at 500 Hz':>12}")
    for hz in (0.1, 0.2, 1, 10, 80, 100):
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

    # 2. Does the converter work? A value that is too big stays at the limit, it does not wrap around
    test = np.array([-1e6, -1.55, 0.0, 0.45, 0.55, 1e6]) * ADC_MICROVOLTS_PER_COUNT
    print(f"\nThe converter, test values (microvolts = counts x size of a count): {quantize(test).tolist()}")

    # 3. One recording of each kind, before and after
    local = local_trials.load_trials()[-1].folder
    moving = moving_trials.load_trials()[0].folder
    for name, path in (("local", local), ("MOVING", moving)):
        raw, to_microvolts = open_recording(path)
        before = raw.get_data() * to_microvolts
        after, channels = adapt_recording(path)
        counts = quantize(after)
        working = np.ptp(before, axis=1) > 0
        print(f"\n{name}: {path.name if name == 'MOVING' else path.parents[1].name + '_' + path.name}")
        print(f"   channels: {len(channels)} ({int((~working).sum())} without signal)")
        print(f"   rate: {raw.info['sfreq']:.0f} -> {TARGET_HZ} Hz, length: "
              f"{before.shape[1] / raw.info['sfreq']:.0f} -> {after.shape[1] / TARGET_HZ:.0f} s")
        print(f"   typical size of the signal (microvolts): "
              f"{np.median(np.std(before[working], axis=1)):.1f} -> {np.median(np.std(after[working], axis=1)):.1f}")

        # What did the converter do? The rounding error is measured where the value is not at the limit
        limits = np.iinfo(counts.dtype)
        at_limit = (counts == limits.min) | (counts == limits.max)
        error = np.abs(counts_to_microvolts(counts) - after)[~at_limit].max() / ADC_MICROVOLTS_PER_COUNT
        quietest = np.std(after[working], axis=1).min() / ADC_MICROVOLTS_PER_COUNT
        print(f"   converter: {counts.dtype}, counts from {counts.min()} to {counts.max()}, "
              f"samples at the limit: {int(at_limit.sum())} of {counts.size:,}")
        print(f"   largest rounding error: {error:.2f} counts (0.50 at most), "
              f"the quietest channel moves {quietest:.0f} counts (RMS)")
        print(f"   channels that do not move: {int((np.ptp(counts, axis=1) == 0).sum())} "
              f"(without signal before: {int((~working).sum())})")
