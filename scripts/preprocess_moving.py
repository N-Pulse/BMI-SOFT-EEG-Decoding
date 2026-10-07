"""Preprocess one MOVING EDF: epochs for rest / motor imagery (MI) / motor execution (ME), per gesture.

Follows Mattei et al. 2024 (Sensors 24(16), 5207):
  10-20 montage -> drop flat channels -> common average reference -> 1-100 Hz Butterworth
  -> 50 Hz notch (not in the paper) -> Extended-Infomax ICA (ICLabel eye/muscle, p > 0.9 removed)
  -> 256 Hz -> band-pass -> 1 s epochs (50 % overlap) -> Welch PSD.

All block types are saved with their trigger number, so the trainer picks the task (state vs gesture).
Trigger IDs are mapped to the protocol order; this mapping is inferred, not stored in the EDF.
"""

import argparse
import re
from pathlib import Path

import mne
import numpy as np
from mne.preprocessing import ICA
from mne_icalabel import label_components
from scipy.signal import welch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BANDS = {"1-45": (1, 45), "1-4": (1, 4), "4-8": (4, 8), "8-12": (8, 12), "12-30": (12, 30), "30-45": (30, 45)}

# Odd triggers start a 6 s block. A repetition is 9 blocks: (rest, MI, ME) x 3 gestures.
CUE_INFO = {
    1: ("rest", "rest"), 7: ("rest", "rest"), 13: ("rest", "rest"),
    3: ("motor_imagery", "open_close"), 9: ("motor_imagery", "wrist_rotation"), 15: ("motor_imagery", "finger_tapping"),
    5: ("motor_execution", "open_close"), 11: ("motor_execution", "wrist_rotation"), 17: ("motor_execution", "finger_tapping"),
}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--file", type=Path, required=True, help="Input MOVING EDF")
parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "results" / "moving_preprocessed")
parser.add_argument("--band", choices=BANDS, default="1-45")
parser.add_argument("--window", type=float, default=1.0,
                    help="Epoch length in s (a 200 ms periodogram has ~5 Hz resolution and is very noisy)")
parser.add_argument("--step", type=float, default=0.5, help="Shift between consecutive epochs in s")
args = parser.parse_args()


def report(step, shape, meaning):
    print(f"[{step}] shape {tuple(shape)} = {meaning}")


# --- 1. Load, 10-20 montage ---
match = re.search(r"Subj[_-]?([A-Za-z0-9]+)", args.file.stem, re.IGNORECASE)
subject = match.group(1) if match else args.file.stem
out = args.outdir / args.file.stem
out.mkdir(parents=True, exist_ok=True)

raw = mne.io.read_raw_edf(args.file, preload=True, infer_types=True, verbose="ERROR")
raw.set_channel_types({c: "misc" for c in ("X", "Y", "Z") if c in raw.ch_names}, verbose="ERROR")
raw.set_montage("standard_1020", on_missing="ignore", verbose="ERROR")
report("1 load", (len(raw.ch_names), raw.n_times), f"channels (incl. X/Y/Z) x samples @ {raw.info['sfreq']:g} Hz")

# --- 2. Reject flat channels (std < 0.01 uV over the first 20 s) ---
eeg = mne.pick_types(raw.info, eeg=True)
head = raw.get_data(picks=eeg, stop=min(raw.n_times, int(20 * raw.info["sfreq"])))
flat = [raw.ch_names[i] for i, s in zip(eeg, head.std(axis=1)) if s < 1e-8]
raw.drop_channels(flat)
n_eeg = len(mne.pick_types(raw.info, eeg=True))
print("Dropped flat channels:", flat or "none")
report("2 drop flat", (n_eeg, raw.n_times), "EEG channels x samples")

# --- 3. CAR, 1-100 Hz Butterworth, 50 Hz notch (shape unchanged) ---
iir = dict(method="iir", iir_params={"order": 4, "ftype": "butter"}, verbose="ERROR")
raw.set_eeg_reference("average", projection=False, verbose="ERROR")
raw.filter(1.0, 100.0, picks="eeg", **iir)
raw.notch_filter(50.0, picks="eeg", **iir)

# --- 4. ICA: remove eye / muscle components with p > 0.9 (shape unchanged) ---
ica = ICA(n_components=0.99, method="infomax", fit_params={"extended": True}, max_iter=500, random_state=97)
ica.fit(raw, picks="eeg", verbose="ERROR")
pred = label_components(raw, ica, method="iclabel")
ica.exclude = [
    i for i, (name, prob) in enumerate(zip(pred["labels"], pred["y_pred_proba"]))
    if name in ("eye blink", "muscle artifact") and prob > 0.9
]
print(f"ICA: {ica.n_components_} components, removed:", [(i, pred["labels"][i]) for i in ica.exclude] or "none")
ica.apply(raw, verbose="ERROR")

# --- 5. Downsample to 256 Hz, band-pass for the chosen band ---
raw.resample(256.0, verbose="ERROR")
low, high = BANDS[args.band]
raw.filter(low, high, picks="eeg", **iir)
report("5 resample", (n_eeg, raw.n_times), "EEG channels x samples @ 256 Hz")

# --- 6. Epochs (50 % overlap) ---
# Each action = 2 s fixation cross, then a 6 s block. Skip the first 1 s of movement and the first 2 s of rest.
picks = mne.pick_types(raw.info, eeg=True)
channels = [raw.ch_names[i] for i in picks]
data = raw.get_data(picks=picks).astype(np.float32)
fs = int(raw.info["sfreq"])
win = round(args.window * fs)
step = round(args.step * fs)
epochs, y, gestures, cues, trials = [], [], [], [], []
for k, (onset, desc) in enumerate(zip(raw.annotations.onset, raw.annotations.description)):
    m = re.fullmatch(r"Trigger#(\d+)", str(desc))
    info = CUE_INFO.get(int(m.group(1))) if m else None
    if info is None:
        continue
    state, gesture = info
    cue_sample = int(round(onset * fs))
    first = cue_sample + (2 if state == "rest" else 1) * fs
    for start in range(first, cue_sample + 6 * fs - win + 1, step):  # last window ends at 6 s
        if start + win <= data.shape[1]:
            epochs.append(data[:, start:start + win])
            y.append(state)
            gestures.append(gesture)
            cues.append(int(m.group(1)))
            trials.append(k)  # trial id: lets the trainer keep overlapping windows together
if not epochs:
    raise SystemExit("No epochs extracted.")
epochs = np.stack(epochs)
report("6 epochs", epochs.shape, f"epochs x channels x samples ({win} samples = {win / fs * 1000:.0f} ms)")

# --- 7. Welch PSD per epoch, then keep only the band ---
freqs, psd = welch(epochs.astype(np.float64) * 1e6, fs=fs, nperseg=win, nfft=fs, axis=-1)  # uV^2/Hz
report("7 welch", psd.shape, "epochs x channels x frequency bins (0-128 Hz)")
keep = (freqs >= low) & (freqs <= high)
features = psd[..., keep].astype(np.float32)
report("8 band crop", features.shape, f"epochs x channels x frequency bins ({low}-{high} Hz)  <- saved as X")

path = out / f"{subject}_epochs_{args.band}.npz"
np.savez_compressed(
    path, X=features, y=np.array(y), gesture=np.array(gestures), cues=np.array(cues), groups=np.array([subject] * len(y)), trials=np.array(trials),
    freqs=freqs[keep], channel_names=np.array(channels),
)
print(f"Saved {path}\nClass counts: {dict(zip(*np.unique(y, return_counts=True)))}")