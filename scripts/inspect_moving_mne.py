"""Inspect MOVING EDF recordings with MNE."""

import argparse
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import mne


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("path", type=Path, help="EDF file, or a folder searched for *.edf")
parser.add_argument("--show", action="store_true", help="Open interactive trace and PSD plots")
args = parser.parse_args()

files = [args.path] if args.path.is_file() else sorted(args.path.rglob("*.edf"))
if not files:
    raise SystemExit(f"No EDF files found at {args.path}")

for file in files:
    raw = mne.io.read_raw_edf(file, preload=False, infer_types=True, verbose="ERROR")
    raw.set_channel_types({c: "misc" for c in ("X", "Y", "Z") if c in raw.ch_names}, verbose="ERROR")

    # A channel is "flat" if its std over the first 20 s is below 0.01 uV
    eeg = mne.pick_types(raw.info, eeg=True)
    data = raw.get_data(picks=eeg, stop=min(raw.n_times, int(20 * raw.info["sfreq"])))
    flat = [raw.ch_names[i] for i, s in zip(eeg, data.std(axis=1)) if s < 1e-8]

    print(f"\n{file}")
    print(f"  {raw.n_times / raw.info['sfreq']:.1f} s at {raw.info['sfreq']:g} Hz, "
          f"{len(raw.ch_names)} channels ({len(eeg)} EEG)")
    print(f"  Filter: {raw.info['highpass']:g}-{raw.info['lowpass']:g} Hz")
    print(f"  Annotations: {dict(Counter(raw.annotations.description))}")
    print(f"  Flat channels: {flat or 'none'}")

    if args.show:
        raw.plot(duration=10, n_channels=12, scalings={"eeg": 50e-6, "misc": 20.0}, block=True)
        good = [raw.ch_names[i] for i in eeg if raw.ch_names[i] not in flat]
        raw.compute_psd(picks=good, fmax=45).plot(average=True)
        plt.show(block=True)