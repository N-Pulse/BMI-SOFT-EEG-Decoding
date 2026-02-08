# ====================================================================
# EEG Dataset Preprocessing 
# ====================================================================

# Project : BMI-SOFT Signal Processing ML
# Created : 2025-10
# File    : processing_archive.py

# Description
# -----------
# This script builds a centralized, clean EEG dataset from multiple .XDF recordings.

# It performs standardized preprocessing for each recording:
#   • Detects the correct EEG stream automatically (based on channel count)
#   • Sets standard 10–20 electrode montage
#   • Applies band-pass (1–50 Hz) and notch (50 Hz) filtering
#   • Runs ICA for artifact inspection (and future automated cleaning)
#   • Extracts event markers and builds epochs
#   • Generates diagnostic plots:
#         - PSD before/after filtering
#         - Montage layout
#         - Raw signal with event markers
#         - Event count histogram
#         - ERP plots per condition (left, right, foot)
#   • Saves the cleaned data as .fif files and compiles a metadata registry
#
# Outputs
# -------
# ├── EEG_clean/
# │   ├── processed/       → preprocessed .fif files
# │   ├── figures/         → diagnostic and ERP plots per recording
# │   └── metadata.csv     → summary of all processed files
#
# ====================================================================


import mne
import numpy as np
import pyxdf
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import pandas as pd
import seaborn as sns

# -------------------------------------------
# Global constants
# -------------------------------------------

DATASET_DIR = Path("EEG_clean")            # Root directory for processed dataset
EVENT_ID_MAP = event_dict = {
            'unknown?': 0,
            'test': 99,
            'start': 88,
            'baseline' : 77, 

            # elbow flexion, elbow extension, forearm supination, forearm pronation, hand close, and hand open
            'elbow_flexion' : 1,
            'elbow_extension' : 2,
            'forearm_supination' : 3,
            'forearm_pronation' : 4,
            'hand_close' : 5,
            'hand_open' : 6
            }

# Choose a subset to analyze 
SELECTED_EVENT_ID_MAP = {
    'elbow_flexion': 1,
    'elbow_extension': 2,
    'forearm_supination' : 3,
    'forearm_pronation' : 4,
    'hand_close': 5,
    'hand_open': 6,
}

EEG_LABELS = {
    "hand_open": 1,
    "hand_close": 2,
    "wrist_flexion": 3,
    "wrist_extension": 4,
    "grasp": 5,
    "pinch": 6,
}


CHANNELS = [
    # prefrontal
    'Fp1', 'Fp2',
    # frontal
    'F7', 'F3', 'Fz', 'F4', 'F8',
    # central and temporal
    'T3', 'C3', 'Cz', 'C4', 'T4',
    # parietal
    'T5', 'P3', 'Pz', 'P4', 'T6',
    # occipital
    'O1', 'O2',
]



# -------------------------------------------
# Helper functions
# -------------------------------------------
def find_eeg_stream(streams, expected_channels=24):
    """
    Automatically find the EEG stream based on number of channels.
    Returns the first stream with the expected number of channels.
    """
    eeg_candidates = []
    for s in streams:
        try:
            n_ch = len(s["info"]["desc"][0]["channels"][0]["channel"])
        except Exception:
            n_ch = 0
        if n_ch == expected_channels:
            eeg_candidates.append(s)

    if not eeg_candidates:
        available = [
            (s["info"]["name"][0], len(s["info"]["desc"][0]["channels"][0]["channel"]))
            for s in streams if "desc" in s["info"]
        ]
        raise ValueError(
            f"No EEG stream found with {expected_channels} channels. "
            f"Available streams: {available}"
        )

    if len(eeg_candidates) > 1:
        names = [s["info"]["name"][0] for s in eeg_candidates]
        print(f"Multiple {expected_channels}-channel streams found: {names}. Using the first one.")

    eeg_stream = eeg_candidates[0]
    print(f"Detected EEG stream: {eeg_stream['info']['name'][0]} ({expected_channels} channels)")
    return eeg_stream



def extract_mne_info_and_events(streams, eeg_name="EEG-stream", stim_name="stimulus_stream"):
    """Extract EEG info and events from XDF streams."""
    # --- Find EEG stream ---
    eeg_stream = find_eeg_stream(streams, expected_channels=24)
    if eeg_stream is None:
        raise ValueError(f"No stream named '{eeg_name}' found.")

    ch_names = [ch["label"][0] for ch in eeg_stream["info"]["desc"][0]["channels"][0]["channel"]]
    samp_frq = float(eeg_stream["info"]["nominal_srate"][0])
    ch_types = ["eeg"] * len(ch_names)
    info = mne.create_info(ch_names, sfreq=samp_frq, ch_types=ch_types)

    # --- Find stimulus stream ---
    stim_stream = next((s for s in streams if s["info"]["name"][0] == stim_name), None)
    if stim_stream is None:
        raise ValueError(f"No '{stim_name}' stream found in dataset.")

    # --- Build events array ---
    event_timestamps = stim_stream["time_stamps"]
    eeg_timestamps = eeg_stream["time_stamps"]
    event_index = np.searchsorted(eeg_timestamps, event_timestamps)
    event_values = stim_stream["time_series"].flatten()
    events = np.column_stack([event_index.astype(int),
                              np.zeros(len(event_timestamps), dtype=int),
                              event_values])
    return info, events


def set_montage(raw, channels=CHANNELS, plot=False, save_dir=None):
    """Set standard 10-20 montage and optionally save montage plot."""
    print("Setting 10-20 montage...")
    raw_1020 = raw.copy().pick_channels(channels)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw_1020.set_montage(montage, match_case=False)

    if plot:
        fig = raw_1020.plot_sensors(show_names=True, show=False)
        if save_dir:
            fig_path = save_dir / "montage_layout.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved montage plot → {fig_path}")

    return raw_1020


def filter_data(raw, save_dir=None):
    """Apply band-pass and notch filtering, save PSD plot."""
    print("Filtering data...")
    
    raw_filt = raw.copy()
    raw_filt.filter(1., 50., fir_design='firwin') # Band-pass filter 1-50Hz
    raw_filt.notch_filter(freqs=[50, 100])        # Notch filter at 50Hz and its harmonics

    # Save PSD figure
    fig1 = raw.compute_psd().plot(show=False, average=True)
    fig2 = raw_filt.compute_psd().plot(show=False, average=True)
    if save_dir:
        fig1.savefig(save_dir / "psd_before_filtering.png", dpi=150, bbox_inches="tight")
        plt.close(fig1)
        print(f"Saved PSD plot → {save_dir / 'psd_before_filtering.png'}")

        fig2.savefig(save_dir / "psd_after_filtering.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved PSD plot → {save_dir / 'psd_after_filtering.png'}")


    return raw_filt

# -------------------------------------------
# Plotting functions
# -------------------------------------------

def plot_events_and_save(events, fs, raw, save_dir):
    """Plot events overlayed on data and save figures if events exist."""
    if events is None or len(events) == 0:
        print("No events found — skipping event plots.")
        return

    fig = raw.plot(events=events, scalings='auto', n_channels=20, show=False)
    fig_path = save_dir / "raw_with_events.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved raw+events plot → {fig_path}")


def run_ica_and_save(raw, save_dir):
    """Run ICA and save main diagnostic plots."""
    print("Running ICA...")
    ica = mne.preprocessing.ICA(n_components=10, random_state=42)
    ica.fit(raw)

    fig1 = ica.plot_components(show=False)
    fig1_path = save_dir / "ica_components.png"
    fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    fig2 = ica.plot_sources(raw, show=False)
    fig2_path = save_dir / "ica_sources.png"
    fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(f"Saved ICA plots → {save_dir}")
    return ica


def plot_event_distribution(events, save_dir):
    """Plot histogram of event IDs."""
    if events is None or len(events) == 0:
        print("No events found — skipping event count plot.")
        return
    df = pd.DataFrame(events, columns=["sample", "prev", "event_id"])
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(x="event_id", data=df, ax=ax)
    ax.set_title("Event count distribution")
    fig.savefig(save_dir / "event_counts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved event distribution → {save_dir / 'event_counts.png'}")


def plot_erps(raw, events, event_id_map, save_dir):
    """
    Create ERPs, ERPs-mean/SEM, ERP topomaps, PSD-band topomaps,
    and ERP-difference topomaps between paired conditions.
    """

    print("\nComputing ERPs per condition...")

    if events is None or len(events) == 0:
        print("No events found — skipping ERP plots.")
        return

    try:
        epochs = mne.Epochs(
            raw, events, event_id=event_id_map,
            tmin=-0.1, tmax=0.5, preload=True,
            baseline=(None, 0),
            on_missing='ignore'
        )
    except Exception as e:
        print(f"Could not create epochs: {e}")
        return

    # Count events
    df = pd.DataFrame(epochs.events, columns=["_", "__", "event_id"])
    counts = df["event_id"].value_counts()
    print(f"Event counts:\n{counts}")

    # Times for topomaps of electrode activity for the average ERP per condition
    TOPO_TIMES = [0.0, 0.1, 0.2, 0.3]

    # container for evokeds of each condition
    all_evokeds = {}

    # LOOP over labels for each condition to extract ERPs and make relevant plots
    for label in event_id_map.keys():

        n_ep = len(epochs[label])

        if n_ep == 0:
            print(f"No epochs for {label}, skipping.")
            continue

        print(f" → ERP for {label}")

        ep = epochs[label]
        evoked = ep.average()

        # save evoked for the contrasts performed later
        all_evokeds[label] = evoked.copy()

        # BUTTERFLY PLOT of the electrode activity for the conditions
        fig_butterfly = evoked.plot(
            spatial_colors=True,
            time_unit='s',
            show=False
        )
        fig_butterfly.suptitle(f"ERP – {label}", fontsize=12)
        fig_butterfly.savefig(save_dir / f"erp_{label}_butterfly.png",
                              dpi=150, bbox_inches="tight")
        plt.close(fig_butterfly)

        # ERP mean + SEM
        data = ep.get_data().mean(axis=1)
        mean = data.mean(axis=0)
        sem = data.std(axis=0) / np.sqrt(data.shape[0])

        n_ep = data.shape[0]
        mean_sem_val = float(np.mean(sem))
        max_sem_val = float(np.max(sem))

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ep.times, mean, label="Mean ERP")
        ax.fill_between(ep.times, mean - sem, mean + sem,
                        alpha=0.3, label="SEM")
        ax.axvline(0, color='k', linestyle='--')
        ax.set_title(f"ERP Average – {label}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (V)")
        ax.legend()

        info_text = (
            f"n_epochs = {n_ep}\n"
            f"mean SEM = {mean_sem_val:.2e}\n"
            f"max SEM = {max_sem_val:.2e}"
        )
        ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
                ha='right', va='top', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

        fig.savefig(save_dir / f"erp_{label}_mean_sem.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ERP TOPOMAPS at the previously defined time ponints
        evoked_micro = evoked.copy()
        evoked_micro._data *= 1e6

        fig_topo = evoked_micro.plot_topomap(
            TOPO_TIMES,
            time_unit='s',
            show=False,
            scalings=1,
            cmap='RdBu_r'
        )
        fig_topo.suptitle(f"Topomap – {label} (µV)", fontsize=12)
        fig_topo.savefig(save_dir / f"erp_{label}_topomap.png",
                         dpi=150, bbox_inches="tight")
        plt.close(fig_topo)

        # PSD BAND-POWER TOPOMAPS: for every frequency range, plot the spectral activity for the electrodes on the average signal of the ERP per condition
        print(f"   • Computing PSD band powers for {label} ...")

        from mne.time_frequency import psd_array_welch
        
        ep_data = ep.get_data()  
        mean_signal = ep_data.mean(axis=0)  

        n_times = mean_signal.shape[1]

        psd, freqs = psd_array_welch(
            mean_signal,
            sfreq=raw.info['sfreq'],
            fmin=1, fmax=45,
            n_fft=n_times,
            n_per_seg=n_times,
            n_overlap=0,
            verbose=False
        )

        bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 12),
            "beta":  (13, 30),
            "gamma": (30, 45)
        }

        band_powers = {}
        for band, (fmin, fmax) in bands.items():
            idx = (freqs >= fmin) & (freqs <= fmax)
            band_powers[band] = psd[:, idx].mean(axis=1)

        fig = plt.figure(figsize=(12, 8))
        fig.suptitle(f"PSD Band Topomaps – {label}", fontsize=14)

        from mne.viz import plot_topomap

        for i, (band, values) in enumerate(band_powers.items(), start=1):
            ax = fig.add_subplot(2, 3, i)
            fmin, fmax = bands[band]

            plot_topomap(values, raw.info, axes=ax,
                         show=False, cmap='Reds')
            ax.set_title(f"{band} ({fmin}-{fmax} Hz)")

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(save_dir / f"psd_{label}_bands_topomap.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"   • Saved PSD band-power maps for {label}")

    # finally use the previously defined ERP and contrast pairs to compute the ERP DIFFERENCE TOPOMAPS

    CONTRASTS = [
        ("hand_open", "hand_close"),
        ("forearm_supination", "forearm_pronation"),
        ("elbow_extension", "elbow_flexion")
    ]

    print("\nEvokeds available:", list(all_evokeds.keys()))

    for cond1, cond2 in CONTRASTS:

        if cond1 not in all_evokeds or cond2 not in all_evokeds:
            print(f"Skipping contrast {cond1}–{cond2}, missing evoked.")
            continue

        ev1 = all_evokeds[cond1]
        ev2 = all_evokeds[cond2]

        diff = ev1.copy()
        diff.data = ev1.data - ev2.data

        diff_micro = diff.copy()
        diff_micro._data *= 1e6

        fig = diff_micro.plot_topomap(
            TOPO_TIMES,
            time_unit='s',
            cmap='RdBu_r',
            scalings=1,
            show=False
        )
        fig.suptitle(f"ERP Difference: {cond1} − {cond2} (µV)", fontsize=12)

        fig.savefig(save_dir / f"erp_diff_{cond1}_minus_{cond2}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"   • Saved ERP difference topomap for {cond1} − {cond2}")


# ------------------------------------------------
# Dataset builder
# ------------------------------------------------

def save_epochs_npz(epochs: mne.Epochs, event_id_map: dict, npz_path: Path, groups_value: int | None = None):
    """
    Save epochs to NPZ compatible with train.py:
      X: (n_epochs, n_channels, n_times)
      y: (n_epochs,)   integer labels (event IDs)
      groups: (n_epochs,) integers (e.g., run index) to avoid leakage
    """
    X = epochs.get_data()  # shape (n_epochs, n_channels, n_times)
    # y from epochs.events[:, 2] already matches your EVENT_ID_MAP values
    y = epochs.events[:, 2].astype(int)

    if groups_value is None:
        groups = np.zeros_like(y)
    else:
        groups = np.full_like(y, fill_value=int(groups_value))

    np.savez(npz_path, X=X, y=y, groups=groups)
    print(f"Saved epochs bundle → {npz_path}  "
          f"[X: {X.shape}, y: {y.shape}, groups: {groups.shape}]")

def preprocess_xdf_file(xdf_path: Path, dataset_root: Path = DATASET_DIR):
    """Preprocess a single XDF file and store results in dataset_root."""

    print(f"\n Processing: {xdf_path.name}")
    streams, header = pyxdf.load_xdf(str(xdf_path))
    eeg_stream = find_eeg_stream(streams, expected_channels=24)
    data = eeg_stream["time_series"].T * 1e-6  # µV → V

    info, events = extract_mne_info_and_events(streams)
    raw = mne.io.RawArray(data, info)
    fs = raw.info["sfreq"]

    # Create output folders
    fig_dir = dataset_root / "figures" / xdf_path.stem
    clean_dir = dataset_root / "processed"
    for d in [fig_dir, clean_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Run preprocessing pipeline
    raw_1020 = set_montage(raw, channels=CHANNELS, plot=True, save_dir=fig_dir) # standard 10-20 montage
    raw_filt = filter_data(raw_1020, save_dir=fig_dir) # band-pass + notch filtering

    plot_events_and_save(events, fs, raw_filt, save_dir=fig_dir) 
    ica = run_ica_and_save(raw_filt, save_dir=fig_dir)

    # Plot event distribution and ERPs
    plot_event_distribution(events, save_dir=fig_dir)
    plot_erps(raw_filt, events, SELECTED_EVENT_ID_MAP, save_dir=fig_dir)

    # ---- Build epochs once (same params as your ERP step) ----
    try:
        epochs = mne.Epochs(raw_filt, events, event_id=SELECTED_EVENT_ID_MAP,
                            tmin=-0.5, tmax=0.8, preload=True,
                            on_missing='ignore', baseline=(None, 0))
        # restrict to 8–30 Hz for motor features if 
        # epochs.filter(8., 30., fir_design='firwin')

        # Derive a simple groups value from the filename (run index)
        # e.g., ..._run-003_...
        import re
        m = re.search(r'run-(\\d+)', xdf_path.stem)
        groups_value = int(m.group(1)) if m else 0

        npz_out = dataset_root / "processed" / f"{xdf_path.stem}_epochs.npz"
        save_epochs_npz(epochs, SELECTED_EVENT_ID_MAP, npz_out, groups_value=groups_value)
    except Exception as e:
        print(f"Could not create/save epochs NPZ: {e}")

    # Save preprocessed data
    out_path = clean_dir / f"{xdf_path.stem}_raw.fif"
    raw_filt.save(out_path, overwrite=True)

    print(f"Saved cleaned data: {out_path}")
    return {
        "file": xdf_path.name,
        "sfreq": fs,
        "n_channels": len(raw_filt.ch_names),
        "n_events": len(events),
        "output_path": str(out_path)
    }


def build_dataset(xdf_paths):
    """Iterate over paths and build dataset registry."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = DATASET_DIR / "metadata.csv"

    all_meta = []
    for path in xdf_paths:
        meta = preprocess_xdf_file(Path(path), dataset_root=DATASET_DIR)
        all_meta.append(meta)

    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_meta[0].keys())
        writer.writeheader()
        writer.writerows(all_meta)

    print(f"\n Dataset built successfully → {metadata_path}")




# ------------------------------------------------
# Main pipeline for dataset
# ------------------------------------------------
if __name__ == "__main__":
    xdf_list = [
        
        "../../2024/EEG/data/sub-P019/ses-S001/eeg/sub-P019_ses-S001_task-Default_run-001_eeg.xdf",
        "../../2024/EEG/data/sub-P019/ses-S001/eeg/sub-P019_ses-S001_task-Default_run-002_eeg.xdf",
        "../../2024/EEG/data/sub-P019/ses-S001/eeg/sub-P019_ses-S001_task-Default_run-003_eeg.xdf",
        "../../2024/EEG/data/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-001_eeg.xdf",
        "../../2024/EEG/data/sub-TEST1/ses-S001/eeg/sub-TEST1_ses-S001_task-Default_run-001_eeg.xdf",

    ]
    build_dataset(xdf_list)




