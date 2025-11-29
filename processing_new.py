# ====================================================================
# NEW EEG Dataset Preprocessing (71-channel eegoSports)
# ====================================================================
#
# Project : BMI-SOFT Signal Processing ML
# Created : 2025-11
# File    : processing_new_fixed.py
#
# Description
# -----------
# This script preprocesses the NEW 71-channel eegoSports EEG recordings
# with BMI protocol triggers and builds a clean ML-ready dataset.
#
# It does:
#   • Loads .xdf files (EEG + trigger stream)
#   • Keeps all EEG channels except AUX/TRIGGER
#   • µV → V, standard_1020 montage, average EEG reference
#   • Band-pass (1–50 Hz) and notch (50, 100 Hz) filtering
#   • ICA (for inspection)
#   • Trigger → Movement Code → 6 coarse EEG classes:
#         1: hand_open
#         2: hand_close
#         3: wrist_flexion
#         4: wrist_extension
#         5: grasp
#         6: pinch
#   • Builds epochs using *only motor-relevant channels*
#   • Generates figures and saves epochs bundle (.npz) for ML.
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
import re
import os
from mne.time_frequency import psd_array_welch
from mne.viz import plot_topomap 

# -------------------------------------------------------------
# CONFIG (NEW DATA)
# -------------------------------------------------------------

DATASET_DIR = Path("NEW_dataset/EEG_clean")
# DATASET_DIR.mkdir(parents=True, exist_ok=True) 

# 6-class EEG labels
EEG_LABELS = {
    "hand_open":      1,
    "hand_close":     2,
    "wrist_flexion":  3,
    "wrist_extension":4,
    "grasp":          5,
    "pinch":          6,
}

# Used both for ERP plots and Epochs/NPZ
SELECTED_EVENT_ID_MAP = EEG_LABELS.copy()

# Motor-related subset for plotting & epochs (must match XDF labels)
PLOTTING_CHANNELS = [
    # Premotor / SMA
    "FC1", "FC2", "FC3", "FC4", "FCZ",
    # Sensorimotor strip
    "C1", "C2", "C3", "C4", "CZ",
    # Centro-parietal
    "CP1", "CP2", "CP3", "CP4",
    # Frontal surround
    "F3", "FZ", "F4",
    # Parietal surround
    "P3", "PZ", "P4",
]

# Contrasts for ERP difference topomaps
CONTRASTS = [
    ("hand_open",      "hand_close"),
    ("wrist_extension","wrist_flexion"),
    ("grasp",          "pinch"),
]

# -------------------------------------------------------------
# TRIGGER MAPPING : LSL → Movement Code → 6 EEG classes
# -------------------------------------------------------------

def decode_and_fuse_lsl_to_eeg_label(lsl_trigger_code):
    """
    Decodes the original LSL trigger code to get the movement_code,
    and fuses it into one of the 6 coarse EEG classes.

    LSL Format: phase*10000 + arm*1000 + baseline*100 + movement_code
    Protocol: PHASE (odd numbers only): cue=1, prep=3, move=5, return=7, iti=9
    
    Returns:
        1 → hand_open
        ...
        6 → pinch
       -1 → discard (non-movement phase or unlisted movement)
    """
    label = int(lsl_trigger_code)
    
    # Special codes (rest / baseline / garbage)
    if label in (9701, 9702, 8888, 8899, 9999):
        return -1

    # Decode components
    phase_label   = label // 10000
    arm_label     = (label - phase_label * 10000) // 1000
    baseline_label= (label - phase_label * 10000 - arm_label * 1000) // 100
    movement_label= label - phase_label * 10000 - arm_label * 1000 - baseline_label * 100
    
    # Only movement phases (5 for 'move' is the most critical)
    if phase_label not in (5,): 
        return -1
    
    # Map Movement Code to 6 EEG Classes (based on Protocol)
    if movement_label in (1,2): 
        # 1: openhand_slow_3sec
        # 2: open thumb
        return EEG_LABELS["hand_open"]
        
    elif movement_label in (2,3,6): 
        # 2, 3: closetofist_allfingerstogether
        # 6: close_onlythumb_3sec
        return EEG_LABELS["hand_close"]
        
    elif movement_label in (4, 5): # wrist_palmarflexion
        return EEG_LABELS["wrist_flexion"]
        
    elif movement_label in (8, 9): # wrist_dorsiflexion
        return EEG_LABELS["wrist_extension"]
        
    elif movement_label in (10, 11): # grasp_donut/cup
        return EEG_LABELS["grasp"]
        
    elif movement_label == 12: # grasping_pinching_pen
        return EEG_LABELS["pinch"]
        
    return -1 # Discard unlisted or irrelevant movements

# -------------------------------------------------------------
# STREAM HANDLING
# -------------------------------------------------------------

def get_eeg_stream(streams):
    # identify the one with most channels as EEG
    eeg_stream = max(streams, key=lambda s: int(s['info']['channel_count'][0]))
    return eeg_stream

def get_trigger_stream(streams):
    # identify the one with 'marker' or 'stim' in type as trigger
    # check if the min amount of channels is unique or not if. not unique make sure that it has data in time_series
    
    # check i fmultiple streams have the same min amount of channels   
    candidates = [s for s in streams if int(s['info']['channel_count'][0]) == min(int(st['info']['channel_count'][0]) for st in streams)]
    if len(candidates) == 1:
        return candidates[0]
    else:
        for s in candidates:
            return max(candidates, key=lambda s: len(s["time_series"]))

    

def get_xdf_file_list(base_dir):
    """
    Scans a directory (and all its subdirectories) for files ending in '.xdf'.

    Args:
        base_dir (str or Path): The root directory to start searching from.

    Returns:
        list: A list of Path objects, each pointing to an .xdf file.
    """
    base_path = Path(base_dir)
    if not base_path.is_dir():
        print(f"Error: Directory not found at {base_dir}")
        return []

    # Use Path.rglob() to recursively find all files matching the pattern
    xdf_files = list(base_path.rglob("*.xdf"))
    
    # Optional: Convert Path objects to strings if needed for external tools
    # return [str(f) for f in xdf_files] 
    
    return xdf_files

def build_eeg_raw(eeg_stream):
    """Create RawArray from EEG stream, removing AUX and TRIGGER channels."""
    chs = eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
    labels_all = [c["label"][0] for c in chs]

    keep_idx = [
        i for i, lbl in enumerate(labels_all)
        if not lbl.startswith("AUX") and lbl != "TRIGGER"
    ]

    data = eeg_stream["time_series"].T * 1e-6  # µV → V
    data = data[keep_idx, :]

    ch_names = [labels_all[i] for i in keep_idx]
    ch_types = ["eeg"] * len(ch_names)

    fs = float(eeg_stream["info"]["nominal_srate"][0])
    info = mne.create_info(ch_names, sfreq=fs, ch_types=ch_types)

    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


def extract_events(streams, eeg_stream):
    """Extract raw events (sample index, 0, code) from trigger stream."""
    stim = get_trigger_stream(streams)
    event_ts   = stim["time_stamps"]
    event_vals = stim["time_series"].flatten().astype(int)
    eeg_ts     = eeg_stream["time_stamps"]

    # Align trigger timestamps to EEG sample indices
    idx = np.searchsorted(eeg_ts, event_ts)
    events = np.column_stack([idx.astype(int),
                              np.zeros_like(idx, dtype=int),
                              event_vals])
    return events


def remap_events(events):
    """LSL codes → Movement Code → 6 EEG classes, drop non-interesting."""
    if events is None or len(events) == 0:
        return events

    raw_codes = events[:, 2].astype(np.int64)
    # Use the new, direct-mapping function
    eeg_labels = np.array([decode_and_fuse_lsl_to_eeg_label(v) for v in raw_codes],
                          dtype=np.int64)

    keep = eeg_labels > 0 # Keep only valid class labels (1-6)
    events_new = events[keep].copy()
    events_new[:, 2] = eeg_labels[keep]
    return events_new


# -------------------------------------------------------------
# MONTAGE, FILTERING, ICA
# -------------------------------------------------------------


def set_montage_and_reference(raw, plot=False, save_dir=None):
    """
    Correct pipeline for eegoSports:
      1) set montage
      2) mark pre-known bad electrodes 
      3) set average reference (needed for interpolation)
    """
    print("Setting montage + average reference...")

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="ignore")

    known_bads = ["M1", "M2", "TP7", "TP8"]
    raw.info["bads"] = [ch for ch in known_bads if ch in raw.ch_names]
    
    # Check for channels that failed to get a 3D position from the montage
    bad_pos_channels = []
    for ch_name in raw.ch_names:
        ch_idx = raw.ch_names.index(ch_name)
        # Check if the location array is non-finite (e.g., NaN from unmatched montage)
        if raw.info['chs'][ch_idx]['loc'].size > 0 and not np.all(np.isfinite(raw.info['chs'][ch_idx]['loc'][:3])):
            bad_pos_channels.append(ch_name)
            
    if bad_pos_channels:
        print(f"Removing channels with non-finite 3D positions (Montage failure): {bad_pos_channels}")
        # Mark as bad and exclude from reference (they will be interpolated later)
        raw.info["bads"].extend(bad_pos_channels)

    raw.set_eeg_reference("average", projection=False)

    # Optional plot
    if plot:
        fig = raw.plot_sensors(show_names=True, show=False)
        if save_dir:
            fig.savefig(save_dir / "montage_layout.png", dpi=150)
        plt.close(fig)

    return raw



def filter_data(raw, save_dir=None):
    """Apply band-pass and notch filtering, save PSD plots (motor subset)."""
    print("Filtering data...")

    if raw.info["bads"]:
        print(f"Skipping interpolation for the following bad channels (will be dropped later): {raw.info['bads']}")
        # Instead of interpolating, we permanently drop the bad channels.
        # This will reduce the number of channels in raw_all_filt.
        raw.drop_channels(raw.info["bads"])
        raw.info["bads"] = [] # Clear the bads list after dropping

    raw_filt = raw.copy()
    # Filter 1-50 Hz (avoids DC offset and line noise alias)
    raw_filt.filter(1., 50., fir_design="firwin", verbose=False)
    # Notch filter for 50 Hz and harmonics
    raw_filt.notch_filter(freqs=[50, 100], verbose=False)

    # Plot PSDs only on motor-subset channels
    try:
        # Note: raw_motor_pre now works on the raw object *after* bad channel dropping
        raw_motor_pre  = raw.copy().pick(PLOTTING_CHANNELS)
        raw_motor_post = raw_filt.copy().pick(PLOTTING_CHANNELS)

        fig1 = raw_motor_pre.compute_psd(fmax=100, n_fft=4096).plot(show=False, average=True, picks='eeg')
        fig2 = raw_motor_post.compute_psd(fmax=100, n_fft=4096).plot(show=False, average=True, picks='eeg')

        if save_dir is not None:
            out1 = save_dir / "psd_before_filtering.png"
            out2 = save_dir / "psd_after_filtering.png"
            fig1.savefig(out1, dpi=150, bbox_inches="tight")
            fig2.savefig(out2, dpi=150, bbox_inches="tight")
            print(f"Saved PSD plot → {out1}")
            print(f"Saved PSD plot → {out2}")
        plt.close(fig1)
        plt.close(fig2)
    except Exception as e:
        print(f"Could not compute/plot PSDs: {e}")

    return raw_filt


def run_ica_and_save(raw, save_dir):
    """Run ICA on full EEG and save diagnostic plots."""
    print("Running ICA...")
    
    # Final data cleaning before ICA (to catch NaNs from interpolation/filtering)
    data = raw.get_data()
    nan_indices = np.where(~np.isfinite(data))
    
    if nan_indices[0].size > 0:
        print(f"Warning: Re-detected and replacing {nan_indices[0].size} non-finite samples with 0 before ICA.")
        data[nan_indices] = 0.0 
        raw._data = data # Update the raw object's data
    # --------------------------------------------------------------------

    ica = mne.preprocessing.ICA(n_components=10, random_state=42, method="infomax")
    # Fit ICA on the now-cleaned raw object
    ica.fit(raw, picks='eeg')

    fig1 = ica.plot_components(show=False)
    fig2 = ica.plot_sources(raw, show=False)

    fig1_path = save_dir / "ica_components.png"
    fig2_path = save_dir / "ica_sources.png"

    fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
    fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    plt.close(fig2)

    print(f"Saved ICA plots → {save_dir}")
    return ica


# -------------------------------------------------------------
# PLOTTING FUNCTIONS
# -------------------------------------------------------------

def plot_events_and_save(events, raw_motor, save_dir):
    """Plot motor-subset channels with events overlaid."""
    if events is None or len(events) == 0:
        print("No events found — skipping raw+events plot.")
        return

    try:
        fig = raw_motor.plot(events=events, scalings="auto",
                             n_channels=len(raw_motor.ch_names),
                             show=False, block=True)
        fig_path = save_dir / "raw_with_events.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved raw+events plot → {fig_path}")
    except Exception as e:
        print(f"Could not plot raw+events: {e}")


def plot_event_distribution(events, save_dir):
    """Plot histogram of event IDs."""
    if events is None or len(events) == 0:
        print("No events found — skipping event count plot.")
        return

    df = pd.DataFrame(events, columns=["sample", "prev", "event_id"])
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(x="event_id", data=df, ax=ax)
    ax.set_title("Event count distribution")
    
    # Use the mapping to get the correct string labels for ticks
    id_to_name = {v: k for k, v in EEG_LABELS.items()}
    tick_names = [id_to_name.get(int(tick.get_text()), '') for tick in ax.get_xticklabels()]
    ax.set_xticklabels(tick_names, rotation=45, ha='right')

    out = save_dir / "event_counts.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved event distribution → {out}")


def plot_erps(raw_motor, events, event_id_map, save_dir):
    """
    Create ERPs (butterfly, mean+SEM), ERP topomaps, PSD band topomaps,
    and ERP-difference topomaps between paired conditions.

    All plots use the motor subset channels (raw_motor).
    """
    print("\nComputing ERPs per condition...")

    if events is None or len(events) == 0:
        print("No events found — skipping ERP plots.")
        return

    try:
        epochs = mne.Epochs(
            raw_motor, events,
            event_id=event_id_map,
            tmin=-0.1, tmax=0.5, preload=True,
            baseline=(None, 0),
            on_missing="ignore",
            event_repeated='drop',
            verbose=False
        )
    except Exception as e:
        print(f"Could not create epochs for ERP: {e}")
        return

    if len(epochs.events) == 0:
        print("No epochs after event matching — skipping ERP plots.")
        return

    df = pd.DataFrame(epochs.events, columns=["_", "__", "event_id"])
    counts = df["event_id"].value_counts()
    print(f"Event counts:\n{counts}")

    TOPO_TIMES = [0.0, 0.1, 0.2, 0.3]
    all_evokeds = {}

    # Per-condition ERPs
    for label_name, label_id in epochs.event_id.items():
        
        ep = epochs[label_name]
        n_ep = len(ep)
        
        if n_ep == 0:
            print(f"No epochs for {label_name} (ID {label_id}), skipping.")
            continue

        print(f" → ERP for {label_name} (n={n_ep})")

        evoked = ep.average()
        all_evokeds[label_name] = evoked.copy()

        # Butterfly ERP
        fig_butterfly = evoked.plot(spatial_colors=True,
                                    time_unit="s",
                                    show=False)
        fig_butterfly.suptitle(f"ERP – {label_name}", fontsize=12)
        fig_butterfly.savefig(save_dir / f"erp_{label_name}_butterfly.png",
                              dpi=150, bbox_inches="tight")
        plt.close(fig_butterfly)

        # ERP mean + SEM over channels
        data = ep.get_data().mean(axis=1)   # (n_epochs, n_times)
        mean = data.mean(axis=0)
        sem  = data.std(axis=0) / np.sqrt(data.shape[0])

        mean_sem_val = float(np.mean(sem))
        max_sem_val  = float(np.max(sem))

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ep.times, mean, label="Mean ERP")
        ax.fill_between(ep.times, mean - sem, mean + sem,
                        alpha=0.3, label="SEM")
        ax.axvline(0, color="k", linestyle="--")
        ax.set_title(f"ERP Average – {label_name}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (V)")
        ax.legend()

        info_text = (
            f"n_epochs = {data.shape[0]}\n"
            f"mean SEM = {mean_sem_val:.2e}\n"
            f"max SEM = {max_sem_val:.2e}"
        )
        ax.text(0.98, 0.98, info_text,
                transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

        fig.savefig(save_dir / f"erp_{label_name}_mean_sem.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ERP topomaps
        evoked_micro = evoked.copy()
        evoked_micro._data *= 1e6

        fig_topo = evoked_micro.plot_topomap(
            TOPO_TIMES,
            time_unit="s",
            show=False,
            scalings=1,
            cmap="RdBu_r"
        )
        fig_topo.suptitle(f"Topomap – {label_name} (µV)",
                          fontsize=12)
        fig_topo.savefig(save_dir / f"erp_{label_name}_topomap.png",
                         dpi=150, bbox_inches="tight")
        plt.close(fig_topo)

        # PSD band-power topomaps over mean ERP signal
        print(f"   • Computing PSD band powers for {label_name} ...")
        ep_data = ep.get_data()             # (n_epochs, n_channels, n_times)
        mean_signal = ep_data.mean(axis=0)  # (n_channels, n_times)

        psd, freqs = psd_array_welch(
            mean_signal,
            sfreq=raw_motor.info["sfreq"],
            fmin=1, fmax=45,
            n_fft=256,
            n_overlap=128,
            verbose=False
        )

        bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 12),
            "beta":  (13, 30),
            "gamma": (30, 45),
        }

        band_powers = {}
        for band, (fmin, fmax) in bands.items():
            idx = (freqs >= fmin) & (freqs <= fmax)
            band_powers[band] = psd[:, idx].mean(axis=1)

        fig = plt.figure(figsize=(12, 8))
        fig.suptitle(f"PSD Band Topomaps – {label_name}", fontsize=14)

        for i, (band, values) in enumerate(band_powers.items(), start=1):
            ax = fig.add_subplot(2, 3, i)
            fmin, fmax = bands[band]
            plot_topomap(values, raw_motor.info, axes=ax,
                         show=False, cmap="Reds")
            ax.set_title(f"{band} ({fmin}-{fmax} Hz)")

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(save_dir / f"psd_{label_name}_bands_topomap.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"   • Saved PSD band-power maps for {label_name}")

    # ERP difference topomaps
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
            time_unit="s",
            cmap="RdBu_r",
            scalings=1,
            show=False
        )
        fig.suptitle(f"ERP Difference: {cond1} − {cond2} (µV)",
                     fontsize=12)

        fig.savefig(save_dir / f"erp_diff_{cond1}_minus_{cond2}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"   • Saved ERP difference topomap for {cond1} − {cond2}")


# ------------------------------------------------
# DATASET BUILDER
# ------------------------------------------------

def save_epochs_npz(epochs: mne.Epochs,
                    npz_path: Path,
                    groups_value: int | None = None):
    """
    Save epochs to NPZ compatible with train.py:
      X: (n_epochs, n_channels, n_times)
      y: (n_epochs,) integer labels (event IDs)
      groups: (n_epochs,) integers (e.g., run index) to avoid leakage
    """
    X = epochs.get_data()                # (n_epochs, n_channels, n_times)
    y = epochs.events[:, 2].astype(int)  # event IDs are already 1–6

    if groups_value is None:
        groups = np.zeros_like(y)
    else:
        groups = np.full_like(y, fill_value=int(groups_value))

    np.savez(npz_path, X=X, y=y, groups=groups)
    print(f"Saved epochs bundle → {npz_path}  "
          f"[X: {X.shape}, y: {y.shape}, groups: {groups.shape}]")


def preprocess_xdf_file(xdf_path: Path,
                        dataset_root: Path = DATASET_DIR):
    """Preprocess a single XDF file and store results in dataset_root."""

    print(f"\n Processing: {xdf_path.name}")
    streams, header = pyxdf.load_xdf(str(xdf_path))
    
    if len(streams) < 2:
        print(f"Error: Only {len(streams)} streams found. Expected at least 2 (Trigger and EEG). Skipping.")
        return None

    # Identify streams (assuming the one with higher channels is EEG)
    eeg_stream = get_eeg_stream(streams)
    trigger_stream = get_trigger_stream(streams)


    raw_all = build_eeg_raw(eeg_stream)        # full EEG (no AUX/TRIGGER)

    # Initial cleaning of non-finite data points
    raw_data = raw_all.get_data()
    nan_indices = np.where(~np.isfinite(raw_data))
    if nan_indices[0].size > 0:
        print(f"Warning: Detected and replacing {nan_indices[0].size} non-finite samples with 0 in the raw data.")
        raw_data[nan_indices] = 0.0 
        raw_all._data = raw_data 


    fs = raw_all.info["sfreq"]
    
    # Events extraction now uses the correct stream identification
    events_raw = extract_events(streams, eeg_stream)
    events     = remap_events(events_raw)
    print(f"Events after remapping: {len(events)}")

    # Create output folders
    fig_dir   = dataset_root / "figures"  / xdf_path.stem
    clean_dir = dataset_root / "processed"
    for d in (fig_dir, clean_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Montage + average ref 
    raw_all = set_montage_and_reference(raw_all, plot=True, save_dir=fig_dir)
    
    # Filtering + Interpolation of bads
    raw_all_filt = filter_data(raw_all, save_dir=fig_dir)

    # Motor-subset view for plotting & epochs
    # raw_motor = raw_all_filt.copy().pick_channels(PLOTTING_CHANNELS, ordered=True)
    raw_motor = raw_all_filt.copy().pick(PLOTTING_CHANNELS)

    # Plots: raw+events, ICA, event distribution, ERPs
    plot_events_and_save(events, raw_motor, save_dir=fig_dir)
    run_ica_and_save(raw_all_filt, save_dir=fig_dir) 
    plot_event_distribution(events, save_dir=fig_dir)
    plot_erps(raw_motor, events, SELECTED_EVENT_ID_MAP, save_dir=fig_dir)

    # Build epochs for ML (same channels as motor subset)
    try:
        epochs = mne.Epochs(
            raw_motor, events,
            event_id=SELECTED_EVENT_ID_MAP,
            tmin=-0.5, tmax=0.8, preload=True,
            on_missing="ignore",
            baseline=(None, 0),
            event_repeated='drop',
            verbose=False
        )

        if len(epochs.events) == 0:
            print("No epochs could be created — skipping NPZ save.")
        else:
            m = re.search(r"run-(\d+)", xdf_path.stem)
            groups_value = int(m.group(1)) if m else 0

            npz_out = dataset_root / "processed" / f"{xdf_path.stem}_epochs.npz"
            save_epochs_npz(epochs, npz_out, groups_value=groups_value)
    except Exception as e:
        print(f"Could not create/save epochs NPZ: {e}")
        return None

    # Save full preprocessed raw (all EEG channels)
    out_path = clean_dir / f"{xdf_path.stem}_raw.fif"
    raw_all_filt.save(out_path, overwrite=True, verbose=False)
    print(f"Saved cleaned data: {out_path}")

    return {
        "file": xdf_path.name,
        "sfreq": fs,
        "n_channels": len(raw_all_filt.ch_names),
        "n_events": int(len(events)),
        "output_path": str(out_path),
    }


def build_dataset(xdf_paths):
    """Iterate over paths and build dataset registry."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = DATASET_DIR / "metadata.csv"

    all_meta = []
    for path in xdf_paths:
        meta = preprocess_xdf_file(Path(path), dataset_root=DATASET_DIR)
        if meta:
            all_meta.append(meta)

    if all_meta:
        with open(metadata_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_meta[0].keys())
            writer.writeheader()
            writer.writerows(all_meta)
        print(f"\n Dataset built successfully → {metadata_path}")
    else:
        print("No metadata to save (no files processed?).")


# ------------------------------------------------
# MAIN
# ------------------------------------------------



if __name__ == "__main__":

    base_directory = "sub-P005"

    # xdf_list = get_xdf_file_list(base_directory) # will find all .xdf within directory
    xdf_list = ["sub-P005/ses-S004/eeg/sub-P005_ses-S004_task-Default_run-001_eeg_down.xdf"]
    build_dataset(xdf_list)