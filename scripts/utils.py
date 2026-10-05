import numpy as np
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GroupShuffleSplit


## -- Load preprocessed data bundles (epochs, channels, freqs) from .npz files ---
def load_bundles(paths):
    """Load every subject, keep the channels they all share, stack the epochs."""
    bundles = []
    for path in paths:
        with np.load(path, allow_pickle=False) as b:
            bundle = {
                "X": np.asarray(b["X"], dtype=np.float32),
                "y": np.asarray(b["y"]),
                "subject": str(b["groups"][0]),
                "trials": [f"{path.stem}:{t}" for t in b["trials"]],
                "freqs": np.asarray(b["freqs"]),
                "channels": np.asarray(b["channel_names"]).astype(str).tolist(),
            }
        report(f"load {bundle['subject']}", bundle["X"].shape, "epochs x channels x frequency bins")
        bundles.append(bundle)

    common = [c for c in bundles[0]["channels"] if all(c in b["channels"] for b in bundles)]
    dropped = sorted(set().union(*(b["channels"] for b in bundles)) - set(common))
    if dropped:
        print(f"Channels missing in at least one subject (excluded everywhere): {dropped}")

    psd = np.concatenate([b["X"][:, [b["channels"].index(c) for c in common], :] for b in bundles])
    y = np.concatenate([b["y"] for b in bundles])
    subjects = np.asarray(sum(([b["subject"]] * len(b["y"]) for b in bundles), []))
    trials = np.asarray(sum((b["trials"] for b in bundles), []))
    return psd, y, subjects, trials, bundles[0]["freqs"], common


# --- Spectral features: psd (epochs, channels, freqs) in uV^2/Hz, freqs (freqs,) -> (epochs, channels) ---
def total_power(psd, freqs):
    """log10 of the power integrated over the band."""
    return np.log10(trapezoid(psd, freqs, axis=-1) + 1e-12)


def mean_freq(psd, freqs):
    """Power-weighted mean frequency (Hz)."""
    return (psd * freqs).sum(axis=-1) / (psd.sum(axis=-1) + 1e-12)


def median_freq(psd, freqs):
    """Frequency below which half of the power lies (Hz)."""
    cumulative = np.cumsum(psd, axis=-1)
    return freqs[(cumulative >= 0.5 * cumulative[..., -1:]).argmax(axis=-1)]


def peak_freq(psd, freqs):
    """Frequency of the highest PSD value (Hz)."""
    return freqs[psd.argmax(axis=-1)]


def log_band_power(psd, freqs, bands=((4, 8), (8, 13), (13, 20), (20, 30), (30, 45))):
    """log10 mean PSD in each sub-band (ERD/ERS lives here) -> (epochs, channels, n_bands).
    Sub-bands with no frequency bin in the data (e.g. outside a narrow --band) are skipped."""
    out = []
    for low, high in bands:
        keep = (freqs >= low) & (freqs < high)
        if keep.any():
            out.append(np.log10(psd[..., keep].mean(axis=-1) + 1e-12))
    return np.stack(out, axis=-1)


def subject_normalize(X, subjects):
    """z-score every feature within each subject (label-free) to remove between-subject offsets."""
    X = X.copy()
    for s in np.unique(subjects):
        m = subjects == s
        X[m] = (X[m] - X[m].mean(axis=0)) / (X[m].std(axis=0) + 1e-9)
    return X


## --- Utility functions ---
def report(step, shape, meaning):
    print(f"[{step}] shape {tuple(shape)} = {meaning}")

def grouped_holdout(y, groups, test_size, seed):
    """Group-wise split; retries seeds until train and test contain every class."""
    classes = set(np.unique(y))
    for attempt in range(100):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed + attempt)
        train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y, groups))
        if set(y[train_idx]) == classes and set(y[test_idx]) == classes:
            return train_idx, test_idx
    raise SystemExit("Could not find a group split containing every class in both train and test.")

## -- Evaluation metrics ---
def save_confusion(cm, class_names, path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", xticks_rotation=30, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)