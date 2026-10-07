import numpy as np
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score
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


SUBBANDS = {"theta": (4, 8), "alpha_mu": (8, 13), "beta_low": (13, 20), "beta_high": (20, 30), "gamma": (30, 45)}


FREQ_FEATURES = {"total_power": total_power, "mean_freq": mean_freq, "median_freq": median_freq, "peak_freq": peak_freq}


def _present_subbands(freqs):
    """Sub-bands with at least 2 frequency bins in the data (a narrow --band can drop some)."""
    return {name: (low, high) for name, (low, high) in SUBBANDS.items() if ((freqs >= low) & (freqs < high)).sum() >= 2}


def band_features(psd, freqs):
    """Every spectral feature inside every sub-band -> (epochs, channels, n_bands, n_features)."""
    per_band = []
    for low, high in _present_subbands(freqs).values():
        keep = (freqs >= low) & (freqs < high)
        per_band.append(np.stack([f(psd[..., keep], freqs[keep]) for f in FREQ_FEATURES.values()], axis=-1))
    return np.stack(per_band, axis=-2)


def feature_names(channels, freqs):
    """Names 'channel|band|feature' matching band_features(...).reshape(n_epochs, -1)."""
    return [f"{ch}|{band}|{feat}" for ch in channels for band in _present_subbands(freqs) for feat in FREQ_FEATURES]


def group_importances(names, importances, part):
    """Sum importances over the channel (0), band (1) or feature type (2) part of the names, largest first."""
    totals = {}
    for name, value in zip(names, importances):
        key = name.split("|")[part]
        totals[key] = totals.get(key, 0.0) + value
    return sorted(totals.items(), key=lambda kv: -kv[1])


def feature_importances(model, n_features):
    """Mean absolute linear coefficient per feature, on standardized features (pipeline's last step)."""
    classifier = model.steps[-1][1]
    if not hasattr(classifier, "coef_"):
        raise TypeError("The fitted classifier does not expose linear coefficients")
    coefficients = np.atleast_2d(np.asarray(classifier.coef_, dtype=np.float64))
    if coefficients.shape[1] != n_features:
        raise RuntimeError("Classifier coefficients do not match the features")
    return np.abs(coefficients).mean(axis=0)


def save_grouped_importances(names, importances, outdir, band):
    """One bar plot each for channels, bands and metrics (importances summed over the other two axes)."""
    for part, label in ((0, "channel"), (1, "band"), (2, "metric")):
        ranked = group_importances(names, importances, part)
        keys, values = zip(*ranked)
        fig, ax = plt.subplots(figsize=(7, 0.3 * len(keys) + 1.5))
        ax.barh(keys[::-1], values[::-1], color="tab:blue")
        ax.set_xlabel("summed mean |coefficient| (standardized features)")
        ax.set_title(f"Feature importance by {label}")
        fig.tight_layout()
        fig.savefig(outdir / f"importance_by_{label}_{band}.png", dpi=160)
        plt.close(fig)


def _score_without(model, X_train, y_train, X_test, y_test, keep):
    fitted = clone(model).fit(X_train[:, keep], y_train)
    return balanced_accuracy_score(y_test, fitted.predict(X_test[:, keep]))


def feature_ablation(model, X_train, y_train, X_test, y_test, order, n_remove=20):
    """Balanced accuracy on the test set after cumulatively removing the n_remove most important features.

    order: feature indices from most to least important (ranked on the training set only).
    Returns n_remove + 1 scores: none removed, then the 1st, 1st-2nd, ... removed.
    """
    n_remove = min(n_remove, X_train.shape[1] - 1)  # at least one feature stays
    scores = []
    for k in range(n_remove + 1):
        keep = np.ones(X_train.shape[1], dtype=bool)
        keep[order[:k]] = False
        scores.append(_score_without(model, X_train, y_train, X_test, y_test, keep))
    return scores


def group_ablation(model, X_train, y_train, X_test, y_test, names, importances, part):
    """Same, removing whole groups: channels (part 0), bands (1) or metrics (2), ranked by summed importance.

    Returns the groups from most to least important and the scores after removing 0, 1, ... of them
    (the last group is never removed).
    """
    keys = np.array([name.split("|")[part] for name in names])
    ranked = [k for k, _ in group_importances(names, importances, part)]
    scores = [
        _score_without(model, X_train, y_train, X_test, y_test, ~np.isin(keys, ranked[:k]))
        for k in range(len(ranked))
    ]
    return ranked, scores


def save_ablation(removed_names, scores, chance, path, unit):
    """scores[k] = balanced accuracy after removing removed_names[:k] (cumulative, left to right)."""
    fig, ax = plt.subplots(figsize=(max(6.0, 0.45 * len(scores) + 2.5), 5))
    ax.plot(range(len(scores)), scores, marker="o", color="tab:blue", label="most important removed first")
    ax.axhline(chance, color="gray", linestyle="--", label=f"chance ({chance:.2f})")
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels(["none"] + [n.replace("|", " ") for n in removed_names[:len(scores) - 1]],
                       rotation=60, ha="right")
    ax.set_xlabel(f"{unit} removed, cumulatively from left to right")
    ax.set_ylabel("balanced accuracy (held-out trials)")
    ax.set_title(f"Performance when removing {unit}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


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
    """Colour = row-normalised rate (diagonal = per-class accuracy); each cell shows % and count."""
    rates = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(rates, display_labels=class_names).plot(
        ax=ax, cmap="Blues", xticks_rotation=30, colorbar=False, values_format="")
    for text in ax.texts:
        text.set_text("")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{rates[i, j]:.0%}\n(n={cm[i, j]})", ha="center", va="center",
                    color="white" if rates[i, j] > 0.4 else "black")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
