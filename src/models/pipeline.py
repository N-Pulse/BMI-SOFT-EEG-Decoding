"""Training pipeline: data loading, featureizers, classifiers, evaluation."""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
from scipy.signal import welch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt

try:
    from mne.decoding import CSP
    _HAS_MNE = True
except Exception:
    _HAS_MNE = False


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_npz_bundle(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    with np.load(path, allow_pickle=True) as npz:
        X = npz['X']
        y = npz['y']
        meta = dict(npz)
    meta.pop('X', None)
    meta.pop('y', None)
    return X, y, meta


def load_dataset(data_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load .npz bundles from a file or directory. Returns X, y, groups."""
    files: List[Path] = []
    if data_path.is_file() and data_path.suffix == '.npz':
        files = [data_path]
    elif data_path.is_dir():
        files = sorted([p for p in data_path.glob('*.npz')])
    else:
        raise FileNotFoundError(f"No .npz found at {data_path}")

    Xs, ys, groups = [], [], []
    for gi, f in enumerate(files):
        X, y, meta = load_npz_bundle(f)
        grp = meta.get('groups')
        if grp is None:
            grp = np.full(len(y), gi, dtype=int)
        Xs.append(X)
        ys.append(y)
        groups.append(np.asarray(grp))

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    groups = np.concatenate(groups, axis=0)
    return X, y, groups


# ------------------------------ Featureizers ------------------------------

@dataclass
class BandConfig:
    fs: float
    bands: List[Tuple[float, float]]
    nperseg: int = 256
    noverlap: int = 128


class Bandpower(TransformerMixin, BaseEstimator):
    """Compute log band power per channel for each epoch."""

    def __init__(self, fs: float, bands: List[Tuple[float, float]] | None = None,
                 nperseg: int = 256, noverlap: int = 128, eps: float = 1e-12):
        self.fs = fs
        self.bands = bands or [(4, 8), (8, 12), (13, 30), (30, 45)]
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.eps = eps
        self._freqs = None
        self.feature_names_: List[str] | None = None

    def fit(self, X, y=None):
        n_ch = X.shape[1]
        names = []
        for ch in range(n_ch):
            for (lo, hi) in self.bands:
                names.append(f"ch{ch}_bp_{lo:g}-{hi:g}Hz")
        self.feature_names_ = names
        return self

    def transform(self, X):
        if X.ndim != 3:
            raise ValueError("Bandpower expects X with shape (n_epochs, n_channels, n_times)")
        n_epochs, n_ch, _ = X.shape
        out = np.zeros((n_epochs, n_ch * len(self.bands)), dtype=np.float32)
        for i in range(n_epochs):
            for ch in range(n_ch):
                f, Pxx = welch(X[i, ch, :], fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
                for b_idx, (lo, hi) in enumerate(self.bands):
                    mask = (f >= lo) & (f < hi)
                    bp = np.trapz(Pxx[mask], f[mask])
                    out[i, ch * len(self.bands) + b_idx] = np.log(bp + self.eps)
        return out


class FlattenIfNeeded(TransformerMixin, BaseEstimator):
    """If X is (n, c, t), flatten to (n, c*t). If already 2D, pass through."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if X.ndim == 3:
            n, c, t = X.shape
            return X.reshape(n, c * t)
        return X


class CSPFeatures(TransformerMixin, BaseEstimator):
    """CSP features (log-variance). Requires MNE."""

    def __init__(self, n_components: int = 4, reg: str | float | None = 'ledoit_wolf',
                 l_freq: float = 8.0, h_freq: float = 30.0, sfreq: float | None = None):
        if not _HAS_MNE:
            raise ImportError("mne not available. Install MNE to use CSP.")
        self.n_components = n_components
        self.reg = reg
        self.l_freq = 8.0
        self.h_freq = 30.0
        self.sfreq = sfreq
        self._csp = CSP(n_components=n_components, reg=reg, log=True, cov_est='concat')

    def fit(self, X, y):
        if X.ndim != 3:
            raise ValueError("CSP expects raw epochs: (n_epochs, n_channels, n_times)")
        self._csp.fit(X, y)
        return self

    def transform(self, X):
        return self._csp.transform(X)


# ------------------------------ Pipeline builder ------------------------------

def build_pipeline(args, fs: float | None, n_channels: int | None) -> Tuple[Pipeline, List[str] | None]:
    feature_names = None
    feature_steps = []

    if args.model == 'csp_lda':
        if not _HAS_MNE:
            raise ImportError("mne not available. Install MNE to use CSP.")
        feat = CSPFeatures(n_components=args.csp_components)
        feature_steps.append(('csp', feat))
    else:
        if args.features == 'bandpower':
            if fs is None:
                raise ValueError("--fs is required when using bandpower on raw epochs")
            bp = Bandpower(fs=fs)
            feature_steps.append(('bandpower', bp))
            feature_names = bp.feature_names_
        elif args.features == 'none':
            feature_steps.append(('flatten', FlattenIfNeeded()))
        else:
            raise ValueError(f"Unknown features: {args.features}")

    if args.model == 'lda' or args.model == 'csp_lda':
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    elif args.model == 'logreg':
        clf = LogisticRegression(max_iter=500, n_jobs=None, C=args.C, class_weight='balanced')
    elif args.model == 'lsvm':
        clf = LinearSVC(C=args.C, class_weight='balanced')
    else:
        raise ValueError(f"Unknown model: {args.model}")

    steps = []
    if feature_steps:
        steps.extend(feature_steps)
    steps.append(('scaler', StandardScaler()))
    steps.append(('clf', clf))
    pipe = Pipeline(steps)
    return pipe, feature_names


# ------------------------------ Evaluation ------------------------------

def evaluate_and_report(y_true, y_pred, outdir: Path) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center')
    plt.tight_layout()
    fig.savefig(outdir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    metrics = {'accuracy': acc, 'balanced_accuracy': bacc, 'f1_macro': f1m}
    import json
    with open(outdir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics
