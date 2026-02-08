#!/usr/bin/env python3
"""
Minimal training script for a motor-decoder baseline.

Input expectation (pick one):
  A) A single NumPy bundle (.npz) with keys:
       - X: shape (n_epochs, n_channels, n_times) OR (n_epochs, n_features)
       - y: shape (n_epochs,)
       - meta (optional): dict-like or ignored

  B) A directory containing one or more .npz files in the above format.

This keeps us decoupled from how epochs are created. You can export them from
your preprocessing (Melina's code or your XDF pipeline) and drop them here.

Featureization options (choose via --features):
  - bandpower: Welch PSD per epoch → band-integrated log-powers per channel
  - none: assumes X is already features (2D)

Model ladder (choose via --model):
  - lda   : Linear Discriminant Analysis (shrinkage)
  - logreg: Logistic Regression (L2)
  - lsvm  : Linear SVM
  - csp_lda: CSP (per pair or OvR) + LDA  [expects raw epochs; uses band 8–30 Hz]

Riemannian/tangent-space models are great but require pyriemann; easy to add later.

Outputs:
  - artifacts/
      model.pkl, scaler.pkl (if used), config.json, metrics.json,
      confusion_matrix.png, feature_names.json (if available)

Usage examples:
  # Process and train on a simplified hand movement dataset (Assumes fs=512)
  python train_new.py --data NEW_dataset/EEG_clean/processed/simplified/hand_dir/ --features bandpower --model lda --fs 512

"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
from scipy.signal import welch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, StratifiedKFold


# Optional imports for CSP; if unavailable, we fall back gracefully
try:
    from mne.decoding import CSP
    _HAS_MNE = True
except Exception:
    _HAS_MNE = False


# ------------------------------
# Utility
# ------------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_npz_bundle(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    with np.load(path, allow_pickle=True) as npz:
        X = npz['X']
        y = npz['y']
        meta = dict(npz)  # may contain extras
    # Remove X and y keys from meta copy
    meta.pop('X', None)
    meta.pop('y', None)
    return X, y, meta


def load_dataset(data_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load .npz bundles from a file or directory.

    Returns
    -------
    X: np.ndarray  shape (n_epochs, ...)
    y: np.ndarray  shape (n_epochs,)
    groups: np.ndarray  grouping vector for splits (e.g., run index)
            If not provided in files, defaults to one group per file.
    """
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
        # Optional per-epoch groups inside the bundle (e.g., run indices)
        grp = meta.get('groups')
        if grp is None:
            grp = np.full(len(y), gi, dtype=int)  # group by file
        Xs.append(X)
        ys.append(y)
        groups.append(np.asarray(grp))

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    groups = np.concatenate(groups, axis=0)
    return X, y, groups


# ------------------------------
# Featureizers
# ------------------------------
@dataclass
class BandConfig:
    fs: float
    bands: List[Tuple[float, float]]
    nperseg: int = 256
    noverlap: int = 128


class Bandpower(TransformerMixin, BaseEstimator):
    """Compute log band power per channel for each epoch.
    Input X: (n_epochs, n_channels, n_times)
    Output: (n_epochs, n_channels * n_bands)
    """
    def __init__(self, fs: float, bands: List[Tuple[float, float]] | None = None,
                 nperseg: int = 256, noverlap: int = 128, eps: float = 1e-12):
        self.fs = fs
        # ADAPTATION: Use Alpha (8-12Hz) and Beta (13-30Hz) as default motor bands
        self.bands = bands or [(4, 8), (8, 12), (13, 30), (30, 45)]
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.eps = eps
        # derived
        self._freqs = None
        self.feature_names_: List[str] | None = None

    def fit(self, X, y=None):
        # Create names
        n_ch = X.shape[1]
        names = []
        # ADAPTATION: Use channel index since names are not passed to train.py
        for ch in range(n_ch):
            for (lo, hi) in self.bands:
                # Format names clearly for feature selection later
                names.append(f"ch{ch}_bp_{lo:g}-{hi:g}Hz")
        self.feature_names_ = names
        return self

    def transform(self, X):
        if X.ndim != 3:
            raise ValueError("Bandpower expects X with shape (n_epochs, n_channels, n_times)")
        n_epochs, n_ch, _ = X.shape
        out = np.zeros((n_epochs, n_ch * len(self.bands)), dtype=np.float32)
        for i in range(n_epochs):
            # PSD per channel
            for ch in range(n_ch):
                f, Pxx = welch(X[i, ch, :], fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
                for b_idx, (lo, hi) in enumerate(self.bands):
                    mask = (f >= lo) & (f < hi)
                    # Integrate power (area under the curve)
                    bp = np.trapz(Pxx[mask], f[mask])
                    out[i, ch*len(self.bands) + b_idx] = np.log(bp + self.eps)
        return out


class FlattenIfNeeded(TransformerMixin, BaseEstimator):
    """If X is (n, c, t), flatten to (n, c*t). If already 2D, pass through."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if X.ndim == 3:
            n, c, t = X.shape
            return X.reshape(n, c*t)
        return X


# ------------------------------
# CSP wrapper pipeline 
# ------------------------------
class CSPFeatures(TransformerMixin, BaseEstimator):
    """CSP features (log-variance). Requires MNE."""
    def __init__(self, n_components: int = 4, reg: str | float | None = 'ledoit_wolf', l_freq: float = 8.0, h_freq: float = 30.0, sfreq: float | None = None):
        if not _HAS_MNE:
            raise ImportError("mne not available. Install MNE to use CSP.")
        self.n_components = n_components
        self.reg = reg
        # ADAPTATION: Use standard motor ERD/ERS band for filtering
        self.l_freq = 8.0
        self.h_freq = 30.0
        self.sfreq = sfreq
        # The MNE CSP object handles the transformation internally
        self._csp = CSP(n_components=n_components, reg=reg, log=True, cov_est='concat')

    def fit(self, X, y):
        if X.ndim != 3:
            raise ValueError("CSP expects raw epochs: (n_epochs, n_channels, n_times)")
        # CSP filter implicitly acts as a feature extractor here
        self._csp.fit(X, y)
        return self

    def transform(self, X):
        return self._csp.transform(X)


# ------------------------------
# Modeling
# ------------------------------

def build_pipeline(args, fs: float | None, n_channels: int | None) -> Tuple[Pipeline, List[str] | None]:
    feature_names = None

    # Feature stage
    feature_steps = []
    if args.model == 'csp_lda':
        # Check for MNE again, just in case
        if not _HAS_MNE:
             raise ImportError("mne not available. Install MNE to use CSP.")
        
        # CSP expects raw 3D data and performs its own filtering/feature extraction
        feat = CSPFeatures(n_components=args.csp_components)
        feature_steps.append(('csp', feat))
    else:
        if args.features == 'bandpower':
            if fs is None:
                raise ValueError("--fs is required when using bandpower on raw epochs")
            bp = Bandpower(fs=fs)
            feature_steps.append(('bandpower', bp))
            feature_names = bp.feature_names_  # will be available after fit
        elif args.features == 'none':
            feature_steps.append(('flatten', FlattenIfNeeded()))
        else:
            raise ValueError(f"Unknown features: {args.features}")

    # Classifier stage
    if args.model == 'lda' or args.model == 'csp_lda':
        # LDA with automatic shrinkage is a robust default for BCI
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    elif args.model == 'logreg':
        # LogReg with balanced class weight to handle unequal trial counts
        clf = LogisticRegression(max_iter=500, n_jobs=None, C=args.C, class_weight='balanced')
    elif args.model == 'lsvm':
        # Linear SVM with balanced class weight
        clf = LinearSVC(C=args.C, class_weight='balanced')
    else:
        raise ValueError(f"Unknown model: {args.model}")

    steps = []
    if feature_steps:
        steps.extend(feature_steps)
    # Scale if features are not already standardized; essential for linear models
    steps.append(('scaler', StandardScaler()))
    steps.append(('clf', clf))

    pipe = Pipeline(steps)
    return pipe, feature_names


# ------------------------------
# Evaluation / plotting
# ------------------------------

def evaluate_and_report(y_true, y_pred, outdir: Path) -> Dict[str, float]:
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    # Macro F1 is useful for multi-class/imbalanced data
    f1m = f1_score(y_true, y_pred, average='macro')

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(4,4))
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
    with open(outdir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics


# ------------------------------
# Main
# ------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Minimal motor-decoder training script')
    p.add_argument('--data', type=str, required=True, help='Path to .npz or a directory of .npz bundles (often the simplified/ folder).')
    p.add_argument('--features', type=str, default='bandpower', choices=['bandpower', 'none'], help='Featureization method')
    p.add_argument('--model', type=str, default='lda', choices=['lda', 'logreg', 'lsvm', 'csp_lda'], help='Classifier / pipeline')
    p.add_argument('--fs', type=float, default=None, help='Sampling rate (Hz). Required if features=bandpower or csp_lda, e.g., 512.')
    p.add_argument('--test_size', type=float, default=0.2, help='Proportion for test split (GroupShuffleSplit)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--C', type=float, default=1.0, help='Regularization for logreg/lsvm')
    p.add_argument('--csp_components', type=int, default=4, help='Number of CSP components if model=csp_lda')
    p.add_argument('--outdir', type=str, default='artifacts', help='Output directory')
    p.add_argument('--ignore_groups', action='store_true', help='Ignore groups and split at epoch level (stratified).')
    p.add_argument('--cv', type=int, default=0, help='If >0, use StratifiedKFold with K folds when groups < 2.')
    return p.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    outdir = Path(args.outdir)
    _ensure_dir(outdir)

    X, y, groups = load_dataset(data_path)

    # Sanity check for sampling rate if 3D data is provided
    fs = args.fs
    is_3d_input = X.ndim == 3
    if is_3d_input and (args.features == 'bandpower' or args.model == 'csp_lda') and fs is None:
        raise ValueError('The --fs argument (sampling rate) is required for featureization on raw epochs (3D data).')
    if args.features == 'bandpower' and not is_3d_input:
        raise ValueError('features=bandpower expects raw epochs (n, ch, t). Got 2D features. Use --features none.')

    pipe, feature_names = build_pipeline(args, fs=fs, n_channels=X.shape[1] if is_3d_input else None)

    # ---------- Splitting logic ----------
    # Prefer GroupShuffleSplit to prevent data leakage across runs/sessions
    n_groups = len(np.unique(groups))

    if not args.ignore_groups and n_groups >= 2:
        # Preferred: group-wise split across runs/files
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        tr_idx, te_idx = next(gss.split(X, y, groups))
    elif args.cv and args.cv > 1:
        # Cross-validation within a single group/run
        skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
        metrics_list = []
        for tr_idx, te_idx in skf.split(X, y):
            pipe.fit(X[tr_idx], y[tr_idx])
            yhat = pipe.predict(X[te_idx])
            metrics_list.append(evaluate_and_report(y[te_idx], yhat, outdir))
        avg = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0].keys()}
        with open(outdir / 'metrics_cv.json', 'w') as f:
            json.dump({'folds': args.cv, 'avg': avg, 'per_fold': metrics_list}, f, indent=2)
        print("CV metrics:", json.dumps(avg, indent=2))
        print(f"Artifacts saved to: {outdir.resolve()}")
        return
    else:
        # Fallback: epoch-level stratified hold-out (ignores groups)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        tr_idx, te_idx = next(sss.split(X, y))

    # Train/evaluate once with the chosen split
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]
    
    # Check for empty split
    if Xtr.shape[0] == 0 or Xte.shape[0] == 0:
        print("Error: Training or testing set is empty after splitting. Check --test_size and group distribution.")
        return

    pipe.fit(Xtr, ytr)
    yhat = pipe.predict(Xte)
    metrics = evaluate_and_report(yte, yhat, outdir)


    # Save artifacts
    joblib.dump(pipe, outdir / 'model.pkl')

    # Capture feature names if available (after fit, bandpower fills names)
    names = None
    try:
        # Look for bandpower step's feature names
        if hasattr(pipe.named_steps.get('bandpower', None), 'feature_names_'):
            names = pipe.named_steps['bandpower'].feature_names_
    except Exception:
        names = None
    if names is not None:
        with open(outdir / 'feature_names.json', 'w') as f:
            json.dump(names, f, indent=2)

    # Save config
    config = vars(args).copy()
    config['n_samples'] = int(len(y))
    config['n_train'] = int(len(ytr))
    config['n_test'] = int(len(yte))
    with open(outdir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("Metrics:", json.dumps(metrics, indent=2))
    print(f"Artifacts saved to: {outdir.resolve()}")


if __name__ == '__main__':
    main()