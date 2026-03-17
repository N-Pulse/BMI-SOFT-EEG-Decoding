#!/usr/bin/env python3
"""
Minimal training script for a motor-decoder baseline.

Input: .npz or directory of .npz bundles (X, y, [groups]).
Features: bandpower | none.
Models: lda, logreg, lsvm, csp_lda.

Usage (from project root):
  python -m src.train --data EEG_clean/processed/simplified/hand_dir/ --features bandpower --model lda --fs 512
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import joblib
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, StratifiedKFold

from src.models import build_pipeline, load_dataset, evaluate_and_report


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description='Minimal motor-decoder training script')
    p.add_argument('--data', type=str, required=True, help='Path to .npz or directory of .npz bundles')
    p.add_argument('--features', type=str, default='bandpower', choices=['bandpower', 'none'])
    p.add_argument('--model', type=str, default='lda', choices=['lda', 'logreg', 'lsvm', 'csp_lda'])
    p.add_argument('--fs', type=float, default=None, help='Sampling rate (Hz). Required if features=bandpower or csp_lda.')
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--C', type=float, default=1.0, help='Regularization for logreg/lsvm')
    p.add_argument('--csp_components', type=int, default=4, help='CSP components if model=csp_lda')
    p.add_argument('--outdir', type=str, default='artifacts')
    p.add_argument('--ignore_groups', action='store_true', help='Ignore groups and split at epoch level')
    p.add_argument('--cv', type=int, default=0, help='If >0, use StratifiedKFold with K folds when groups < 2.')
    return p.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    outdir = Path(args.outdir)
    _ensure_dir(outdir)

    X, y, groups = load_dataset(data_path)

    fs = args.fs
    is_3d_input = X.ndim == 3
    if is_3d_input and (args.features == 'bandpower' or args.model == 'csp_lda') and fs is None:
        raise ValueError('--fs is required for featureization on raw epochs (3D data).')
    if args.features == 'bandpower' and not is_3d_input:
        raise ValueError('features=bandpower expects raw epochs (n, ch, t). Use --features none for 2D.')

    pipe, feature_names = build_pipeline(args, fs=fs, n_channels=X.shape[1] if is_3d_input else None)

    n_groups = len(np.unique(groups))

    if not args.ignore_groups and n_groups >= 2:
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        tr_idx, te_idx = next(gss.split(X, y, groups))
    elif args.cv and args.cv > 1:
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
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        tr_idx, te_idx = next(sss.split(X, y))

    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]

    if Xtr.shape[0] == 0 or Xte.shape[0] == 0:
        print("Error: Training or testing set is empty after splitting.")
        return

    pipe.fit(Xtr, ytr)
    yhat = pipe.predict(Xte)
    metrics = evaluate_and_report(yte, yhat, outdir)

    joblib.dump(pipe, outdir / 'model.pkl')

    names = None
    try:
        if hasattr(pipe.named_steps.get('bandpower', None), 'feature_names_'):
            names = pipe.named_steps['bandpower'].feature_names_
    except Exception:
        names = None
    if names is not None:
        with open(outdir / 'feature_names.json', 'w') as f:
            json.dump(names, f, indent=2)

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
