import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from utils import log_band_power, subject_normalize, report, save_confusion, load_bundles, grouped_holdout


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "artifacts" / "moving_preprocessed")
    parser.add_argument("--band", choices=("1-45", "1-4", "4-8", "8-12", "12-30", "30-45"), default="1-45")
    parser.add_argument("--C", type=float, default=0.01, help="Small C: many correlated features, few subjects")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=("subject", "trial"), default="subject",
                        help="Hold out whole subjects (default) or whole trials; "
                             "falls back to 'trial' when only one subject exists")
    parser.add_argument("--cv", type=int, default=0,
                        help="Also run N-fold grouped cross-validation (0 = off; -1 = leave-one-group-out)")
    parser.add_argument("--no-subject-norm", action="store_true", help="Skip per-subject feature z-scoring")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "artifacts" / "moving_svm")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.data.rglob(f"*_four-class_{args.band}.npz"))
    if not paths:
        raise SystemExit(f"No *_four-class_{args.band}.npz bundles under {args.data}")

    # --- Load and stack subjects ---
    psd, y, subjects, trials, freqs, channels = load_bundles(paths)
    report("all subjects", psd.shape, "epochs x shared channels x frequency bins")

    # --- Spectral features: log band power per channel -> one flat vector per epoch ---
    features = log_band_power(psd, freqs)
    report("features", features.shape, "epochs x channels x sub-bands (log10 power)")
    X = features.reshape(len(features), -1)
    if not args.no_subject_norm:
        X = subject_normalize(X, subjects)
    report("SVM input", X.shape, "epochs x (channels * sub-bands) features")

    classes = sorted(np.unique(y).tolist())
    n_classes = len(classes)
    counts = {c: int((y == c).sum()) for c in classes}
    unique_subjects = sorted(np.unique(subjects).tolist())

    chance_bacc = 1.0 / n_classes

    print(
        f"Classes: {counts}  "
        f"(chance balanced accuracy {chance_bacc:.2f}, "
        f"majority {max(counts.values()) / len(y):.2f}); "
        f"{len(unique_subjects)} subject(s)"
    )

    # --- Split ---
    split = args.split
    if split == "subject" and len(unique_subjects) < 2:
        print("Only one subject found: falling back to --split trial.")
        split = "trial"

    groups = trials if split == "trial" else subjects
    train_idx, test_idx = grouped_holdout(y, groups, args.test_size, args.seed)

    report("train", X[train_idx].shape, f"epochs x features (held-out by {split})")
    report("test", X[test_idx].shape, "epochs x features")

    # --- Train and evaluate ---
    model = make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=args.C,
            class_weight="balanced",
            dual=False,
            max_iter=10000,
            random_state=args.seed,
        ),
    )

    if args.cv > 1 or args.cv == -1:
        n_splits = len(np.unique(groups)) if args.cv == -1 else min(args.cv, len(np.unique(groups)))
        cv = LeaveOneGroupOut() if args.cv == -1 else GroupKFold(n_splits=n_splits)

        scores = cross_val_score(
            model,
            X,
            y,
            groups=groups,
            cv=cv,
            scoring="balanced_accuracy",
        )

        print(
            f"{n_splits}-fold grouped CV balanced accuracy: "
            f"{scores.mean():.3f} ± {scores.std():.3f}"
        )

    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])

    cm = confusion_matrix(y[test_idx], pred, labels=classes)

    accuracy = accuracy_score(y[test_idx], pred)
    bacc = balanced_accuracy_score(y[test_idx], pred)

    print(
        f"\nAccuracy: {accuracy:.3f} | "
        f"Balanced accuracy: {bacc:.3f} | "
    )

    print(classification_report(y[test_idx], pred, labels=classes))

    print("Confusion matrix (rows = true, cols = predicted):")
    print(cm)

    args.outdir.mkdir(parents=True, exist_ok=True)

    save_confusion(
        cm,
        classes,
        args.outdir / f"confusion_{args.band}.png",
        f"4-class ({args.band} Hz) — BA {bacc:.2f}",
    )

    joblib.dump(
        {
            "model": model,
            "classes": classes,
            "channels": channels,
            "features": "log_band_power",
            "subject_normalized": not args.no_subject_norm,
        },
        args.outdir / f"model_{args.band}.joblib",
    )

    print(f"Saved model and confusion matrix to {args.outdir}")


if __name__ == "__main__":
    main()
