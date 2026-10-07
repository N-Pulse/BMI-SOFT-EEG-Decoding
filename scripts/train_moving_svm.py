import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from utils import band_features, feature_names, group_importances, feature_importances, save_grouped_importances, feature_ablation, group_ablation, save_ablation, subject_normalize, report, save_confusion, load_bundles, grouped_holdout


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ABLATION_TOP_FEATURES = 20  # features removed one by one in the per-feature ablation plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "results" / "moving_preprocessed")
    parser.add_argument("--band", choices=("1-45", "1-4", "4-8", "8-12", "12-30", "30-45"), default="1-45")
    parser.add_argument("--C", type=float, default=0.01, help="Small C: many correlated features, few subjects")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv", type=int, default=0,
                        help="Also run N-fold trial-grouped cross-validation (0 = off)")
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "results" / "moving_svm")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.data.rglob(f"*_epochs_{args.band}.npz"))
    if not paths:
        raise SystemExit(f"No *_epochs_{args.band}.npz bundles under {args.data}")

    # --- Load and stack subjects ---
    psd, y, subjects, trials, freqs, channels = load_bundles(paths)
    report("all subjects", psd.shape, "epochs x shared channels x frequency bins")

    # --- Spectral features: total power, mean, median and peak frequency per band and channel, flattened ---
    features = band_features(psd, freqs)
    report("features", features.shape, "epochs x channels x sub-bands x spectral features")
    X = features.reshape(len(features), -1)
    names = feature_names(channels, freqs)
    X = subject_normalize(X, subjects)  # per-subject z-scoring, always on
    report("SVM input", X.shape, "epochs x (channels * sub-bands * spectral features)")

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

    # --- Split: whole trials (6 s blocks) are held out, so overlapping windows never straddle train/test ---
    groups = trials
    train_idx, test_idx = grouped_holdout(y, groups, args.test_size, args.seed)

    report("train", X[train_idx].shape, "epochs x features (held-out by trial)")
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

    if args.cv > 1:
        n_splits = min(args.cv, len(np.unique(groups)))

        scores = cross_val_score(
            model,
            X,
            y,
            groups=groups,
            cv=GroupKFold(n_splits=n_splits),
            scoring="balanced_accuracy",
        )

        print(
            f"{n_splits}-fold trial-grouped CV balanced accuracy: "
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

    # --- Feature importance: mean |coef| over the one-vs-rest SVMs, on standardized features ---
    importances = feature_importances(model, X.shape[1])
    order = np.argsort(importances)[::-1]
    print("\nTop 15 features (mean |coefficient|):")
    for i in order[:15]:
        print(f"  {names[i].replace(chr(124), chr(32)):<32s} {importances[i]:.3f}")
    for label, part in (("metric", 2), ("band", 1), ("channel", 0)):
        ranked = group_importances(names, importances, part)
        print(f"By {label}: " + ", ".join(f"{k} {v:.2f}" for k, v in ranked))

    args.outdir.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.outdir / f"feature_importance_{args.band}.csv",
               np.column_stack([np.array(names)[order], importances[order].round(6)]),
               fmt="%s", delimiter=",", header="feature,mean_abs_coef", comments="")
    save_grouped_importances(names, importances, args.outdir, args.band)

    # --- Ablation: retrain while removing the most important features/groups (ranked on training trials only) ---
    Xtr, ytr, Xte, yte = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
    top = [names[i] for i in order[:ABLATION_TOP_FEATURES]]
    scores = feature_ablation(model, Xtr, ytr, Xte, yte, order, ABLATION_TOP_FEATURES)
    save_ablation(top, scores, chance_bacc, args.outdir / f"ablation_features_{args.band}.png",
                  f"top {len(scores) - 1} features")
    for part, unit in ((0, "channels"), (1, "bands"), (2, "metrics")):
        ranked, scores = group_ablation(model, Xtr, ytr, Xte, yte, names, importances, part)
        save_ablation(ranked, scores, chance_bacc, args.outdir / f"ablation_{unit}_{args.band}.png", unit)
        print(f"Balanced accuracy removing {unit} (most important first): "
              + ", ".join(f"{n} {v:.3f}" for n, v in zip(["none"] + ranked, scores)))

    save_confusion(
        cm,
        classes,
        args.outdir / f"confusion_{args.band}.png",
        f"{n_classes}-class state ({args.band} Hz) — Acc {accuracy:.2f} | BA {bacc:.2f}",
    )

    joblib.dump(
        {
            "model": model,
            "classes": classes,
            "channels": channels,
            "features": "band_features",
            "feature_names": names,
            "subject_normalized": True,
        },
        args.outdir / f"model_{args.band}.joblib",
    )

    print(f"Saved model and confusion matrix to {args.outdir}")


if __name__ == "__main__":
    main()
