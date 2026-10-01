"""Compare non-overlapping and overlapping raw EMG predictions on one XDF."""

import argparse 
from pathlib import Path

import joblib
import numpy as np

from bmiemg.models.model_factories import DecisionTreeFactory
from bmiemg.preprocessing.features import get_emg_features
from bmiemg.preprocessing.list_features import log_det, mav, maxav, rms, ssc, std, wl

from recog_eval_data import CHANNELS, load_recording, make_batches, split_trials
from recog_eval_metrics import report_predictions, write_predictions


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XDF = ROOT / "Data" / "sub-05_ses-02_task-Side_run-Side_raw.xdf"
# Keep the feature order used by scripts/models/triple_tree.py.
TIME_FEATURES = [mav, std, maxav, rms, wl, ssc, log_det]

# ================================================================
# 1. Record the inputs needed to reuse a saved model
# ================================================================
# Save channels, feature order, window size, sampling rate,
# classes, train_per_code, and max_depth with the model.
# Check these settings when loading to avoid mismatched inputs.
def model_contract(window_ms: int, sfreq: float, classes: list[str],
                   train_per_code: int, max_depth: int | None) -> dict:
    return {
        "channels": CHANNELS,
        "features": tuple(feature.__name__ for feature in TIME_FEATURES),
        "batch_ms": window_ms,
        "sfreq": sfreq,
        "classes": tuple(classes),
        "train_per_code": train_per_code,
        "max_depth": max_depth,
    }

# ================================================================
# 2. Train on non-overlapping batches or load that model
# ================================================================
# Reject saved models whose settings do not match.
# Otherwise, extract 35 features (5 channels x 7) from labeled
# batches and train a decision tree with random_state=42.
# Require gestures and noGesture; save the model if requested.
def get_model(train_batches, train_infos, sfreq, window_ms, train_per_code,
              model_path, save_model, max_depth=None):
    labeled = np.asarray([info.clean for info in train_infos])
    labels = np.asarray([info.truth for info in train_infos if info.clean])
    classes = sorted(set(labels))
    if len(classes) < 2 or "noGesture" not in classes:
        raise ValueError("Training split must contain gestures and noGesture")
    contract = model_contract(window_ms, sfreq, classes, train_per_code, max_depth)

    if model_path is not None:
        artifact = joblib.load(model_path)  # Only load trusted model files.
        if not isinstance(artifact, dict) or artifact.get("contract") != contract:
            raise ValueError("Model contract does not match this recording/setup")
        return artifact["model"]

    features = get_emg_features(train_batches[labeled], sfreq, TIME_FEATURES, [])
    model = DecisionTreeFactory(random_state=42, max_depth=max_depth).create()
    model.fit(features, labels)
    print(f"Training accuracy: {model.score(features,labels):.1%} "
          f"(depth {model.get_depth()}, {model.get_n_leaves()} leaves)")
    if save_model is not None:
        joblib.dump({"model": model, "contract": contract}, save_model)
    return model

# ================================================================
# 3. Evaluate both window steps with the same model
# ================================================================
# Load one XDF and split trials by movement code.
# Train or load a model using non-overlapping training batches.
# Test with a full-window step and/or overlap_step_ms.
# Compare accuracy, recognition, and false-gesture rates.
# Save predictions to CSV if a path is provided.
def evaluate(
    xdf: Path,
    window_ms: int = 100,
    overlap_step_ms: int = 20,
    mode: str = "both",
    train_per_code: int = 4,
    model_path: Path | None = None,
    save_model: Path | None = None,
    csv_path: Path | None = None,
    max_depth: int | None = None,
) -> None:
    if not xdf.is_file():
        raise FileNotFoundError(xdf)
    if window_ms <= 0 or train_per_code <= 0:
        raise ValueError("window_ms and train_per_code must be positive")
    if mode in ("both", "overlap") and not 0 < overlap_step_ms < window_ms:
        raise ValueError("Overlapping step must be between 0 and window size")

    emg, timestamps, sfreq, trials, bounds = load_recording(xdf)
    train_trials, test_trials = split_trials(trials, train_per_code)
    train_batches, train_infos = make_batches(
        emg, timestamps, sfreq, trials, bounds, train_trials, window_ms, window_ms
    )
    model = get_model(
        train_batches, train_infos, sfreq, window_ms, train_per_code,
        model_path, save_model, max_depth,
    )
    del train_batches

    modes = []
    if mode in ("both", "non-overlap"):
        modes.append(("non-overlap", window_ms))
    if mode in ("both", "overlap"):
        modes.append(("overlap", overlap_step_ms))

    print(f"XDF: {xdf}")
    print(f"EMG: {len(CHANNELS)} channels, {sfreq:g} Hz")
    print(f"Trials: {len(train_trials)} train, {len(test_trials)} test")
    print(f"Window: {window_ms} ms; same model in every test mode")

    results = {}
    csv_rows = []
    for mode_name, step_ms in modes:
        batches, infos = make_batches(
            emg, timestamps, sfreq, trials, bounds, test_trials,
            window_ms, step_ms,
        )
        features = get_emg_features(batches, sfreq, TIME_FEATURES, [])
        predictions = model.predict(features)
        results[mode_name] = report_predictions(
            mode_name, trials, test_trials, infos, predictions, window_ms
        )
        if csv_path is not None:
            csv_rows.extend(zip([mode_name] * len(infos), infos, predictions))
        del batches, features

    print("\nComparison (raw predictions, no post-processing):")
    print(f"{'Mode':<16} {'Step':>7} {'Labeled':>8} {'Classification':>15} {'Recognition':>13} {'False gesture':>14} {'Flips/s':>9}")
    for mode_name, step_ms in modes:
        result = results[mode_name]
        print(
            f"{mode_name:<16} {step_ms:>5} ms "
            f"{result['labeled_batches']:>8} "
            f"{result['classification']:>14.1%} "
            f"{result['recognition']:>13.1%} "
            f"{result['false_gesture']:>14.1%} "
            f"{result['flips_per_second']:>9.1f}"
        )
    print("Recognition uses XDF move/return markers, not manual EMG onset labels.")
    print("Overlapping batches are correlated; their counts are not independent samples.")

    if csv_path is not None:
        write_predictions(csv_path, csv_rows)
        print(f"Prediction sequence: {csv_path}")

# ================================================================
# 4. Read command-line options and start the comparison
# ================================================================
# Reject --model and --save-model used together, then evaluate.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xdf", type=Path, default=DEFAULT_XDF)
    parser.add_argument("--window-ms", "--batch-ms", type=int, default=100)
    parser.add_argument("--overlap-step-ms", type=int, default=20)
    parser.add_argument("--mode", choices=("both", "non-overlap", "overlap"), default="both")
    parser.add_argument("--train-per-code", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=None, help="Limit the tree depth (default: no limit)")
    parser.add_argument("--model", type=Path, help="Load a trusted model saved by this script")
    parser.add_argument("--save-model", type=Path, help="Save the newly trained model")
    parser.add_argument("--csv", type=Path, help="Save test predictions from both modes")
    args = parser.parse_args()
    if args.model is not None and args.save_model is not None:
        parser.error("--model and --save-model cannot be used together")
    evaluate(
        args.xdf, args.window_ms, args.overlap_step_ms, args.mode,
        args.train_per_code, args.model, args.save_model, args.csv,
        max_depth=args.max_depth,
    )


if __name__ == "__main__":
    main()
