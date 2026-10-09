"""Score raw batch predictions without smoothing or voting."""

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,)
from scripts.evaluation.recog_eval_data import BatchInfo, Trial


# ================================================================
# 1. Write predictions from both modes to one CSV
# ================================================================
# Save each batch's mode, time, trial, phase, truth, and prediction.
# Leave the ground-truth field empty for unlabeled batches.
def write_predictions(path: Path, rows: list[tuple[str, BatchInfo, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("mode", "xdf_time_s", "trial", "phase", "ground_truth", "prediction"))
        for mode, info, prediction in rows:
            writer.writerow((mode, info.time, info.trial, info.phase,
                             info.truth or "", prediction))


# ================================================================
# 2. Print a one-vs-all percentage matrix for each gesture
# ================================================================
# Treat one gesture as positive and all other labels as negative.
# Count true positives, false negatives, false positives,
# and true negatives, then print percentages for each reality row.
def print_gesture_matrices(truth: np.ndarray, predictions: np.ndarray) -> None:
    print("Gesture matrices: positive=this gesture, negative=all other labels")
    print("Each reality row adds up to 100%.")

    for gesture in sorted(set(truth) - {"noGesture"}):
        real_positive = truth == gesture
        predicted_positive = predictions == gesture

        tp = np.count_nonzero(real_positive & predicted_positive)
        fn = np.count_nonzero(real_positive & ~predicted_positive)
        fp = np.count_nonzero(~real_positive & predicted_positive)
        tn = np.count_nonzero(~real_positive & ~predicted_positive)

        print(f"\n{gesture}:")
        print(f"{'Reality / prediction':>22} {'Positive predicted':>20} {'Negative predicted':>20}")
        print(f"{'Positive real':>22} {tp / (tp + fn):>17.1%} TP {fn / (tp + fn):>17.1%} FN")
        print(f"{'Negative real':>22} {fp / (fp + tn):>17.1%} FP {tn / (fp + tn):>17.1%} TN")


# ================================================================
# 3. Count trials where the predicted labels overlap the true gesture (rho > 0.25)
# ================================================================
# Per test trial, A = truly labeled with the trial's gesture batches; B = predicted as that gesture batches. rho = 2*|A and B|/(|A|+|B|), so gaps flicker only lower rho.
def raw_recognition_count(
    trials: list[Trial],
    test_trials: set[int],
    infos: list[BatchInfo],
    predictions: np.ndarray,
) -> int:
    recognized = 0

    for trial in trials:
        if trial.number not in test_trials:
            continue

        pairs = [
            (info.phase == "movement", prediction == trial.label)
            for info, prediction in zip(infos, predictions)
            if info.trial == trial.number
        ]
        true_count = sum(true for true, _ in pairs)
        predicted_count = sum(predicted for _, predicted in pairs)
        both = sum(true and predicted for true, predicted in pairs)

        if true_count + predicted_count and 2*both/(true_count + predicted_count) > 0.25:
            recognized += 1
    
    return recognized

# ================================================================
# 3a. Count how often the prediction flips inside each true gesture
# ================================================================
# Per test trial, take the predictions of its movement batches in time order.
# Count label changes between consecutive batches and divide by the time spanned.
# Return the median over trials in flips per second; lower means a steadier prediction.
def prediction_flips_per_second(
    test_trials: set[int],
    infos: list[BatchInfo],
    predictions: np.ndarray,
) -> float:
    rates = []
    for trial in sorted(test_trials):
        moving = [
            (info.time, prediction)
            for info, prediction in zip(infos, predictions)
            if info.trial == trial and info.phase == "movement"
        ]
        if len(moving) < 2:
            continue
        flips = sum(a[1] != b[1] for a, b in zip(moving, moving[1:]))
        rates.append(flips / (moving[-1][0] - moving[0][0]))
    return float(np.median(rates)) if rates else float("nan")

# ================================================================
# 4. Report classification and raw recognition for one mode
# ================================================================
# Score labeled movement and rest batches for classification.
# Print accuracy, class scores, confusion matrices, and rest alarms.
# Count raw temporal recognition using all batches, even unlabeled.
# Return the main rates for the final mode comparison.
def report_predictions(
    mode: str,
    trials: list[Trial],
    test_trials: set[int],
    infos: list[BatchInfo],
    predictions: np.ndarray,
    window_ms: int,
) -> dict:
    clean = np.asarray([info.clean for info in infos])
    all_truth = np.asarray([info.truth for info in infos])
    truth = all_truth[clean]
    scored_predictions = predictions[clean]
    classes = sorted(set(truth) | set(predictions))
    accuracy = accuracy_score(truth, scored_predictions)
    balanced = balanced_accuracy_score(truth, scored_predictions)
    rest = truth == "noGesture"
    false_gesture_rate = np.mean(scored_predictions[rest] != "noGesture")
    recognized = raw_recognition_count(
        trials, test_trials, infos, predictions
    )
    flips = prediction_flips_per_second(test_trials, infos, predictions)

    print(f"\n{mode}: {len(infos)} test batches ({len(truth)} labeled)")
    print(f"Classification accuracy: {accuracy:.1%}")
    print(classification_report(truth, scored_predictions, labels=classes, zero_division=0))
    print("Confusion matrix (rows=truth, columns=prediction):")
    print(confusion_matrix(truth, scored_predictions, labels=classes))
    print_gesture_matrices(truth, scored_predictions)
    print(f"\nFalse-gesture rate during labeled rest: {false_gesture_rate:.1%}")
    all_rest = all_truth == "noGesture"
    print(f"With transitions ({len(infos)} batches): "
          f"accuracy {accuracy_score(all_truth, predictions):.1%}, "
          f"false-gesture rate {np.mean(predictions[all_rest] != "noGesture"):.1%}"
          )
    print(f"Raw temporal recognition: {recognized}/{len(test_trials)} trials")
    print(f"Prediction flips inside a gesture: {flips:.1f} per second (median over trials)")

    return {
        "labeled_batches": len(truth),
        "classification": accuracy,
        "always_no_gesture": float(np.mean(rest)),
        "balanced": balanced,
        "truth": truth,
        "predictions": scored_predictions,
        "recognition": recognized / len(test_trials),
        "false_gesture": false_gesture_rate,
        "flips_per_second": flips,
    }
