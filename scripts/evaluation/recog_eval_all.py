"""
Evaluates a model on every MOVING person and prints one table per mode.
Each row is one person, the last row is the mean over all people.
"""
# ================================================================
# 0. Section: IMPORTS
# ================================================================
import argparse
import contextlib
import io

import numpy as np

from scripts.evaluation.recog_eval_offline import MODELS, evaluate
from scripts.treeoftrees.moving_data import DATA
from scripts.evaluation.recog_eval_metrics import print_gesture_matrices

from sklearn.metrics import confusion_matrix


# ================================================================
# 1. Section: INPUTS
# ================================================================
MODES: list[str] = ["non-overlap", "overlap"]

# Columns of the tables: title, key in the results and how to print it
COLUMNS: list[tuple[str, str, str]] = [
    ("Classification", "classification", "{:.1%}"),
    ("Always noGesture", "always_no_gesture", "{:.1%}"),
    ("Balanced", "balanced", "{:.1%}"),
    ("Recognition", "recognition", "{:.1%}"),
    ("False gesture", "false_gesture", "{:.1%}"),
    ("Flips/s", "flips_per_second", "{:.1f}"),
]

LEGEND: str = """
Classification:   share of correct answers
Always noGesture: the same score for a model that never reacts (to beat)
Balanced:         the share found in each class, averaged (random = 25%)
Recognition:      share of the test movements that were found
False gesture:    share of rest windows where a movement was invented
Flips/s:          how often the answer changes inside a movement (lower = steadier)"""


# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def evaluate_everyone(model_name: str) -> dict[str, dict[int, dict]]:
    """Evaluate every person and keep only the numbers of each mode."""
    results: dict[str, dict[int, dict]] = {mode: {} for mode in MODES}
    for path in sorted(DATA.glob("*.edf")):
        subject = int(path.stem.split("_")[2])
        # The long report of each person is hidden, only the numbers are kept
        with contextlib.redirect_stdout(io.StringIO()):
            person = evaluate(path, model_name=model_name)
        for mode in MODES:
            results[mode][subject] = person[mode]
    return results


def print_row(label: str | int, cells: list[str], widths: list[int]) -> None:
    """Print a label and its cells, every cell right-aligned in its column."""
    line = "".join(f"{cell:>{width}}" for cell, width in zip(cells, widths))
    print(f"{label:<8}{line}")


def print_table(mode: str, rows: dict[int, dict]) -> None:
    """Print one row per person and the mean over all people."""
    widths = [len(title) + 3 for title, _, _ in COLUMNS]

    print(f"\n{mode}")
    print_row("Person", [title for title, _, _ in COLUMNS], widths)
    for subject, row in rows.items():
        print_row(subject, [fmt.format(row[key]) for _, key, fmt in COLUMNS], widths)

    means = []
    for _, key, fmt in COLUMNS:
        means.append(fmt.format(np.nanmean([row[key] for row in rows.values()])))
    print_row("Mean", means, widths)

def print_gestures(rows: dict[int, dict]) -> None:
    """Print the confusion matrix and the TP/FP tables for all people together."""
    truth = np.concatenate([row["truth"] for row in rows.values()])
    predictions = np.concatenate([row["predictions"] for row in rows.values()])
    classes = ["noGesture"] + sorted((set(truth) | set(predictions)) - {"noGesture"})
    matrix = confusion_matrix(truth, predictions, labels=classes)

    print("\nConfusion matrix, all people together (non-overlap)")
    print("Rows: what really happened. Columns: the answer. Each row adds up to 100%.")
    header = "".join(f"{name:>16}" for name in classes)
    print(f"{'':<16}{header}{'windows':>10}")
    for name, counts in zip(classes, matrix):
        cells = "".join(f"{count / counts.sum():>16.1%}" for count in counts)
        print(f"{name:<16}{cells}{counts.sum():>10}")

    print()
    print_gesture_matrices(truth, predictions)

# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    # 1. Read which model to evaluate
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS), default="treeoftrees")
    args = parser.parse_args()

    # 2. Evaluate all people
    results = evaluate_everyone(args.model)

    # 3. Print one table per mode and explain the columns
    print(f"Model: {args.model}")
    for mode in MODES:
        print_table(mode, results[mode])
    print(LEGEND)
    print_gestures(results["non-overlap"])
