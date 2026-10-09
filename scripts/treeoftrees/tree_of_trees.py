"""
Trains the TreeOfTrees on the MOVING EEG windows and tests it on other people.

The TreeOfTrees is a chain of yes/no decision trees, one per class. Each tree only
answers "is this window my class?". To classify a window the trees are asked one
after the other and the first "yes" wins. If every tree says "no", the answer is
"noGesture" (do nothing).
"""
# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import Self

import numpy as np
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

from scripts.treeoftrees.features import get_features
from scripts.treeoftrees.moving_data import DATA, cut_windows, load_subject


# ================================================================
# 1. Section: INPUTS
# ================================================================
# Trees: the order in which they are asked (the first ones decide most often)
TREE_ORDER: list[str] = ["noGesture", "open_close", "wrist_rotation", "finger_tapping"]
FALLBACK: str = "noGesture"
MAX_DEPTH: int | None = 5
RANDOM_STATE: int = 42

# People that are only used for testing, the others are used for training
TEST_SUBJECTS: list[int] = [9, 10, 11]


# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
class TreeOfTrees:
    """A chain of yes/no decision trees, one per class."""

    def __init__(self) -> None:
        self.trees: dict[str, DecisionTreeClassifier] = {}

    def fit(self, features: np.ndarray, labels: np.ndarray) -> Self:
        """Train every tree on its own question: this class (True) or not (False)."""
        for name in TREE_ORDER:
            tree = DecisionTreeClassifier(
                max_depth=MAX_DEPTH, class_weight="balanced", random_state=RANDOM_STATE
            )
            self.trees[name] = tree.fit(features, labels == name)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Ask the trees in order and keep the first "yes" of every window."""
        # 1. Ask every tree about every window: a table of yes/no (windows x trees)
        answers = np.array([self.trees[name].predict(features) for name in TREE_ORDER])

        # 2. For every window take the first tree that said yes
        predictions = []
        for row in answers.T:
            yes = np.flatnonzero(row)
            predictions.append(TREE_ORDER[yes[0]] if len(yes) else FALLBACK)
        return np.array(predictions)


def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all people: the features and the label of every window, and its person."""
    features, labels, subjects = [], [], []
    for path in sorted(DATA.glob("*.edf")):
        windows, window_labels = cut_windows(load_subject(path))
        features.append(get_features(windows))
        labels.append(window_labels)
        subjects.append(np.full(len(window_labels), int(path.stem.split("_")[2])))
    return np.concatenate(features), np.concatenate(labels), np.concatenate(subjects)


# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    # 1. Load all people and split them: some to train, the others to test
    features, labels, subjects = load_dataset()
    is_test = np.isin(subjects, TEST_SUBJECTS)
    print(f"Train: {(~is_test).sum()} windows | test: {is_test.sum()} windows")

    # 2. Train the trees on the training people only
    model = TreeOfTrees().fit(features[~is_test], labels[~is_test])

    # 3. Predict the test people and compare with what they really did
    predictions = model.predict(features[is_test])
    truth = labels[is_test]
    baseline = max((truth == name).mean() for name in TREE_ORDER)
    print(f"Accuracy: {(predictions == truth).mean():.1%}")
    print(f"Always answering the most common class: {baseline:.1%}")
    print(classification_report(truth, predictions, zero_division=0))