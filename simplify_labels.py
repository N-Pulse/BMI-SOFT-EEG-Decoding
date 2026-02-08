#!/usr/bin/env python3
"""
Simplify multi-class EEG labels (6 classes) into binary or grouped versions.
Each simplified file is saved into a subdirectory named after the chosen mode.

Usage:
  python simplify_labels.py --data dataset/EEG_clean/processed --mode elbow
"""

from pathlib import Path
import numpy as np
import argparse

# ------------------------------------------------------------
# Define label grouping modes
# ------------------------------------------------------------
MAPS = {
    "elbow": {
        "flexion":  [1],
        "extension": [2],
    },
    "forearm": {
        "supination": [3],
        "pronation":  [4],
    },
    "hand": {
        "close": [5],
        "open":  [6],
    },
}


# ------------------------------------------------------------
# Simplify function
# ------------------------------------------------------------
def simplify_npz(path: Path, mapping: dict[str, list[int]], mode: str = "") -> Path:
    """Load .npz, remap labels, and save in a subdirectory for that mode."""
    with np.load(path, allow_pickle=True) as npz:
        X = npz["X"]
        y = npz["y"]
        groups = npz.get("groups", np.zeros(len(y), int))

    # Build new label vector
    y_new = np.full_like(y, fill_value=-1)
    for new_label, old_ids in enumerate(mapping.values()):
        mask = np.isin(y, old_ids)
        y_new[mask] = new_label

    valid_mask = y_new != -1
    X, y_new, groups = X[valid_mask], y_new[valid_mask], groups[valid_mask]

    # --------------------------------------------------------
    # Save in subdirectory: e.g. processed/simplified/elbow/
    # --------------------------------------------------------
    mode_dir = path.parent / "simplified" / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    out_path = mode_dir / (path.stem + f"_{mode}.npz")
    np.savez(out_path, X=X, y=y_new, groups=groups)

    # Print summary
    unique, counts = np.unique(y_new, return_counts=True)
    print(f"✅ Saved {out_path}: {X.shape[0]} samples ({len(unique)} classes)")

    for label_name, class_id in zip(mapping.keys(), unique):
        print(f"   Class {class_id}: {label_name} → {counts[class_id]} samples")

    return out_path


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Simplify EEG labels into binary or grouped versions")
    parser.add_argument("--data", type=str, required=True, help="Path to .npz file or directory")
    parser.add_argument("--mode", type=str, default="elbow", choices=list(MAPS.keys()), help="Grouping mode")
    args = parser.parse_args()

    mapping = MAPS[args.mode]
    data_path = Path(args.data)

    if data_path.is_file() and data_path.suffix == ".npz":
        simplify_npz(data_path, mapping, mode=args.mode)
    else:
        for npz_file in sorted(data_path.glob("*.npz")):
            simplify_npz(npz_file, mapping, mode=args.mode)


if __name__ == "__main__":
    main()
