#!/usr/bin/env python3
"""
Simplify multi-class EEG labels (6 classes) into binary or grouped versions.
Each simplified file is saved into a subdirectory named after the chosen mode.

Usage:
  python simplify_labels.py --data EEG_clean/processed --mode hand_dir
"""

from pathlib import Path
import numpy as np
import argparse

# ------------------------------------------------------------
# Define label grouping modes
# ------------------------------------------------------------
# The keys (1-6) must match the final integer event IDs from processing_new.py:
# 1: hand_open, 2: hand_close, 3: wrist_flexion, 4: wrist_extension, 5: grasp, 6: pinch

MAPS = {
    # 1. Classification based on WRIST MOVEMENT direction
    "wrist_dir": {
        "flexion":   [3],  # wrist_flexion
        "extension": [4],  # wrist_extension
    },
    
    # 2. Classification based on GROSS HAND MOVEMENT direction
    "hand_dir": {
        "open":      [1],  # hand_open
        "close":     [2],  # hand_close
    },
    
    # 3. Classification based on FINE GRASP TYPE
    "fine_type": {
        "fine_open":     [5],  # fine_open
        "fine_closed":     [6],  # fine_closed
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
    
    # Note: enumerate(mapping.values()) sets the new label ID (0, 1, 2, ...)
    # based on the order of the dictionary keys.
    for new_label, old_ids in enumerate(mapping.values()):
        mask = np.isin(y, old_ids)
        y_new[mask] = new_label

    # Filter out epochs whose original ID was not in the mapping (y_new == -1)
    valid_mask = y_new != -1
    X, y_new, groups = X[valid_mask], y_new[valid_mask], groups[valid_mask]

    # Handle case where the mapping results in no valid epochs
    if X.shape[0] == 0:
        print(f"⚠️ Skipped {path.name}: No valid epochs found for mode '{mode}'.")
        return path

    # --------------------------------------------------------
    # Save in subdirectory: e.g. processed/simplified/mode/
    # --------------------------------------------------------
    mode_dir = path.parent / "simplified" / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    out_path = mode_dir / (path.stem + f"_{mode}.npz")
    np.savez(out_path, X=X, y=y_new, groups=groups)

    # Print summary
    unique, counts = np.unique(y_new, return_counts=True)
    print(f"✅ Saved {out_path}: {X.shape[0]} samples ({len(unique)} classes)")

    label_names = list(mapping.keys())
    for class_id, count in zip(unique, counts):
        if class_id < len(label_names):
            print(f"   Class {class_id}: {label_names[class_id]} → {count} samples")
        else:
            print(f"   Class {class_id}: Unknown Label → {count} samples")

    return out_path


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Simplify EEG labels into binary or grouped versions")
    parser.add_argument("--data", type=str, required=True, help="Path to .npz file or directory")
    parser.add_argument("--mode", type=str, default="hand_dir", choices=list(MAPS.keys()), help="Grouping mode")
    args = parser.parse_args()

    mapping = MAPS[args.mode]
    data_path = Path(args.data)
    print(f"Starting simplification in mode: '{args.mode}'")

    if data_path.is_file() and data_path.suffix == ".npz":
        simplify_npz(data_path, mapping, mode=args.mode)
    else:
        for npz_file in sorted(data_path.glob("*.npz")):
            simplify_npz(npz_file, mapping, mode=args.mode)


if __name__ == "__main__":
    main()