#!/usr/bin/env python3
"""
Evaluation entry point: run model comparison (single mode or all modes).

Usage (from project root):
  # Compare on a single dataset/mode
  python -m src.eval --data EEG_clean/processed/simplified/hand_dir --fs 300 --base_outdir comparisons

  # Run all comparisons across hand_dir and wrist_dir, then aggregate
  python -m src.eval --all --cv 5
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Evaluate / compare models")
    parser.add_argument("--all", action="store_true", help="Run all modes (hand_dir, wrist_dir) and aggregate")
    parser.add_argument("--data", type=str, help="Path to .npz or dir (for single comparison)")
    parser.add_argument("--fs", type=str, default="300", help="Sampling rate (Hz)")
    parser.add_argument("--base_outdir", type=str, default="comparisons")
    parser.add_argument("--cv", type=int, default=5)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    if args.all:
        cmd = [
            sys.executable, "-m", "src.evaluation.run_all_comparisons",
            "--base", "all",
            "--cv", str(args.cv),
        ]
        subprocess.run(cmd, cwd=project_root, check=True)
        return

    if not args.data:
        parser.error("Provide --data for single comparison, or --all for all modes")

    cmd = [
        sys.executable, "-m", "src.evaluation.compare_training",
        "--data", args.data,
        "--fs", args.fs,
        "--base_outdir", args.base_outdir,
        "--cv", str(args.cv),
    ]
    subprocess.run(cmd, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
