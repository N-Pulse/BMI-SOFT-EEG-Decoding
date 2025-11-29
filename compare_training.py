#!/usr/bin/env python3
"""
Compare multiple training configurations on the same EEG/EMG dataset (.npz or directory).

Automatically evaluates combinations of:
  • Featureization:  none  |  bandpower
  • Models:          lda, logreg, lsvm, csp_lda

Results:
  comparisons/<mode>/summary_<timestamp>.csv
  comparisons/<mode>/<model>_<features>/metrics.json etc.

Usage:
  python compare_training.py --data NEW_dataset/EEG_clean/processed/simplified/hand_dir --fs 300 --base_outdir comparisons/hand_dir
"""

from pathlib import Path
import json
import shutil
import subprocess
import argparse
import pandas as pd
import itertools
import time
import re


# ------------------------------------------------
# Argument parser
# ------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Compare different training configurations")
    p.add_argument('--data', type=str, required=True, help='Path to .npz or directory of .npz bundles')
    p.add_argument('--fs', type=float, default=None, help='Sampling rate (Hz) — required if using bandpower.')
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--cv', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--base_outdir', type=str, default='comparisons', help='Root folder for results')
    return p.parse_args()


# ------------------------------------------------
# Comparison logic
# ------------------------------------------------
def main():
    args = parse_args()
    data_path = Path(args.data)

    # --- Detect mode name from filename ---
    # Example filename: sub-P019_ses-S001_task-Default_run-003_eeg_epochs_hand_binary.npz
    mode_match = re.search(r'(hand|leftright|elbow|forearm)', data_path.stem)
    mode_name = mode_match.group(1) if mode_match else "default"

    base_outdir = Path(args.base_outdir) / mode_name
    base_outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n🧠 Detected mode: '{mode_name}'")
    print(f"📂 Output directory: {base_outdir.resolve()}\n")

    configs = list(itertools.product(
        #['none', 'bandpower'],             # feature methods
        #['lda', 'logreg', 'lsvm', 'csp_lda']  # models
        ['none', 'bandpower'],             # feature methods
        ['logreg', 'lsvm', 'csp_lda']  # models
    ))

    summary = []
    start_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    summary_path = base_outdir / f"summary_{mode_name}_{start_time}.csv"

    print(f"🚀 Running {len(configs)} training configurations...\n")

    for feat, model in configs:
    # Skip duplicate CSP+none (since we force it to bandpower anyway)
        if model == 'csp_lda' and feat == 'none':
            continue

        if model == 'csp_lda':
            feature_choice = 'bandpower'  # CSP always uses bandpower
        else:
            feature_choice = feat

        run_name = f"{model}_{feature_choice}"
        outdir = base_outdir / run_name
        if outdir.exists():
            shutil.rmtree(outdir)
        outdir.mkdir(parents=True)

        print(f"\n--- Training {run_name} ---")

        cmd = [
            'python', 'train.py',
            '--data', str(args.data),
            '--features', feature_choice,
            '--model', model,
            '--outdir', str(outdir),
            '--test_size', str(args.test_size),
            '--seed', str(args.seed)
        ]
        if args.fs:
            cmd += ['--fs', str(args.fs)]
        if args.cv > 0:
            cmd += ['--cv', str(args.cv)]

        # Run training
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed: {run_name}\n{result.stderr}")
            summary.append({
                "mode": mode_name,
                "model": model,
                "features": feature_choice,
                "status": "failed",
                "accuracy": None,
                "balanced_accuracy": None,
                "f1_macro": None
            })
            continue

        # Parse metrics
        metrics_file = outdir / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            summary.append({
                "mode": mode_name,
                "model": model,
                "features": feature_choice,
                **metrics,
                "status": "ok"
            })
        else:
            summary.append({
                "mode": mode_name,
                "model": model,
                "features": feature_choice,
                "status": "no_metrics",
                "accuracy": None,
                "balanced_accuracy": None,
                "f1_macro": None
            })

    # --- Save results as CSV ---
    df = pd.DataFrame(summary)
    df.to_csv(summary_path, index=False)

    print("\n✅ Comparison complete.")
    print(f"Results saved to: {summary_path}")
    print(df)


if __name__ == "__main__":
    main()
