# run_all comparisons
"""
Run compare_training.py across all label modes
either using all preprocessed datasets or a single base dataset.

Each mode’s results are saved under:
    comparisons/<mode>/

A combined summary of all experiments is saved as:
    comparisons/all_modes_summary.csv

Run with:
  python run_all_comparisons.py --base all
"""

import subprocess
from pathlib import Path
import pandas as pd
import glob
import argparse

# ------------------------------------------------
# Configuration
# ------------------------------------------------
parser = argparse.ArgumentParser(description="Run comparisons using all or one base dataset")
parser.add_argument("--base", type=str, default="all",
                    help="'all' to use all processed datasets, or path to one base .npz file")
parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds")
args = parser.parse_args()

modes = ["hand_dir", "wrist_dir"]
base_data = Path("EEG_clean/processed")
base_out = Path("comparisons")
FS = "300"  # Hz

# ------------------------------------------------
# Case 1: SINGLE base dataset
# ------------------------------------------------
if args.base != "all":
    base_path = Path(args.base)
    if not base_path.exists():
        raise FileNotFoundError(f"❌ Base dataset not found: {base_path}")

    print(f"\n🎯 Using single base dataset: {base_path.name}")

    # Run simplify_labels.py for all modes
    for mode in modes:
        subprocess.run([
            "python", "simplify_labels_new.py",
            "--data", str(base_path),
            "--mode", mode
        ], check=True)

    # After simplification, run compare_training on those generated files
    for mode in modes:
        data_path = base_data / "simplified" / mode
        print(f"\n🚀 Running compare_training_new.py for mode: {mode}")
        subprocess.run([
            "python", "compare_training_new.py",
            "--data", str(data_path),
            "--fs", FS,
            "--base_outdir", str(base_out),
            "--cv", str(args.cv)
        ])

# ------------------------------------------------
# Case 2: ALL processed datasets
# ------------------------------------------------
else:
    print("\n🧠 Running across ALL processed datasets...")
    for mode in modes:
        data_path = base_data / "simplified" / mode
        print(f"\n🚀 Running compare_training_new.py for mode: {mode}")
        subprocess.run([
            "python", "compare_training_new.py",
            "--data", str(data_path),
            "--fs", FS,
            "--base_outdir", str(base_out),
            "--cv", str(args.cv)
        ])

# ------------------------------------------------
# Aggregate all summaries into one CSV
# ------------------------------------------------
print("\n📊 Aggregating all mode summaries...")

csv_files = glob.glob(str(base_out / "*" / "summary_*.csv"))
if not csv_files:
    print("⚠️  No summary CSVs found — check if compare_training.py ran successfully.")
else:
    all_dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        mode_name = Path(file).parent.name
        if "mode" not in df.columns:
            df.insert(0, "mode", mode_name)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(base_out / "all_modes_summary.csv", index=False)
    print(f"✅ Combined summary saved → {base_out / 'all_modes_summary.csv'}")

    print("\n--- Summary preview ---")
    print(combined.head())
