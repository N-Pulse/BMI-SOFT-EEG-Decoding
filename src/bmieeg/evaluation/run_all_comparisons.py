# run_all comparisons
"""
Run compare_training across all label modes using all preprocessed datasets or a single base dataset.

Each mode's results: comparisons/<mode>/
Combined summary: comparisons/all_modes_summary.csv

Usage (from project root):
  python -m src.evaluation.run_all_comparisons --base all
  python -m src.evaluation.run_all_comparisons --base path/to/single.npz --cv 5
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd
import glob
import argparse

parser = argparse.ArgumentParser(description="Run comparisons using all or one base dataset")
parser.add_argument("--base", type=str, default="all",
                    help="'all' to use all processed datasets, or path to one base .npz file")
parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds")
args = parser.parse_args()

modes = ["hand_dir", "wrist_dir"]
base_data = Path("EEG_clean/processed")
base_out = Path("comparisons")
FS = "300"

# Project root (parent of src)
project_root = Path(__file__).resolve().parent.parent.parent


def run_cmd(cmd, description):
    r = subprocess.run(cmd, cwd=project_root)
    if r.returncode != 0:
        raise SystemExit(f"{description} failed with return code {r.returncode}")


if args.base != "all":
    base_path = Path(args.base)
    if not base_path.exists():
        raise FileNotFoundError(f"❌ Base dataset not found: {base_path}")

    print(f"\n🎯 Using single base dataset: {base_path.name}")

    for mode in modes:
        run_cmd([
            sys.executable, "-m", "src.data.simplify_labels",
            "--data", str(base_path),
            "--mode", mode
        ], "simplify_labels")

    for mode in modes:
        data_path = base_data / "simplified" / mode
        print(f"\n🚀 Running compare_training for mode: {mode}")
        run_cmd([
            sys.executable, "-m", "src.evaluation.compare_training",
            "--data", str(data_path),
            "--fs", FS,
            "--base_outdir", str(base_out),
            "--cv", str(args.cv)
        ], "compare_training")
else:
    print("\n🧠 Running across ALL processed datasets...")
    for mode in modes:
        data_path = base_data / "simplified" / mode
        print(f"\n🚀 Running compare_training for mode: {mode}")
        run_cmd([
            sys.executable, "-m", "src.evaluation.compare_training",
            "--data", str(data_path),
            "--fs", FS,
            "--base_outdir", str(base_out),
            "--cv", str(args.cv)
        ], "compare_training")

# Aggregate summaries
print("\n📊 Aggregating all mode summaries...")

csv_files = glob.glob(str(base_out / "*" / "summary_*.csv"))
if not csv_files:
    print("⚠️  No summary CSVs found — check if compare_training ran successfully.")
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
