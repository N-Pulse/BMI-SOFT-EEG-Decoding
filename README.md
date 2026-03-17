## 📄 README: Motor Decoder Signal Processing Pipeline

This repository contains the pipeline for processing EEG data and developing classifiers for prosthetic hand control.

### Project Objective

The primary goal is to transform raw EEG signals into machine learning-ready features to classify distinct motor intentions for potential use in a Brain-Computer Interface (BCI).

-----

### ⚙️ Pipeline Overview

The project is organised following a [Lightning-Hydra–style template](https://github.com/ashleve/lightning-hydra-template) layout:

| Folder        | Role |
|---------------|------|
| **`src/data/`**   | Data preprocessing: **`processing.py`** (signal preprocessing, filtering, ICA, epochs) and **`simplify_labels.py`** (binary/specialised label modes). |
| **`src/models/`** | Training pipeline: featureizers (Bandpower, CSP), classifiers (LDA, LogReg, SVM), evaluation. |
| **`src/evaluation/`** | Model comparison: **`compare_training.py`** (multi-config comparison), **`run_all_comparisons.py`** (all modes + aggregate). |
| **`configs/`**  | Hydra/experiment configs (optional). |
| **`data/`**    | Project data (e.g. raw inputs). |
| **`logs/`**    | Training/eval logs and checkpoints. |
| **`notebooks/`** | Jupyter notebooks. |
| **`tests/`**   | Tests (TODO: add unit and integration tests). |

Workflow:

1. **Preprocessing** — `src/data/processing.py`: signal preprocessing, filtering, referencing, ICA; LSL markers → integer labels; QC plots.
2. **Label simplification** — `src/data/simplify_labels.py`: reduce to binary or specialised tasks (e.g. Hand Open vs Hand Close).
3. **Training** — `src/train.py`: feature extraction (Bandpower, CSP) and linear classifiers (LDA, LogReg, SVM) with optional cross-validation.
4. **Evaluation** — `src/eval.py` / `src/evaluation/`: compare multiple feature/model configs and aggregate results.

-----

### 📂 Data & Input

This repository contains data from multiple acquisition phases, including 2024 data (`archives/2024/EEG`), and more recent 2025 data (`sub-P005/` which should be at same level as the processing files). 

  * **Input Data:** Raw `.xdf` files are organized by subject and session.
  * **Output Data:** Cleaned epochs (`.npz`) and processed metadata are saved into the `EEG_clean/` directory.



-----

###  Getting Started

Run all commands from the **project root**.

1. **Preprocessing** (from directory containing `.xdf` files, e.g. `sub-P005`):
   ```bash
   python -m src.data.processing
   ```
   Or edit `src/data/processing.py` `if __name__ == "__main__"` to point at your data path.

2. **Label simplification** (e.g. hand open/close):
   ```bash
   python -m src.data.simplify_labels --data EEG_clean/processed/ --mode hand_dir
   ```

3. **Training** (single run):
   ```bash
   python -m src.train --data EEG_clean/processed/simplified/hand_dir/ --features bandpower --model lda --fs 512
   ```

4. **Evaluation / comparison** (single mode or all modes):
   ```bash
   python -m src.eval --data EEG_clean/processed/simplified/hand_dir --fs 300 --base_outdir comparisons
   python -m src.eval --all --cv 5
   ```

To run the processing on 2024 data, use the scripts under `archives/2025/EEG` as reference.