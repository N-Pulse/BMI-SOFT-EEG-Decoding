## 📄 README: Motor Decoder Signal Processing Pipeline

This repository contains the pipeline for classifiying EEG data from 2024 (under `archives/2024/EEG`). 

-----

### ⚙️ Pipeline Overview

The workflow is uses three main scripts:

1.  **`processing.py`:**
      * Handles **signal preprocessing** (filtering, referencing, ICA) and converts raw LSL markers into the final integer labels.
      * Generates plots for quality control and feature visualization.
2.  **`simplify_labels.py`:**
      * **Reduces the problem** into binary or specialized classification tasks to optimize model training.
3.  **`train.py`:**
      * Performs **feature extraction** (Bandpower, CSP) and trains robust linear classifiers (LDA, SVM) using cross-validation.

-----

### 📂 Data & Input

  * **Input Data:** Raw `.xdf` files are organized by subject and session.
  * **Output Data:** Cleaned epochs (`.npz`) and processed metadata are saved into the `EEG_clean/` directory.



-----

###  Getting Started

To execute the full pipeline and generate a first trained model:

1.  **Directory:** Start by making sure you are in the correct directory.
    ```bash
    cd archives/2025/EEG 
    ```
2.  **Run Preprocessing:** Clean the raw signals.
    ```bash
    python processing.py
    ```
3.  **Run Simplification:** Create the binary classification sets.
    ```bash
    python simplify_labels.py --data EEG_clean/processed/ --mode elbow
    ```
4.  **Run Training:** Train a baseline model on the simplified data.
    ```bash
    python train.py --data EEG_clean/processed/simplified/elbow/ --features none --model lda
    ```

