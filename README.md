## 📄 README: Motor Decoder Signal Processing Pipeline

This repository contains the pipeline for processing EEG data and developing classifiers for prosthetic hand control.

### Project Objective

The primary goal is to transform raw EEG signals into machine learning-ready features to classify distinct motor intentions for potential use in a Brain-Computer Interface (BCI).

-----

### ⚙️ Pipeline Overview

The workflow is uses three main scripts:

1.  **`processing.py`:**
      * Handles **signal preprocessing** (filtering, referencing, ICA) and converts raw LSL markers into the final integer labels.
      * Generates plots for quality control and feature visualization.
2.  **`simplify_labels.py`:**
      * **Reduces the problem** into binary or specialized classification tasks (e.g., Hand Open vs. Hand Close) to optimize model training.
3.  **`train.py`:**
      * Performs **feature extraction** (Bandpower, CSP) and trains robust linear classifiers (LDA, SVM) using cross-validation.

-----

### 📂 Data & Input

This repository contains data from multiple acquisition phases, including 2024 data (`archives/2024/EEG`), and more recent 2025 data (`sub-P005/`). 

  * **Input Data:** Raw `.xdf` files are organized by subject and session.
  * **Output Data:** Cleaned epochs (`.npz`) and processed metadata are saved into the `EEG_clean/` directory.



-----

###  Getting Started

To execute the full pipeline and generate a first trained model:

1.  **Run Preprocessing:** Start by cleaning the raw signals.
    ```bash
    python processing_new.py
    ```
2.  **Run Simplification:** Create the binary classification sets (e.g., for hand movements, based on codes from the acquisition team).
    ```bash
    python simplify_labels.py --data EEG_clean/processed/ --mode hand_dir
    ```
3.  **Run Training:** Train a baseline model on the simplified data.
    ```bash
    python train.py --data EEG_clean/processed/simplified/hand_dir/ --features bandpower --model lda --fs 512
    ```

To run the processing on 2024 data, run the following steps under the `archives/2025/EEG` directory.