# Brain-to-Text Neural Decoding

## Overview
This project explores neural decoding methods for translating recorded brain signals into phoneme-level and text outputs.
The goal is to improve performance over an original GRU-based baseline by experimenting with stronger sequence modeling
approaches, preprocessing strategies, and evaluation workflows.

This repository is forked from a collaborative group project. This fork highlights my individual contributions and
experimentation.

## Problem
Decoding neural activity into language is challenging due to:
- High temporal noise in neural signals
- Long-range dependencies in phoneme and word sequences
- Sensitivity of sequence models to preprocessing and smoothing choices

The baseline implementation relied on a GRU-based model, which provided a strong starting point but left room for
improvements in temporal modeling and robustness.

## My Contributions
My personal contributions to this project focused on improving the original baseline and evaluating alternative
modeling choices. Specifically, I worked on:
- Improving upon the original GRU baseline architecture with a transformer architecture
- Designing and tuning preprocessing and smoothing strategies
- Contributing to evaluation methodology and result interpretation

## Approach & Results
Model development followed an iterative process guided by phoneme and word error rate metrics. Improvements were driven by:
- Better handling of temporal dependencies
- Reduced sensitivity to noise through preprocessing
- More stable training dynamics during optimization

These changes resulted in measurable improvements over the original GRU baseline in controlled evaluations.


## Prerequisites

* **Git:** Install **Git Bash** for Windows from [git-scm.com](https://git-scm.com/).
* **Conda:** Anaconda or Miniconda.

---

## Installation

1. **Clone the Repo**

    ```bash
    git clone https://github.com/Sean0418/brain-to-text-project
    cd brain-to-text-project
    ```

2. **One-Time Fix for Git Bash on Windows**
    * Open **Anaconda Prompt** and run `conda init bash`.
    * Restart all your terminals.

3. **Run the Setup Script**
    * In a new Git Bash terminal, run the following:

    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```

    This creates a local environment in `./env` and installs all packages. This step will take a few minutes.

---

## Usage

1. **Activate the Environment**

    ```bash
    conda activate ./env
    ```

2. **Deactivate When Done**

    ```bash
    conda deactivate
    ```

## Load Data

Navigate to the [NEJM Data Github](https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text/tree/main/data) to find instructions for downloading the correct dataset.



```bash
sbatch -p volta-gpu baseline_wer_eval.sh

# If using volta-gpu
cd /work/users/s/j/sjshen/brain-to-text-project
# Patch both the main script and the helper script
sed -i 's/bfloat16/float16/g' baseline/evaluate_model.py baseline/evaluate_model_helpers.py
```
