# Brain-to-Text Project

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