#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/teammate_eval_%j.txt
#SBATCH --error=logs/teammate_eval_%j.err

echo "Job started on $(hostname)"

# 1. Load Modules
module purge
module load anaconda

# 2. Define Paths (Absolute)
PROJECT_ROOT="/work/users/s/j/sjshen/brain-to-text-project"
ENV_PATH="$PROJECT_ROOT/b2txt25_lm"
PYTHON_EXEC="$ENV_PATH/bin/python"

# 3. Activate Environment
source /nas/longleaf/rhel9/apps/anaconda/2024.02/etc/profile.d/conda.sh
conda activate "$ENV_PATH"

# 4. Isolation & Safety 
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$ENV_PATH/lib/python3.9/site-packages/nvidia/cublas/lib:$ENV_PATH/lib/python3.9/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"

# 5. Verify Execution
if [ ! -f "$PYTHON_EXEC" ]; then
    echo "CRITICAL: Python binary not found at $PYTHON_EXEC"
    exit 1
fi

# 6. Run Evaluation
echo "Starting teammate evaluation..."
cd "$PROJECT_ROOT"

"$PYTHON_EXEC" baseline/eval_teammate.py \
    --npz_file "$PROJECT_ROOT/data/teammate_logits/val_logits_latest.npz" \
    --lm_path "$PROJECT_ROOT/language_model/pretrained_language_models/languageModel" \
    --lm_beta 90.0 \
    --acoustic_scale 0.325 \
    --logits_key "logits" \
    --labels_key "sentence_label"

echo "Evaluation Complete."