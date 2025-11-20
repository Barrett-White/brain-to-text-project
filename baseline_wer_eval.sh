#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/wer_eval_%j.txt
#SBATCH --error=logs/wer_eval_%j.err

echo "Job started on $(hostname)"

# 1. Load Core Modules
module purge
module load anaconda

# 2. Define Absolute Paths (The Robust Fix)
PROJECT_ROOT="/work/users/s/j/sjshen/brain-to-text-project"
ENV_PATH="$PROJECT_ROOT/b2txt25_lm"
PYTHON_EXEC="$ENV_PATH/bin/python"

# 3. Activate Environment
source /nas/longleaf/rhel9/apps/anaconda/2024.02/etc/profile.d/conda.sh
conda activate "$ENV_PATH"

# 4. Isolation & Safety Variables
unset PYTHONPATH
export PYTHONNOUSERSITE=1
# Explicitly point to our environment's NVIDIA libs (Prevents libcublas errors)
export LD_LIBRARY_PATH="$ENV_PATH/lib/python3.9/site-packages/nvidia/cublas/lib:$ENV_PATH/lib/python3.9/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"

# 5. Verify Execution
echo "Using Python: $PYTHON_EXEC"
if [ ! -f "$PYTHON_EXEC" ]; then
    echo "CRITICAL: Python binary not found."
    exit 1
fi
"$PYTHON_EXEC" -c "import torch; print(f'Torch loaded: {torch.__version__}')" || exit 1

# 6. Run Evaluation (Using Absolute Paths for Everything)
echo "Starting evaluation..."

# Note: We run from PROJECT_ROOT/baseline so imports inside the script work, 
# but we pass absolute paths to the arguments so they never break.
cd "$PROJECT_ROOT/baseline"

"$PYTHON_EXEC" evaluate_model.py \
    --model_path "$PROJECT_ROOT/data/t15_pretrained_rnn_baseline" \
    --data_dir "$PROJECT_ROOT/data/hdf5_data_final" \
    --lm_path "$PROJECT_ROOT/language_model/pretrained_language_models/languageModel" \
    --lm_alpha 0.55 \
    --lm_beta 1.0 \
    --eval_type val \
    --gpu_number 0
