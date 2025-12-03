#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/wer_eval_%j.txt
#SBATCH --error=logs/wer_eval_%j.err

echo "Job started on $(hostname)"

# 1. Load Core Modules
module purge
module load anaconda

# 2. Define Absolute Paths
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
echo "Using Python: $PYTHON_EXEC"
if [ ! -f "$PYTHON_EXEC" ]; then
    echo "CRITICAL: Python binary not found."
    exit 1
fi

# 6. Run Evaluation
echo "Starting evaluation..."
cd "$PROJECT_ROOT/baseline"

"$PYTHON_EXEC" evaluate_model.py \
    --model_path "$PROJECT_ROOT/baseline/trained_models/finetuned_rnn_unfrozen" \
    --data_dir "$PROJECT_ROOT/data/hdf5_data_final" \
    --lm_path "$PROJECT_ROOT/language_model/pretrained_language_models/languageModel" \
    --lm_alpha 0.55 \
    --lm_beta 90 \
    --acoustic_scale 0.325 \
    --eval_type val \
    --gpu_number 0
