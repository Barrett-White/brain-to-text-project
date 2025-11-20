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

# 1. Load Modules
module purge
module load anaconda

# 2. Define The EXACT Python Interpreter (The Gigachad Fix)
# We do not rely on 'python', we rely on the full path.
ENV_PATH="/work/users/s/j/sjshen/brain-to-text-project/b2txt25_lm"
PYTHON_EXEC="$ENV_PATH/bin/python"

# 3. Activate (Mainly to set LD_LIBRARY_PATH for CUDA)
source /nas/longleaf/rhel9/apps/anaconda/2024.02/etc/profile.d/conda.sh
conda activate "$ENV_PATH"

# 4. Clean Environment Variables (Prevent Home Directory Conflict)
unset PYTHONPATH
export PYTHONNOUSERSITE=1

# 5. Verify Dependencies BEFORE running
echo "Using Python interpreter: $PYTHON_EXEC"
if [ ! -f "$PYTHON_EXEC" ]; then
    echo "CRITICAL ERROR: Python binary not found at $PYTHON_EXEC"
    exit 1
fi

# Check Torch specifically using the absolute binary
"$PYTHON_EXEC" -c "import torch; print(f'Torch successfully loaded: {torch.__version__}')" || exit 1
"$PYTHON_EXEC" -c "import pandas; print(f'Pandas successfully loaded: {pandas.__version__}')" || exit 1

# 6. Run Evaluation
echo "Starting evaluation..."
cd baseline

"$PYTHON_EXEC" evaluate_model.py \
    --model_path ../data/t15_pretrained_rnn_baseline \
    --data_dir ../data/hdf5_data_final \
    --lm_path ../language_model/pretrained_language_models/languageModel \
    --lm_alpha 0.55 \
    --lm_beta 1.0 \
    --eval_type val \
    --gpu_number 0