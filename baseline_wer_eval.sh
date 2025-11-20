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
module load cuda/12.2

ENV_PATH="/work/users/s/j/sjshen/brain-to-text-project/b2txt25_lm"
PYTHON_EXEC="$ENV_PATH/bin/python"

# 2. Activate Environment
source /nas/longleaf/rhel9/apps/anaconda/2024.02/etc/profile.d/conda.sh
conda activate "$ENV_PATH"

unset PYTHONPATH
export PYTHONNOUSERSITE=1

echo "Using Python interpreter: $PYTHON_EXEC"
echo "Verifying Torch..."
"$PYTHON_EXEC" -c "import torch; print(f'Torch version: {torch.__version__}')" || exit 1

# 3. Run Evaluation
cd baseline
python evaluate_model.py \
    --model_path ../data/t15_pretrained_rnn_baseline \
    --data_dir ../data/hdf5_data_final \
    --lm_path ../language_model/pretrained_language_models/languageModel \
    --lm_alpha 0.55 \
    --lm_beta 1.0 \
    --eval_type val \
    --gpu_number 0
