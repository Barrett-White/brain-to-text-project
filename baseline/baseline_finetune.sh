#!/bin/bash

#SBATCH --job-name=finetune_rnn
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=l40-gpu     
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

# 1. Skip module loads
# If you get a "conda: command not found" error, uncomment the next line:
# module load anaconda

# 2. Activate your custom environment
cd /work/users/s/j/sjshen/brain-to-text-project

# 3. Ensure log directory exists
mkdir -p logs

# 4. Run the Fine-Tuning Script
echo "Starting Fine-Tuning Job on $(hostname)..."
echo "Using Python: $(which python)"
echo "Using PyTorch CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"

./env/bin/python baseline/finetune.py
echo "Job Complete."