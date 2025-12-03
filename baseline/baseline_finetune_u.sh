#!/bin/bash

#SBATCH --job-name=rnn_stage2
#SBATCH --output=logs/stage2_%j.out
#SBATCH --error=logs/stage2_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=l40-gpu     
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

# Force script to project root
cd /work/users/s/j/sjshen/brain-to-text-project

mkdir -p logs

echo "Starting Stage 2 (Unfrozen) Fine-Tuning..."
echo "Running: baseline/finetuned_unfrozen.py"

./env/bin/python baseline/finetune_unfrozen.py

echo "Job Complete."