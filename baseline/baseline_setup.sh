#!/bin/bash

#SBATCH --job-name=install_env
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16g             
#SBATCH --partition=general   
#SBATCH --time=02:00:00         

mkdir -p logs
module load anaconda

echo "Starting Environment Setup on $(hostname)..."
bash setup.sh
echo "Setup Complete."