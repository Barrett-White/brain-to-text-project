#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=install_b2txt
#SBATCH --output=install_log_%j.out
#SBATCH --error=install_log_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --partition=general

# Ensure script is executable
chmod +x setup_lm.sh

# Run the main setup script
# setup_lm.sh handles module loading, conda creation, and compiling internally.
./setup_lm.sh

