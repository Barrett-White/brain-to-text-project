#!/bin/bash

source "$(conda info --base)/etc/profile.d/conda.sh"

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Creating Conda environment in ./env ---"
# Create conda environment with Python 3.10
conda create --prefix ./env python=3.10 -y

# Use conda run to execute commands within the specified environment
# but continue even with an errors during pip upgrade
echo "--- Upgrading pip in the new environment ---"
conda run --prefix ./env pip install --upgrade pip || true

echo "--- Installing PyTorch ---"
conda run --prefix ./env pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126

echo "--- Installing all other packages ---"
# NOTE: The indentation here is with standard spaces, which will work correctly.
conda run --prefix ./env pip install -r requirements.txt

echo
echo "--- Setup Complete! ---"
echo "The 'env' environment is fully installed."
echo "To activate it in your terminal, navigate to this folder and run: conda activate ./env"
echo