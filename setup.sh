#!/bin/bash

source "$(conda info --base)/etc/profile.d/conda.sh"

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Creating Conda environment in ./env ---"
# Create conda environment with Python 3.10
conda create --prefix ./env python=3.10 -y

echo "--- Upgrading pip in the new environment ---"
# Use conda run to execute commands within the specified environment
conda run --prefix ./env pip install --upgrade pip

echo "--- Installing PyTorch ---"
conda run --prefix ./env pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo "--- Installing all other packages ---"
# NOTE: The indentation here is with standard spaces, which will work correctly.
conda run --prefix ./env pip install \
    ipykernel \
    redis==5.2.1 \
    jupyter==1.1.1 \
    numpy==2.1.2 \
    pandas==2.3.0 \
    matplotlib==3.10.1 \
    scipy==1.15.2 \
    scikit-learn==1.6.1 \
    tqdm==4.67.1 \
    g2p_en==2.1.0 \
    h5py==3.13.0 \
    omegaconf==2.3.0 \
    editdistance==0.8.1 \
    huggingface-hub==0.33.1 \
    transformers==4.53.0 \
    tokenizers==0.21.2 \
    accelerate==1.8.1 \
    bitsandbytes==0.46.0 \
    torch

echo
echo "--- Setup Complete! ---"
echo "The 'env' environment is fully installed."
echo "To activate it in your terminal, navigate to this folder and run: conda activate ./env"
echo