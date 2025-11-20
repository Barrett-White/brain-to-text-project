#!/bin/bash

# 1. PRE-FLIGHT CHECKS
if [ ! -f "setup_lm.sh" ]; then
    echo "Error: This script must be run from the root directory of the project."
    exit 1
fi

# Ensure clean slate (Script fails if previous build exists, to prevent mixing)
if [ -d "language_model/runtime/server/x86/build" ]; then
    echo "Error: 'build' directory exists. Please run: rm -rf language_model/runtime/server/x86/build"
    exit 1
fi

if [ -d "language_model/runtime/server/x86/fc_base" ]; then
    echo "Error: 'fc_base' directory exists. Please run: rm -rf language_model/runtime/server/x86/fc_base"
    exit 1
fi

# 2. HPC MODULE SETUP (Auto-detects Longleaf)
if command -v module &> /dev/null; then
    echo "HPC Environment detected. Loading modules..."
    module purge
    module load anaconda
    module load cmake
    module load gcc
fi

# 3. VERIFY TOOLS
if ! command -v cmake &> /dev/null; then
    echo "Error: CMake is not installed (>= 3.14 required)."
    exit 1
fi

if ! command -v gcc &> /dev/null; then
    echo "Error: GCC is not installed (>= 10.1 required)."
    exit 1
fi

# 4. CONDA SETUP
# Try to find conda base path dynamically, fallback to Longleaf default
CONDA_BASE=$(conda info --base 2>/dev/null || echo "/nas/longleaf/rhel9/apps/anaconda/2024.02")
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 5. CREATE ENVIRONMENT
echo "Creating Conda Environment (b2txt25_lm)..."
conda create --prefix ./b2txt25_lm python=3.9 -y

# 6. DEPENDENCY INSTALLATION
# We define the Explicit Python Executable to bypass activation issues
PYTHON_EXEC="$(pwd)/b2txt25_lm/bin/python"

# ISOLATION: Prevent pip from seeing your home directory (~/.local)
export PYTHONNOUSERSITE=1

echo "Installing Golden Dependency List..."
# Upgrade pip first
"$PYTHON_EXEC" -m pip install --upgrade pip

# Install Exact Versions (Fixes Numpy 2.0 / Torch 1.13 / CUDA 11 conflicts)
"$PYTHON_EXEC" -m pip install \
    torch==1.13.1 \
    nvidia-cublas-cu11==11.10.3.66 \
    nvidia-cuda-runtime-cu11==11.7.99 \
    typing_extensions==4.10.0 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    redis==5.0.6 \
    jupyter==1.1.1 \
    matplotlib==3.9.0 \
    scipy==1.11.1 \
    scikit-learn==1.6.1 \
    tqdm==4.66.4 \
    g2p_en==2.1.0 \
    omegaconf==2.3.0 \
    huggingface-hub==0.23.4 \
    transformers==4.40.0 \
    tokenizers==0.19.1 \
    accelerate==0.33.0 \
    bitsandbytes==0.41.1

# 7. COMPILE DECODER
echo "Compiling Decoder..."
cd language_model/runtime/server/x86

# Unload system anaconda to prevent CMake gflags conflict
if command -v module &> /dev/null; then
    module unload anaconda
fi

# Point CMake to our isolated environment
export CMAKE_PREFIX_PATH="$(pwd)/../../../../b2txt25_lm"
export PKG_CONFIG_PATH="$(pwd)/../../../../b2txt25_lm/lib/pkgconfig"
export CMAKE_BUILD_PARALLEL_LEVEL=12

# Run setup using our explicit python
"$PYTHON_EXEC" setup.py install

# 8. FINISH
cd ../../../..

echo
echo "Setup complete!"
echo "To use: source $CONDA_BASE/etc/profile.d/conda.sh"
echo "        conda activate ./b2txt25_lm"
echo