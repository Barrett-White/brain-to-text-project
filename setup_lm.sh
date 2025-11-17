#!/bin/bash

# Ensure that the script is run from the root directory of the project
if [ ! -f "setup_lm.sh" ]; then
    echo "This script must be run from the root directory of the project."
    exit 1
fi

# Load required modules first
module load cmake
module load gcc

# Check if kaldi directory exists, if not, provide instructions
if [ ! -d "language_model/runtime/server/x86/kaldi" ]; then
    echo "WARNING: Kaldi directory not found at language_model/runtime/server/x86/kaldi"
    echo "This might cause the build to fail."
    echo "You may need to download and build Kaldi separately."
    read -p "Continue without Kaldi? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Clean build directories
rm -rf language_model/runtime/server/x86/build
rm -rf language_model/runtime/server/x86/fc_base

# Ensure conda is available
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create conda environment locally within the language_model folder
echo "--- Creating Conda environment in ./language_model/env_lm ---"
conda create --prefix ./language_model/env_lm python=3.9 -y

# Activate the new environment
conda activate ./language_model/env_lm

# Upgrade pip
pip install --upgrade pip

# Install the C++ dependencies
echo "--- Installing C++ dependencies from conda-forge ---"
conda install -c conda-forge gflags glog -y

# Install additional packages
echo "--- Installing Python packages ---"
pip install \
    torch==1.13.1 \
    redis==5.0.6 \
    jupyter==1.1.1 \
    numpy==1.24.4 \
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

# Build the language model components
echo "--- Compiling C++ components ---"
cd language_model/runtime/server/x86

# Set up build environment
unset CMAKE_PREFIX_PATH
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include:$CPLUS_INCLUDE_PATH"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Try to build with Kaldi workaround
if [ ! -d "kaldi" ]; then
    echo "Kaldi not found, attempting build without full Kaldi dependencies..."
    # Create a dummy kaldi directory structure to satisfy CMake
    mkdir -p kaldi/src
    touch kaldi/CMakeLists.txt
fi

# Build with verbose output to see errors
python setup.py install --verbose

cd ../../../..

conda deactivate

echo
echo "Setup complete! The 'env_lm' environment is installed in ./language_model/"
echo "You can now run the 'sbatch baseline_wer_eval.sbatch' script."
echo