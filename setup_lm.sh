#!/bin/bash

# Ensure that the script is run from the root directory of the project
if [ ! -f "setup_lm.sh" ]; then
    echo "This script must be run from the root directory of the project."
    exit 1
fi

# ensure that the language_model/runtime/server/x86/build directory does not exist
if [ -d "language_model/runtime/server/x86/build" ]; then
    echo "The language_model/runtime/server/x86/build directory already exists. Please remove it before running this script."
    exit 1
fi

# ensure that the language_model/runtime/server/x86/fc_base directory does not exist
if [ -d "language_model/runtime/server/x86/fc_base" ]; then
    echo "The language_model/runtime/server/x86/fc_base directory already exists. Please remove it before running this script."
    exit 1
fi

# Load required modules (add them here instead of relying on pre-loaded modules)
echo "--- Loading required modules ---"
module purge
module load cmake
module load gcc
# Do NOT load anaconda module to avoid conflicts

# Ensure conda is available (should be in base without loading anaconda module)
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please ensure conda is available in your PATH."
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

# Remove existing environment if it exists
if [ -d "language_model/env_lm" ]; then
    echo "--- Removing existing conda environment ---"
    rm -rf language_model/env_lm
fi

# Create conda environment locally within the language_model folder
echo "--- Creating Conda environment in ./language_model/env_lm ---"
conda create --prefix ./language_model/env_lm python=3.9 -y

# Activate the new environment
conda activate ./language_model/env_lm

# Upgrade pip
pip install --upgrade pip

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

# cd to the language model directory and install the language model
echo "--- Compiling C++ components ---"
cd language_model/runtime/server/x86

echo "Cleaning up environment variables to prevent conflicts..."
# Clean environment variables that cause conflicts
unset CMAKE_PREFIX_PATH
unset CMAKE_INSTALL_PREFIX
export PKG_CONFIG_PATH=""

# Set specific environment for clean build
export CC=$(which gcc)
export CXX=$(which g++)

echo "Current CMake version:"
cmake --version

echo "Attempting to compile C++ components..."
# Run the setup with error handling
if ! python setup.py install; then
    echo "ERROR: C++ compilation failed!"
    echo "Trying alternative approach with CMake directly..."
    
    # Try building manually
    mkdir -p build
    cd build
    
    # Try different CMake configurations
    if cmake .. -DCMAKE_BUILD_TYPE=Release; then
        echo "CMake configuration successful, building..."
        if make -j$(nproc); then
            echo "Manual build successful!"
        else
            echo "Manual build failed!"
            exit 1
        fi
    else
        echo "CMake configuration failed!"
        exit 1
    fi
fi

# cd back to the root directory
cd ../../../..

conda deactivate

echo
echo "Setup complete! The 'env_lm' environment is installed in ./language_model/"
echo "You can now run the 'sbatch run_wer_eval.sbatch' script."
echo