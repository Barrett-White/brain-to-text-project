#!/bin/bash

# Request all resources
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH --output=logs/wer_eval.txt
#SBATCH --error=logs/wer_eval.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00


# Setup a cleanup trap
# This "trap" function automatically runs when the script exits
# (either on success, error, or timeout).
cleanup() {
    echo "Job finished. Shutting down background servers."
    # The $LM_PID and $REDIS_PID are set below
    if [ -n "$LM_PID" ]; then kill $LM_PID; fi
    if [ -n "$REDIS_PID" ]; then kill $REDIS_PID; fi
}
trap cleanup EXIT

# Load modules
echo "Job started on $(hostname)"
module load anaconda
module load cuda/12.2  
module load redis   
source "$(conda info --base)/etc/profile.d/conda.sh"
mkdir -p logs

# Start Server 1: Redis
echo "Starting Redis Server"
redis-server &
# Save the Process ID (PID) so we can kill it later
REDIS_PID=$!
sleep 5 # Give Redis a moment to start

# Start Server 2: Language Model
echo "Starting Language Model Server"
# We must first activate the environment, then run python
conda activate ./language_model/env_lm 
python language_model/language-model-standalone.py \
    --model_path language_model/pretrained_language_models/3gram.model \
    > logs/lm_server.log 2>&1 &
# Save its PID
LM_PID=$!
conda deactivate

# Wait for LM to Load
# This is the most critical step. The LM takes a long time
# to load 60GB into RAM. We must wait.
echo "Waiting 3 minutes for LM to load model..."
sleep 180
echo "Wait finished. Assuming LM is ready."

# Run Client: Evaluation (Foreground)
echo "Activating Baseline Environment"
conda activate ./env

echo "Running Full Evaluation (WER)"
cd baseline
# We must use the *original* evaluate_model.py
python evaluate_model.py \
    --model_path trained_models/baseline_rnn \
    --eval_type val \
    --gpu_number 0

echo "Evaluation Finished"
cd ..

# The 'trap' function will automatically run on exit to kill the servers.