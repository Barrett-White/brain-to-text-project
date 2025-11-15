#!/bin/bash

# --- 1. Request All Resources ---
# We need high RAM for the LM and a GPU for the RNN.
#SBATCH --job-name=b2t-wer-eval
#SBATCH --output=logs/wer_eval.txt
#SBATCH --error=logs/wer_eval.err
#SBATCH --ntasks=1
#SBATCH --qos=gpu_access
#SBATCH --cpus-per-task=8
#SBATCH --mem=310G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00


# --- 2. Setup a Cleanup Trap ---
# This "trap" function automatically runs when the script exits
# (either on success, error, or timeout).
cleanup() {
    echo "--- Job finished. Shutting down background servers. ---"
    # The $LM_PID and $REDIS_PID are set below
    if [ -n "$LM_PID" ]; then kill $LM_PID; fi
    if [ -n "$REDIS_PID" ]; then kill $REDIS_PID; fi
}
trap cleanup EXIT

# --- 3. Load Modules ---
echo "Job started on $(hostname)"
module load anaconda
module load cuda/12.x # Or the version your envs require
source "$(conda info --base)/etc/profile.d/conda.sh"
mkdir -p logs

# --- 4. Start Server 1: Redis ---
echo "--- Starting Redis Server ---"
redis-server &
# Save the Process ID (PID) so we can kill it later
REDIS_PID=$!
sleep 5 # Give Redis a moment to start

# --- 5. Start Server 2: Language Model ---
echo "--- Starting Language Model Server ---"
conda activate ./language_model/env_lm 
python language_model/language-model-standalone.py \
    --model_path language_model/pretrained_language_models/5gram.model \
    > logs/lm_server.log 2>&1 &
# Save its PID
LM_PID=$!
conda deactivate

# --- 6. Wait for LM to Load ---
# This is the most critical step. The LM takes a long time
# to load 300GB into RAM. We must wait.
echo "--- Waiting 5 minutes for LM to load model... ---"
sleep 300
echo "--- Wait finished. Assuming LM is ready. ---"

# --- 7. Run Client: Evaluation (Foreground) ---
echo "--- Activating Baseline Environment ---"
conda activate ./env

echo "--- Running Full Evaluation (WER) ---"
cd baseline
python evaluate_model.py \
    --model_path trained_models/baseline_rnn \
    --eval_type val \
    --gpu_number 0

echo "--- Evaluation Finished ---"
cd ..

# The 'trap' function will automatically run on exit to kill the servers.