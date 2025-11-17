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
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# Setup a cleanup trap
cleanup() {
    echo "Job finished. Shutting down background servers."
    if [ -n "$LM_PID" ]; then kill $LM_PID 2>/dev/null || true; fi
    if [ -n "$REDIS_PID" ]; then kill $REDIS_PID 2>/dev/null || true; fi
    # Clean up any remaining Redis processes
    pkill -f "redis-server" 2>/dev/null || true
}
trap cleanup EXIT

# Load modules
echo "Job started on $(hostname)"
module load anaconda
module load cuda/12.2
source "$(conda info --base)/etc/profile.d/conda.sh"
mkdir -p logs

# Start Server 1: Redis
echo "Starting Redis Server"
conda activate ./language_model/env_lm

# Clean any existing Redis processes and data
pkill -f "redis-server" 2>/dev/null || true
rm -f dump.rdb

# Start Redis with explicit configuration
redis-server --daemonize yes --port 6379 --bind 127.0.0.1 --save "" --appendonly no
REDIS_PID=$(pgrep redis-server)
sleep 8 # Give Redis more time to start

# Verify Redis is running
if ! pgrep redis-server > /dev/null; then
    echo "ERROR: Redis failed to start!"
    exit 1
fi

echo "Redis started with PID: $REDIS_PID"

# Start Server 2: Language Model
echo "Starting Language Model Server"
python language_model/language-model-standalone.py \
    --model_path language_model/pretrained_language_models/3gram.model \
    > logs/lm_server.log 2>&1 &
LM_PID=$!

conda deactivate

# Wait for LM to Load - increase wait time
echo "Waiting 4 minutes for LM to load model..."
sleep 240
echo "Wait finished. Assuming LM is ready."

# Run Client: Evaluation (Foreground)
echo "Activating Baseline Environment"
conda activate ./env

echo "Running Full Evaluation (WER)"
cd baseline

# Test Redis connection first
python -c "
import redis
try:
    r = redis.Redis(host='localhost', port=6379)
    r.ping()
    print('Redis connection successful')
except Exception as e:
    print(f'Redis connection failed: {e}')
    exit(1)
"

python evaluate_model.py \
    --model_path trained_models/baseline_rnn \
    --eval_type val \
    --gpu_number 0

echo "Evaluation Finished"
cd ..