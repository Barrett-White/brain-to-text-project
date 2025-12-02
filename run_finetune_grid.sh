#!/bin/bash

# --- CONFIGURATION RANGES ---
# Alpha: Controls grammar strength. (Higher = stricter grammar)
ALPHAS=(0.3 0.5 0.7)

# Beta: Controls word length/silence. (Higher = shorter sentences/more silence)
# We test "Sane" values (2.0) and "Legacy" values (90.0) just in case.
BETAS=(2.0 5.0 90.0)

# Scale: Controls trust in the RNN. (Higher = trust brain signals more)
SCALES=(0.5 0.8 1.0 1.2)

# --- THE LOOP ---
for alpha in "${ALPHAS[@]}"; do
  for beta in "${BETAS[@]}"; do
    for scale in "${SCALES[@]}"; do
      
      # 1. Create a unique job name
      JOB_NAME="tune_a${alpha}_b${beta}_s${scale}"
      
      # 2. Generate a temporary SLURM script
      cat <<EOT > temp_${JOB_NAME}.sh
#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/${JOB_NAME}.txt
#SBATCH --job-name=${JOB_NAME}

# Standard Environment Setup
module purge; module load anaconda
source /nas/longleaf/rhel9/apps/anaconda/2024.02/etc/profile.d/conda.sh
conda activate /work/users/s/j/sjshen/brain-to-text-project/b2txt25_lm

PROJECT_ROOT="/work/users/s/j/sjshen/brain-to-text-project"
PYTHON_EXEC="\$PROJECT_ROOT/b2txt25_lm/bin/python"
unset PYTHONPATH; export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="\$PROJECT_ROOT/b2txt25_lm/lib/python3.9/site-packages/nvidia/cublas/lib:\$LD_LIBRARY_PATH"

cd "\$PROJECT_ROOT/baseline"

echo "Running Grid Search: Alpha=${alpha}, Beta=${beta}, Scale=${scale}"

"\$PYTHON_EXEC" evaluate_model.py \
    --model_path "\$PROJECT_ROOT/baseline/trained_models/finetuned_rnn_unfrozen" \
    --data_dir "\$PROJECT_ROOT/data/hdf5_data_final" \
    --lm_path "\$PROJECT_ROOT/language_model/pretrained_language_models/languageModel" \
    --lm_alpha ${alpha} \
    --lm_beta ${beta} \
    --acoustic_scale ${scale} \
    --eval_type val \
    --gpu_number 0
EOT

      # 3. Submit the job
      echo "Submitting ${JOB_NAME}..."
      sbatch temp_${JOB_NAME}.sh
      
      # 4. Cleanup the temp file
      rm temp_${JOB_NAME}.sh
      
      sleep 1
      
    done
  done
done