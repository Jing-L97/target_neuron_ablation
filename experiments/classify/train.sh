#!/bin/bash
#SBATCH --job-name=hyperplane
#SBATCH --export=ALL
#SBATCH --partition=cpu
#SBATCH --mem=70G
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/classify/hyperplane_%a.log
#SBATCH --array=0-1  # Update if number of models changes

# Define constants
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/classify"

# Define model array
MODELS=("EleutherAI/pythia-70m-deduped" "EleutherAI/pythia-410m-deduped")  

# Validate array task ID
if [[ $SLURM_ARRAY_TASK_ID -ge ${#MODELS[@]} ]]; then
  echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds number of models (${#MODELS[@]})"
  exit 1
fi

# Select model based on array index
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

# Log selected configuration
echo "Running analysis with model: $MODEL"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"

# Run the analysis script
python "$SCRIPT_ROOT/train.py" \
  -m "$MODEL" \
  --resume

echo "Analysis complete for model: $MODEL"
