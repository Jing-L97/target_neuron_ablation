#!/bin/bash
#SBATCH --job-name=a_geometry
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=12
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/ablation/a_geometry_%a.debug
#SBATCH --array=0-1

# Define constants
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/analysis"

# Define arrays
MODELS=(
  "EleutherAI/pythia-70m-deduped"
  "EleutherAI/pythia-410m-deduped"
)

# Total combinations = length of MODELS
TOTAL_COMBINATIONS=${#MODELS[@]}

# Safety check
if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]]; then
  echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($TOTAL_COMBINATIONS)"
  exit 1
fi

# Select model
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

# Log what's happening
echo "Processing model: $MODEL"

# Run the Python analysis
python "$SCRIPT_ROOT/analyze_activation_geometry.py" \
  -m "$MODEL"
