#!/bin/bash
#SBATCH --job-name=geometry
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/ablation/geometry_%a.debug
#SBATCH --array=0-19

# Define constants for better readability and maintenance
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/ablations"

# Define arrays for parameters
MODELS=(
  "EleutherAI/pythia-70m-deduped"
  "EleutherAI/pythia-410m-deduped"
)

VECTORS=(
  "longtail"
  "mean"
)

NEURON_FILES=(
  "500_10.csv"
  "500_100.csv"
  "500_50.csv"
  "500_500.csv"
)

# Calculate total combinations for validation
TOTAL_COMBINATIONS=$((${#MODELS[@]} * ${#VECTORS[@]} * ${#NEURON_FILES[@]}))

# Validate array task ID
if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]]; then
  echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($TOTAL_COMBINATIONS)"
  exit 1
fi

# Calculate which combination to use based on the SLURM array task ID
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / (${#VECTORS[@]} * ${#NEURON_FILES[@]})))
REMAINDER=$((SLURM_ARRAY_TASK_ID % (${#VECTORS[@]} * ${#NEURON_FILES[@]})))
VECTOR_IDX=$((REMAINDER / ${#NEURON_FILES[@]}))
NEURON_FILE_IDX=$((REMAINDER % ${#NEURON_FILES[@]}))

# Get the actual values from arrays using the calculated indices
MODEL="${MODELS[$MODEL_IDX]}"
VECTOR="${VECTORS[$VECTOR_IDX]}"
NEURON_FILE="${NEURON_FILES[$NEURON_FILE_IDX]}"

# Default neuron number for the script
NEURON_NUM=0

# Log which combination is being processed
echo "Processing combination:"
echo " Model: $MODEL"
echo " Vector: $VECTOR"
echo " Neuron file: $NEURON_FILE"
echo " Neuron number: $NEURON_NUM"
echo " Combination index: $SLURM_ARRAY_TASK_ID of $TOTAL_COMBINATIONS"

# Run the direction analysis
python $SCRIPT_ROOT/analyze_geometry.py \
  -m "$MODEL" \
  -n "$NEURON_FILE" \
  --vector "$VECTOR" \
  --neuron_num "$NEURON_NUM" \
  --resume