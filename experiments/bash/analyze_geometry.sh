#!/bin/bash
#SBATCH --job-name=geometry
#SBATCH --export=ALL
#SBATCH --partition=cpu
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/ablation/geometry_%a.debug
#SBATCH --array=0-163  # Total of 2 models × 2 vectors × 41 neuron numbers = 164 jobs (0-163)

# Define range of neuron numbers to process
NEURON_MIN=10
NEURON_MAX=50
NEURON_RANGE=$((NEURON_MAX - NEURON_MIN + 1))  # Number of neuron numbers to process (41)

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
  "500_50.csv"
  "500_500.csv"
)

# Calculate total combinations for verification
TOTAL_COMBINATIONS=$((${#MODELS[@]} * ${#VECTORS[@]} * NEURON_RANGE))

# Validate array task ID
if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]]; then
  echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($TOTAL_COMBINATIONS)"
  exit 1
fi

# Calculate which combination to use based on the SLURM array task ID
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / (${#VECTORS[@]} * NEURON_RANGE)))
REMAINDER=$((SLURM_ARRAY_TASK_ID % (${#VECTORS[@]} * NEURON_RANGE)))
VECTOR_IDX=$((REMAINDER / NEURON_RANGE))
NEURON_IDX=$((REMAINDER % NEURON_RANGE))

# Calculate the actual neuron number
NEURON_NUM=$((NEURON_MIN + NEURON_IDX))

# Get the actual values from arrays using the calculated indices
MODEL="${MODELS[$MODEL_IDX]}"
VECTOR="${VECTORS[$VECTOR_IDX]}"

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