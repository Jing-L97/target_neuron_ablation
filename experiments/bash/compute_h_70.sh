#!/bin/bash
#SBATCH --job-name=70_h
#SBATCH --partition=cpu
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/surprisal/70_h_%a.log
#SBATCH --array=0-47

# Define constants for better readability and maintenance
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/surprisal"

# Define constants
EFFECTS=(
  "boost"
  "suppress"
)

# Define the input arrays
VECTORS=(
  "longtail"
  "mean"
)

# Fixed word variable
WORD="context/stas/c4-en-10k/5/longtail_words.json"

NEURON_FILES=(
  "500_10.csv"
  "500_50.csv"
  "500_500.csv"
)

ABLATIONS=(
  "mean"
  "zero"
  "random"
  "full"
  "scaled"
)

# Fixed model variable
MODEL="EleutherAI/pythia-410m-deduped"

# Fixed effect (first one from the array)
EFFECT="${EFFECTS[0]}"

# Calculate total combinations for validation
TOTAL_COMBINATIONS=$((${#VECTORS[@]} * ${#NEURON_FILES[@]} * ${#ABLATIONS[@]}))

echo "DEBUG: Total combinations = $TOTAL_COMBINATIONS"
echo "DEBUG: Vectors = ${#VECTORS[@]}, Neuron files = ${#NEURON_FILES[@]}, Ablations = ${#ABLATIONS[@]}"

if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]]; then
  echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($TOTAL_COMBINATIONS)"
  exit 1
fi

# Calculate which combination to use based on the SLURM array task ID
# Using integer division and modulo to extract indices
VECTOR_IDX=$(( SLURM_ARRAY_TASK_ID / (${#NEURON_FILES[@]} * ${#ABLATIONS[@]}) ))
REMAINDER=$(( SLURM_ARRAY_TASK_ID % (${#NEURON_FILES[@]} * ${#ABLATIONS[@]}) ))
NEURON_IDX=$(( REMAINDER / ${#ABLATIONS[@]} ))
ABLATION_IDX=$(( REMAINDER % ${#ABLATIONS[@]} ))

# Get the actual values from arrays using the calculated indices
VECTOR="${VECTORS[$VECTOR_IDX]}"
NEURON_FILE="${NEURON_FILES[$NEURON_IDX]}"
ABLATION="${ABLATIONS[$ABLATION_IDX]}"

# Log which combination is being processed
echo "Processing combination $SLURM_ARRAY_TASK_ID of $TOTAL_COMBINATIONS:"
echo " Effect: $EFFECT (fixed)"
echo " Vector: $VECTOR"
echo " Model: $MODEL (fixed)"
echo " Word: $WORD (fixed)"
echo " Neuron file: $NEURON_FILE"
echo " Ablation: $ABLATION"

# Run the surprisal computation with the selected combination
python $SCRIPT_ROOT/compute_surprisal.py \
  -m "$MODEL" \
  -w "$WORD" \
  -n "$NEURON_FILE" \
  -a "$ABLATION" \
  --effect "$EFFECT" \
  --vector "$VECTOR" \
  --resume