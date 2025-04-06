#!/bin/bash
#SBATCH --job-name=410_h
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/surprisal/410_h_%a.log
#SBATCH --array=0-5

# Define constants for better readability and maintenance
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/surprisal"

# Define constants
EFFECTS=(
  "suppress"
  "boost"
)

# Define the input arrays
VECTORS=(
  "longtail"
)

# Fixed word variable
WORD="context/stas/c4-en-10k/5/longtail_words.json"

NEURON_FILES=(
  "500_10.csv"
  "500_50.csv"
)

ABLATIONS=(
  "mean"
  "zero"
)

# Fixed model variable
MODEL="EleutherAI/pythia-70m-deduped"

# Calculate total combinations for validation, including EFFECTS
TOTAL_COMBINATIONS=$((${#EFFECTS[@]} * ${#VECTORS[@]} * ${#NEURON_FILES[@]} * ${#ABLATIONS[@]}))

if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]]; then
  echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($TOTAL_COMBINATIONS)"
  exit 1
fi

# Calculate which combination to use based on the SLURM array task ID
# Using integer division and modulo to extract indices
EFFECT_IDX=$(( SLURM_ARRAY_TASK_ID / (${#VECTORS[@]} * ${#NEURON_FILES[@]} * ${#ABLATIONS[@]}) ))
REMAINDER=$(( SLURM_ARRAY_TASK_ID % (${#VECTORS[@]} * ${#NEURON_FILES[@]} * ${#ABLATIONS[@]}) ))
VECTOR_IDX=$(( REMAINDER / (${#NEURON_FILES[@]} * ${#ABLATIONS[@]}) ))
REMAINDER=$(( REMAINDER % (${#NEURON_FILES[@]} * ${#ABLATIONS[@]}) ))
NEURON_IDX=$(( REMAINDER / ${#ABLATIONS[@]} ))
ABLATION_IDX=$(( REMAINDER % ${#ABLATIONS[@]} ))

# Get the actual values from arrays using the calculated indices
EFFECT="${EFFECTS[$EFFECT_IDX]}"
VECTOR="${VECTORS[$VECTOR_IDX]}"
NEURON_FILE="${NEURON_FILES[$NEURON_IDX]}"
ABLATION="${ABLATIONS[$ABLATION_IDX]}"

# Log which combination is being processed
echo "Processing combination $SLURM_ARRAY_TASK_ID of $TOTAL_COMBINATIONS:"
echo " Effect: $EFFECT"
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