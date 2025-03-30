#!/bin/bash
#SBATCH --job-name=mean_boost
#SBATCH --partition=gpu
#SBATCH --mem=70G
#SBATCH --exclude=puck5
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/surprisal/mean_boost_%a.log
#SBATCH --array=0-23

# Define constants for better readability and maintenance
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/surprisal"
VECTOR="mean"
EFFECT="boost"

# Define the input arrays
NEURON_FILES=(
    "500_10.csv"
    "500_50.csv"
    "500_500.csv"
)

ABLATIONS=(
    "mean"
    "zero"
)

WORDS=(
    "context/stas/c4-en-10k/5/longtail_words.json"
    "context/stas/c4-en-10k/5/merged.json"
)

MODELS=(
    "EleutherAI/pythia-70m-deduped"
    "EleutherAI/pythia-410m-deduped"
)

# Calculate total combinations for validation
TOTAL_COMBINATIONS=$((${#NEURON_FILES[@]} * ${#ABLATIONS[@]} * ${#WORDS[@]} * ${#MODELS[@]}))
if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($TOTAL_COMBINATIONS)"
    exit 1
fi

# Calculate which combination to use based on the SLURM array task ID
# Using integer division and modulo to extract indices
NEURON_IDX=$(( SLURM_ARRAY_TASK_ID / (${#ABLATIONS[@]} * ${#WORDS[@]} * ${#MODELS[@]}) ))
REMAINDER=$(( SLURM_ARRAY_TASK_ID % (${#ABLATIONS[@]} * ${#WORDS[@]} * ${#MODELS[@]}) ))

ABLATION_IDX=$(( REMAINDER / (${#WORDS[@]} * ${#MODELS[@]}) ))
REMAINDER=$(( REMAINDER % (${#WORDS[@]} * ${#MODELS[@]}) ))

WORD_IDX=$(( REMAINDER / ${#MODELS[@]} ))
MODEL_IDX=$(( REMAINDER % ${#MODELS[@]} ))

# Get the actual values from arrays using the calculated indices
NEURON_FILE="${NEURON_FILES[$NEURON_IDX]}"
ABLATION="${ABLATIONS[$ABLATION_IDX]}"
WORD="${WORDS[$WORD_IDX]}"
MODEL="${MODELS[$MODEL_IDX]}"

# Log which combination is being processed
echo "Processing combination:"
echo "  Model: $MODEL"
echo "  Word: $WORD"
echo "  Neuron file: $NEURON_FILE"
echo "  Ablation: $ABLATION"

# Run the surprisal computation with the selected combination
python $SCRIPT_ROOT/compute_surprisal.py \
    -m "$MODEL" \
    -w "$WORD" \
    -n "$NEURON_FILE" \
    -a "$ABLATION" \
    --effect "$EFFECT" \
    --vector "$VECTOR" \
    --resume \