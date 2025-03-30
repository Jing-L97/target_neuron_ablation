#!/bin/bash
#SBATCH --job-name=sel_mean
#SBATCH --export=ALL
#SBATCH --partition=cpu
#SBATCH --mem=60G
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/ablation/sel_mean_%a.log
#SBATCH --array=0-7

SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/ablations"
EFFECT="boost"
VECTOR="mean"

# Define the input arrays
TOP_NS=(
    10
    100
    50
    500
)

MODELS=(
    "EleutherAI/pythia-70m-deduped"
    "EleutherAI/pythia-410m-deduped"
)

# Calculate total combinations for validation
TOTAL_COMBINATIONS=$((${#TOP_NS[@]} * ${#MODELS[@]}))
if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($TOTAL_COMBINATIONS)"
    exit 1
fi

# Calculate which combination to use based on the SLURM array task ID
TOP_N_IDX=$(( SLURM_ARRAY_TASK_ID / ${#MODELS[@]} ))
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID % ${#MODELS[@]} ))

# Get the actual values from arrays using the calculated indices
TOP_N="${TOP_NS[$TOP_N_IDX]}"
MODEL="${MODELS[$MODEL_IDX]}"

# Log which combination is being processed
echo "Processing combination:"
echo "  Model: $MODEL"
echo "  Top N: $TOP_N"

# Run the analysis with the selected combination
python $SCRIPT_ROOT/analyze_unigram.py \
    -m "$MODEL" \
    --effect "$EFFECT" \
    --top_n "$TOP_N" \
    --vector "$VECTOR"