#!/bin/bash
#SBATCH --job-name=sel_neuron
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=18:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/ablation/sel_neuron_%a.log
#SBATCH --array=0-5

SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/ablations"

# Define the input arrays
EFFECTS=(
    "suppress"
    "boost"
)

VECTORS=(
    "longtail"
)

TOP_NS=(
    10
    50
)

MODELS=(
    "EleutherAI/pythia-70m-deduped"
    "EleutherAI/pythia-410m-deduped"
)

# Calculate total combinations for validation
TOTAL_COMBINATIONS=$((${#EFFECTS[@]} * ${#VECTORS[@]} * ${#TOP_NS[@]} * ${#MODELS[@]}))

if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($TOTAL_COMBINATIONS)"
    exit 1
fi

# Calculate indices for each variable based on the SLURM array task ID
EFFECT_IDX=$(( SLURM_ARRAY_TASK_ID / (${#VECTORS[@]} * ${#TOP_NS[@]} * ${#MODELS[@]}) ))
VECTOR_IDX=$(( (SLURM_ARRAY_TASK_ID / (${#TOP_NS[@]} * ${#MODELS[@]})) % ${#VECTORS[@]} ))
TOP_N_IDX=$(( (SLURM_ARRAY_TASK_ID / ${#MODELS[@]}) % ${#TOP_NS[@]} ))
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID % ${#MODELS[@]} ))

# Get the actual values from arrays using the calculated indices
EFFECT="${EFFECTS[$EFFECT_IDX]}"
VECTOR="${VECTORS[$VECTOR_IDX]}"
TOP_N="${TOP_NS[$TOP_N_IDX]}"
MODEL="${MODELS[$MODEL_IDX]}"

# Log which combination is being processed
echo "Processing combination:"
echo " Effect: $EFFECT"
echo " Vector: $VECTOR"
echo " Model: $MODEL"
echo " Top N: $TOP_N"

# Run the analysis with the selected combination
python $SCRIPT_ROOT/sel_neuron.py \
    -m "$MODEL" \
    --effect "$EFFECT" \
    --top_n "$TOP_N" \
    --vector "$VECTOR"