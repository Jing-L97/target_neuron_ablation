#!/bin/bash
#SBATCH --job-name=sel_tail
#SBATCH --export=ALL
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/selection/sel_elbow_%a.log
#SBATCH --array=0-3

SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/selection"
HEURISTIC="prob"
SEL_FREQ="longtail_elbow"
STEP_MODE="multi"
# Define the input arrays
EFFECTS=(
    "boost"
)

VECTORS=(
    "longtail_elbow"
    
)

TOP_NS=(
    100
    -1
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
    --vector "$VECTOR" \
    --sel_freq "$SEL_FREQ" \
    --step_mode "$STEP_MODE" \
    --heuristic "$HEURISTIC"