#!/bin/bash
#SBATCH --job-name=thre_pythia
#SBATCH --export=ALL
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=30:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/selection/thre_pythia_%a.log
#SBATCH --array=0-39  # adjust depending on total combinations

SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/selection"
HEURISTIC="prob"
SEL_FREQ="longtail"
STEP_MODE="multi"

# Define the input arrays
EFFECTS=("boost" "suppress")
VECTORS=("longtail_0_50")
TOP_NS=(-1)
MODELS=(
    "EleutherAI/pythia-1.4B-deduped"
    "EleutherAI/pythia-160M-deduped"
)
MAX_FREQS=(50 45 40 35 30 25 20 15 10 5)
MIN_FREQ=0
# Total combinations
TOTAL_COMBINATIONS=$((${#EFFECTS[@]} * ${#VECTORS[@]} * ${#TOP_NS[@]} * ${#MODELS[@]} * ${#MAX_FREQS[@]}))

if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($TOTAL_COMBINATIONS)"
    exit 1
fi

# Calculate indices
EFFECT_IDX=$(( SLURM_ARRAY_TASK_ID / (${#VECTORS[@]} * ${#TOP_NS[@]} * ${#MODELS[@]} * ${#MAX_FREQS[@]}) ))
VECTOR_IDX=$(( (SLURM_ARRAY_TASK_ID / (${#TOP_NS[@]} * ${#MODELS[@]} * ${#MAX_FREQS[@]})) % ${#VECTORS[@]} ))
TOP_N_IDX=$(( (SLURM_ARRAY_TASK_ID / (${#MODELS[@]} * ${#MAX_FREQS[@]})) % ${#TOP_NS[@]} ))
MODEL_IDX=$(( (SLURM_ARRAY_TASK_ID / ${#MAX_FREQS[@]}) % ${#MODELS[@]} ))
MAX_FREQ_IDX=$(( SLURM_ARRAY_TASK_ID % ${#MAX_FREQS[@]} ))

# Assign values
EFFECT="${EFFECTS[$EFFECT_IDX]}"
VECTOR="${VECTORS[$VECTOR_IDX]}"
TOP_N="${TOP_NS[$TOP_N_IDX]}"
MODEL="${MODELS[$MODEL_IDX]}"
MAX_FREQ="${MAX_FREQS[$MAX_FREQ_IDX]}"

# Log info
echo "Processing combination:"
echo " Effect: $EFFECT"
echo " Vector: $VECTOR"
echo " Model: $MODEL"
echo " Top N: $TOP_N"
echo " Max Freq: $MAX_FREQ"

# Run the analysis
python $SCRIPT_ROOT/sel_neuron.py \
    -m "$MODEL" \
    --effect "$EFFECT" \
    --top_n "$TOP_N" \
    --vector "$VECTOR" \
    --sel_freq "$SEL_FREQ" \
    --step_mode "$STEP_MODE" \
    --max_freq "$MAX_FREQ" \
    --min_freq "$MIN_FREQ" \
    --heuristic "$HEURISTIC"
