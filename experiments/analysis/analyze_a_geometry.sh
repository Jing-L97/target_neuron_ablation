#!/bin/bash
#SBATCH --job-name=group_geometry
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --exclude=puck5
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/analysis/group_geometry_%a.log
#SBATCH --array=0-11%1

# Define constants
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/analysis"
Sel_by_Med="False"
GROUP_TYPE="group"

# Define arrays
MODELS=(
    "EleutherAI/pythia-70m-deduped"  
    "EleutherAI/pythia-410m-deduped"
)
VECTORS=(
    "longtail_50"
)
TOP_NS=(
    10
    50
    100
)
HEURISTICS=(
    "prob"
)
GROUP_SIZES=(
    "best"
    "target_size"
)

# Calculate total combinations
TOTAL_MODELS=${#MODELS[@]}
TOTAL_VECTORS=${#VECTORS[@]}
TOTAL_TOP_NS=${#TOP_NS[@]}
TOTAL_HEURISTICS=${#HEURISTICS[@]}
TOTAL_GROUP_SIZES=${#GROUP_SIZES[@]}
TOTAL_COMBINATIONS=$((TOTAL_MODELS * TOTAL_VECTORS * TOTAL_TOP_NS * TOTAL_HEURISTICS * TOTAL_GROUP_SIZES))

# Safety check
if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($TOTAL_COMBINATIONS)"
    exit 1
fi

# Calculate indices for each parameter
temp=$SLURM_ARRAY_TASK_ID
model_idx=$((temp / (TOTAL_VECTORS * TOTAL_TOP_NS * TOTAL_HEURISTICS)))
temp=$((temp % (TOTAL_VECTORS * TOTAL_TOP_NS * TOTAL_HEURISTICS)))
vector_idx=$((temp / (TOTAL_TOP_NS * TOTAL_HEURISTICS)))
temp=$((temp % (TOTAL_TOP_NS * TOTAL_HEURISTICS)))
top_n_idx=$((temp / TOTAL_TOP_NS))
heuristic_idx=$((temp % TOTAL_HEURISTICS))
group_size_idx=$((temp % TOTAL_GROUP_SIZES))

# Select parameters
MODEL="${MODELS[$model_idx]}"
VECTOR="${VECTORS[$vector_idx]}"
TOP_N="${TOP_NS[$top_n_idx]}"
HEURISTIC="${HEURISTICS[$heuristic_idx]}"
GROUP_SIZE="${GROUP_SIZES[$group_size_idx]}"

# Log what's happening
echo "Processing parameters:"
echo " Model: $MODEL"
echo " Vector: $VECTOR"
echo " Top_N: $TOP_N"
echo " Heuristic: $HEURISTIC"
echo " Group_size: $GROUP_SIZE"
echo " Task ID: $SLURM_ARRAY_TASK_ID of $TOTAL_COMBINATIONS"

# Run the Python analysis
python "$SCRIPT_ROOT/analyze_activation_geometry.py" \
    -m "$MODEL" \
    --vector "$VECTOR" \
    --group_type "$GROUP_TYPE" \
    --group_size "$GROUP_SIZE" \
    --top_n "$TOP_N" \
    --heuristic "$HEURISTIC" \
    --resume

echo "Analysis complete for combination $SLURM_ARRAY_TASK_ID"