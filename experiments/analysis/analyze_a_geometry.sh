#!/bin/bash
#SBATCH --job-name=group_geometry
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/analysis/group_geometry_%a.log
#SBATCH --array=0-1

# Define constants
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/analysis"
Sel_by_Med="False"
GROUP_TYPE="group"

# Define arrays
MODELS=(
"EleutherAI/pythia-410m-deduped"
)
VECTORS=(
"longtail_50"
)
TOP_NS=(
10
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

# Calculate total combinations for each level
HEURISTIC_COMBINATIONS=$TOTAL_GROUP_SIZES
TOP_N_COMBINATIONS=$((TOTAL_HEURISTICS * TOTAL_GROUP_SIZES))
VECTOR_COMBINATIONS=$((TOTAL_TOP_NS * TOTAL_HEURISTICS * TOTAL_GROUP_SIZES))
MODEL_COMBINATIONS=$((TOTAL_VECTORS * TOTAL_TOP_NS * TOTAL_HEURISTICS * TOTAL_GROUP_SIZES))

TOTAL_COMBINATIONS=$MODEL_COMBINATIONS

# Safety check
if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($TOTAL_COMBINATIONS)"
    exit 1
fi

# Calculate indices for each parameter correctly
task_id=$SLURM_ARRAY_TASK_ID

# Calculate each index from task_id
model_idx=$((task_id / VECTOR_COMBINATIONS))
remainder=$((task_id % VECTOR_COMBINATIONS))

vector_idx=$((remainder / TOP_N_COMBINATIONS))
remainder=$((remainder % TOP_N_COMBINATIONS))

top_n_idx=$((remainder / HEURISTIC_COMBINATIONS))
remainder=$((remainder % HEURISTIC_COMBINATIONS))

heuristic_idx=$((remainder / TOTAL_GROUP_SIZES))
group_size_idx=$((remainder % TOTAL_GROUP_SIZES))

# Debug output to verify indices
echo "Debug indices: model=$model_idx, vector=$vector_idx, top_n=$top_n_idx, heuristic=$heuristic_idx, group_size=$group_size_idx"

# Select parameters with bounds checking
if [[ $model_idx -ge $TOTAL_MODELS ]]; then
    echo "Error: model_idx ($model_idx) out of bounds (max $((TOTAL_MODELS-1)))"
    exit 1
fi
MODEL="${MODELS[$model_idx]}"

if [[ $vector_idx -ge $TOTAL_VECTORS ]]; then
    echo "Error: vector_idx ($vector_idx) out of bounds (max $((TOTAL_VECTORS-1)))"
    exit 1
fi
VECTOR="${VECTORS[$vector_idx]}"

if [[ $top_n_idx -ge $TOTAL_TOP_NS ]]; then
    echo "Error: top_n_idx ($top_n_idx) out of bounds (max $((TOTAL_TOP_NS-1)))"
    exit 1
fi
TOP_N="${TOP_NS[$top_n_idx]}"

if [[ $heuristic_idx -ge $TOTAL_HEURISTICS ]]; then
    echo "Error: heuristic_idx ($heuristic_idx) out of bounds (max $((TOTAL_HEURISTICS-1)))"
    exit 1
fi
HEURISTIC="${HEURISTICS[$heuristic_idx]}"

if [[ $group_size_idx -ge $TOTAL_GROUP_SIZES ]]; then
    echo "Error: group_size_idx ($group_size_idx) out of bounds (max $((TOTAL_GROUP_SIZES-1)))"
    exit 1
fi
GROUP_SIZE="${GROUP_SIZES[$group_size_idx]}"

# Verify parameters are not empty
if [[ -z "$MODEL" ]]; then
    echo "Error: MODEL is empty. Index: $model_idx, Available models: ${MODELS[@]}"
    exit 1
fi

if [[ -z "$VECTOR" ]]; then
    echo "Error: VECTOR is empty. Index: $vector_idx, Available vectors: ${VECTORS[@]}"
    exit 1
fi

if [[ -z "$TOP_N" ]]; then
    echo "Error: TOP_N is empty. Index: $top_n_idx, Available top_ns: ${TOP_NS[@]}"
    exit 1
fi

if [[ -z "$HEURISTIC" ]]; then
    echo "Error: HEURISTIC is empty. Index: $heuristic_idx, Available heuristics: ${HEURISTICS[@]}"
    exit 1
fi

if [[ -z "$GROUP_SIZE" ]]; then
    echo "Error: GROUP_SIZE is empty. Index: $group_size_idx, Available group sizes: ${GROUP_SIZES[@]}"
    exit 1
fi

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