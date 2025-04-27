#!/bin/bash
#SBATCH --job-name=geometry_dyna
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/analysis/geometry_dyna_%a.log
#SBATCH --array=0-35  # Updated array size to include all combinations

# Define constants
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/analysis"
Sel_by_Med="False"

# Define arrays
GROUP_TYPES=("individual" "group")
MODELS=("EleutherAI/pythia-410m-deduped")
VECTORS=("longtail_50")
TOP_NS=(10 50 100)
HEURISTICS=("prob")
GROUP_SIZES=("best" "target_size")
STEP_INDEXS=(-1 0 7)

# Calculate total combinations
TOTAL_MODELS=${#MODELS[@]}
TOTAL_VECTORS=${#VECTORS[@]}
TOTAL_TOP_NS=${#TOP_NS[@]}
TOTAL_HEURISTICS=${#HEURISTICS[@]}
TOTAL_GROUP_SIZES=${#GROUP_SIZES[@]}
TOTAL_STEP_INDEXS=${#STEP_INDEXS[@]}
TOTAL_GROUP_TYPES=${#GROUP_TYPES[@]}

TOTAL_COMBINATIONS=$((TOTAL_MODELS * TOTAL_VECTORS * TOTAL_TOP_NS * TOTAL_HEURISTICS * TOTAL_GROUP_SIZES * TOTAL_STEP_INDEXS * TOTAL_GROUP_TYPES))

# Safety check
if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]]; then
  echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($TOTAL_COMBINATIONS)"
  exit 1
fi

# Calculate indices for each parameter
temp=$SLURM_ARRAY_TASK_ID
group_type_idx=$((temp % TOTAL_GROUP_TYPES))
temp=$((temp / TOTAL_GROUP_TYPES))

step_idx=$((temp % TOTAL_STEP_INDEXS))
temp=$((temp / TOTAL_STEP_INDEXS))

group_size_idx=$((temp % TOTAL_GROUP_SIZES))
temp=$((temp / TOTAL_GROUP_SIZES))

heuristic_idx=$((temp % TOTAL_HEURISTICS))
temp=$((temp / TOTAL_HEURISTICS))

top_n_idx=$((temp % TOTAL_TOP_NS))
temp=$((temp / TOTAL_TOP_NS))

vector_idx=$((temp % TOTAL_VECTORS))
temp=$((temp / TOTAL_VECTORS))

model_idx=$((temp % TOTAL_MODELS))

# Select parameters
MODEL="${MODELS[$model_idx]}"
VECTOR="${VECTORS[$vector_idx]}"
TOP_N="${TOP_NS[$top_n_idx]}"
HEURISTIC="${HEURISTICS[$heuristic_idx]}"
GROUP_SIZE="${GROUP_SIZES[$group_size_idx]}"
STEP_INDEX="${STEP_INDEXS[$step_idx]}"
GROUP_TYPE="${GROUP_TYPES[$group_type_idx]}"

# Log what's happening
echo "Processing parameters:"
echo " Model: $MODEL"
echo " Vector: $VECTOR"
echo " Top_N: $TOP_N"
echo " Heuristic: $HEURISTIC"
echo " Group_size: $GROUP_SIZE"
echo " Step_index: $STEP_INDEX"
echo " Group_type: $GROUP_TYPE"
echo " Task ID: $SLURM_ARRAY_TASK_ID of $TOTAL_COMBINATIONS"

# Run the Python analysis
python "$SCRIPT_ROOT/analyze_activation_geometry_dynamics.py" \
  -m "$MODEL" \
  --vector "$VECTOR" \
  --group_type "$GROUP_TYPE" \
  --group_size "$GROUP_SIZE" \
  --top_n "$TOP_N" \
  --step_index "$STEP_INDEX" \
  --heuristic "$HEURISTIC" \
  --resume \
  --exclude_random \
  --load_stat

echo "Analysis complete for combination $SLURM_ARRAY_TASK_ID"
