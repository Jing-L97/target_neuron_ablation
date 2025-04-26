#!/bin/bash
#SBATCH --job-name=large_geometry
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/analysis/group_geometry_large_%a.log
#SBATCH --array=0-7  # Update if you change number of combinations

# Define constants
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/analysis"
Sel_by_Med="False"
GROUP_TYPE="group"

# Define parameter arrays
MODELS=("EleutherAI/pythia-70m-deduped" "EleutherAI/pythia-410m-deduped")
VECTORS=("longtail_50")
TOP_NS=(50 100)
HEURISTICS=("prob")
GROUP_SIZES=("best" "target_size")

# Total combinations
TOTAL_COMBINATIONS=$((${#MODELS[@]} * ${#VECTORS[@]} * ${#TOP_NS[@]} * ${#HEURISTICS[@]} * ${#GROUP_SIZES[@]}))

# Validate array index
if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]]; then
  echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($TOTAL_COMBINATIONS)"
  exit 1
fi

# Compute indices
COMBO_IDX=$SLURM_ARRAY_TASK_ID

MODEL_IDX=$((COMBO_IDX / (${#VECTORS[@]} * ${#TOP_NS[@]} * ${#HEURISTICS[@]} * ${#GROUP_SIZES[@]})))
REMAINDER=$((COMBO_IDX % (${#VECTORS[@]} * ${#TOP_NS[@]} * ${#HEURISTICS[@]} * ${#GROUP_SIZES[@]})))

VECTOR_IDX=$((REMAINDER / (${#TOP_NS[@]} * ${#HEURISTICS[@]} * ${#GROUP_SIZES[@]})))
REMAINDER=$((REMAINDER % (${#TOP_NS[@]} * ${#HEURISTICS[@]} * ${#GROUP_SIZES[@]})))

TOP_N_IDX=$((REMAINDER / (${#HEURISTICS[@]} * ${#GROUP_SIZES[@]})))
REMAINDER=$((REMAINDER % (${#HEURISTICS[@]} * ${#GROUP_SIZES[@]})))

HEURISTIC_IDX=$((REMAINDER / ${#GROUP_SIZES[@]}))
GROUP_SIZE_IDX=$((REMAINDER % ${#GROUP_SIZES[@]}))

# Retrieve actual values
MODEL="${MODELS[$MODEL_IDX]}"
VECTOR="${VECTORS[$VECTOR_IDX]}"
TOP_N="${TOP_NS[$TOP_N_IDX]}"
HEURISTIC="${HEURISTICS[$HEURISTIC_IDX]}"
GROUP_SIZE="${GROUP_SIZES[$GROUP_SIZE_IDX]}"

# Run the analysis
python "$SCRIPT_ROOT/analyze_activation_geometry.py" \
    -m "$MODEL" \
    --vector "$VECTOR" \
    --group_type "$GROUP_TYPE" \
    --group_size "$GROUP_SIZE" \
    --top_n "$TOP_N" \
    --heuristic "$HEURISTIC" \
    --resume

echo "Analysis complete for combination $SLURM_ARRAY_TASK_ID"
