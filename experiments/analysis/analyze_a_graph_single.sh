#!/bin/bash
#SBATCH --job-name=modularity_gpt2
#SBATCH --export=ALL
#SBATCH --partition=cpu
#SBATCH --mem=70G
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/analysis/modularity_gpt2_%a.log
#SBATCH --array=0-8  # Update if number of combinations changes

# Define constants
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/analysis"
SEL_FREQ="longtail_50"
# Define parameter arrays
MODELS=("gpt2" "gpt2-large" "gpt2-xl")
VECTORS=("longtail_50")
TOP_NS=(10 50 100)
HEURISTICS=("prob")
GROUP_SIZES=("best") #"target_size"
GROUP_TYPES=("individual") #"group"
STEP_MODE="single"

# Calculate total number of combinations
TOTAL_COMBINATIONS=$((${#MODELS[@]} * ${#VECTORS[@]} * ${#TOP_NS[@]} * ${#HEURISTICS[@]} * ${#GROUP_SIZES[@]} * ${#GROUP_TYPES[@]}))

# Validate array task ID
if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]]; then
  echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($TOTAL_COMBINATIONS)"
  exit 1
fi

# Calculate indices
COMBO_IDX=$SLURM_ARRAY_TASK_ID

MODEL_IDX=$((COMBO_IDX / (${#VECTORS[@]} * ${#TOP_NS[@]} * ${#HEURISTICS[@]} * ${#GROUP_SIZES[@]} * ${#GROUP_TYPES[@]})))
REMAINDER=$((COMBO_IDX % (${#VECTORS[@]} * ${#TOP_NS[@]} * ${#HEURISTICS[@]} * ${#GROUP_SIZES[@]} * ${#GROUP_TYPES[@]})))

VECTOR_IDX=$((REMAINDER / (${#TOP_NS[@]} * ${#HEURISTICS[@]} * ${#GROUP_SIZES[@]} * ${#GROUP_TYPES[@]})))
REMAINDER=$((REMAINDER % (${#TOP_NS[@]} * ${#HEURISTICS[@]} * ${#GROUP_SIZES[@]} * ${#GROUP_TYPES[@]})))

TOP_N_IDX=$((REMAINDER / (${#HEURISTICS[@]} * ${#GROUP_SIZES[@]} * ${#GROUP_TYPES[@]})))
REMAINDER=$((REMAINDER % (${#HEURISTICS[@]} * ${#GROUP_SIZES[@]} * ${#GROUP_TYPES[@]})))

HEURISTIC_IDX=$((REMAINDER / (${#GROUP_SIZES[@]} * ${#GROUP_TYPES[@]})))
REMAINDER=$((REMAINDER % (${#GROUP_SIZES[@]} * ${#GROUP_TYPES[@]})))

GROUP_SIZE_IDX=$((REMAINDER / ${#GROUP_TYPES[@]}))
GROUP_TYPE_IDX=$((REMAINDER % ${#GROUP_TYPES[@]}))

# Retrieve actual parameter values
MODEL="${MODELS[$MODEL_IDX]}"
VECTOR="${VECTORS[$VECTOR_IDX]}"
TOP_N="${TOP_NS[$TOP_N_IDX]}"
HEURISTIC="${HEURISTICS[$HEURISTIC_IDX]}"
GROUP_SIZE="${GROUP_SIZES[$GROUP_SIZE_IDX]}"
GROUP_TYPE="${GROUP_TYPES[$GROUP_TYPE_IDX]}"

# Log selected configuration
echo "Running analysis with:"
echo " Model: $MODEL"
echo " Vector: $VECTOR"
echo " Top-N: $TOP_N"
echo " Heuristic: $HEURISTIC"
echo " Group Size: $GROUP_SIZE"
echo " Group Type: $GROUP_TYPE"
echo " Combination Index: $SLURM_ARRAY_TASK_ID of $TOTAL_COMBINATIONS"

# Run the analysis script
python "$SCRIPT_ROOT/activation_modularity.py" \
  -m "$MODEL" \
  --vector "$VECTOR" \
  --group_type "$GROUP_TYPE" \
  --group_size "$GROUP_SIZE" \
  --top_n "$TOP_N" \
  --heuristic "$HEURISTIC" \
  --sel_longtail "$SEL_FREQ" \
  --step_mode "$STEP_MODE" \
  --resume

echo "Analysis complete for combination $SLURM_ARRAY_TASK_ID"
