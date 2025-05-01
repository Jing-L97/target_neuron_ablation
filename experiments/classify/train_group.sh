#!/bin/bash
#SBATCH --job-name=hyperplane
#SBATCH --export=ALL
#SBATCH --partition=cpu
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/classify/hyper_group_%a.log
#SBATCH --array=0-5 # Total combinations 

# Script root path
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/classify"
LABEL_TYPE="fixed"
GROUP_TYPE="group"

# Define configuration arrays
MODELS=(
    "EleutherAI/pythia-70m-deduped"
    "EleutherAI/pythia-410m-deduped"
)

GROUP_SIZES=(
    "best"
    "target_size"
)

CLASS_NUMS=(
    2
    3
)

# Calculate total combinations
TOTAL_COMBINATIONS=$((${#MODELS[@]} * ${#GROUP_SIZES[@]} * ${#CLASS_NUMS[@]}))

# Validate array task ID
if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($TOTAL_COMBINATIONS)"
    exit 1
fi

# Calculate indices based on task ID
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / (${#CLASS_NUMS[@]} * ${#GROUP_SIZES[@]}) ))
REMAINDER=$(( SLURM_ARRAY_TASK_ID % (${#CLASS_NUMS[@]} * ${#GROUP_SIZES[@]}) ))
GROUP_SIZE_IDX=$(( REMAINDER / ${#CLASS_NUMS[@]} ))
CLASS_NUM_IDX=$(( REMAINDER % ${#CLASS_NUMS[@]} ))

# Select configuration based on calculated indices
MODEL="${MODELS[$MODEL_IDX]}"
GROUP_SIZE="${GROUP_SIZES[$GROUP_SIZE_IDX]}"
CLASS_NUM="${CLASS_NUMS[$CLASS_NUM_IDX]}"

# Log selected configuration
echo "Processing combination $SLURM_ARRAY_TASK_ID of $TOTAL_COMBINATIONS:"
echo "- Model: $MODEL"
echo "- Group Size: $GROUP_SIZE"
echo "- Class Number: $CLASS_NUM"
echo "- Group Type: $GROUP_TYPE"
echo "- Label Type: $LABEL_TYPE"

# Run the analysis script
python "$SCRIPT_ROOT/train.py" \
    -m "$MODEL" \
    --group_type "$GROUP_TYPE" \
    --label_type "$LABEL_TYPE" \
    --class_num "$CLASS_NUM" \
    --group_size "$GROUP_SIZE" \
    --resume

echo "Analysis complete for configuration: Model=$MODEL, Group_Size=$GROUP_SIZE, Class_Num=$CLASS_NUM"