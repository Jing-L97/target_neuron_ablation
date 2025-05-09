#!/bin/bash
#SBATCH --job-name=ind_ref
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=70G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/classify/ind_ref_%a.log
#SBATCH --array=0-5 # Updated to match total combinations

# Script root path
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/classify"
GROUP_TYPE="individual"

# Define configuration arrays
MODELS=(
    "EleutherAI/pythia-70m-deduped"
    "EleutherAI/pythia-410m-deduped"
)


CLASS_NUMS=(
    2
    
)

TOP_NS=(
    100
    50
    10
)

# Calculate total combinations
TOTAL_COMBINATIONS=$((${#MODELS[@]} * ${#CLASS_NUMS[@]} * ${#TOP_NS[@]}))

# Validate array task ID
if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($TOTAL_COMBINATIONS)"
    exit 1
fi

# Calculate indices based on task ID
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / (${#CLASS_NUMS[@]} * ${#TOP_NS[@]}) ))
REMAINDER=$(( SLURM_ARRAY_TASK_ID % (${#CLASS_NUMS[@]} * ${#TOP_NS[@]}) ))
CLASS_NUM_IDX=$(( REMAINDER / ${#TOP_NS[@]} ))
TOP_N_IDX=$(( REMAINDER % ${#TOP_NS[@]} ))

# Select configuration based on calculated indices
MODEL="${MODELS[$MODEL_IDX]}"
CLASS_NUM="${CLASS_NUMS[$CLASS_NUM_IDX]}"
TOP_N="${TOP_NS[$TOP_N_IDX]}"

# Log selected configuration
echo "Processing combination $SLURM_ARRAY_TASK_ID of $TOTAL_COMBINATIONS:"
echo "- Model: $MODEL"
echo "- Class Number: $CLASS_NUM"
echo "- Top N: $TOP_N"
echo "- Group Type: $GROUP_TYPE"
echo "- Label Type: $LABEL_TYPE"

# Run the analysis script
python "$SCRIPT_ROOT/reflection.py" \
    -m "$MODEL" \
    --group_type "$GROUP_TYPE" \
    --class_num "$CLASS_NUM" \
    --resume \
    --top_n "$TOP_N" 

echo "Analysis complete for configuration: Model=$MODEL, Class_Num=$CLASS_NUM, Top_N=$TOP_N"