#!/bin/bash
#SBATCH --job-name=direction
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/ablation/direction%a.log
#SBATCH --array=0-27  

# Define constants for better readability and maintenance
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/ablations"
NEURON_FILE="500_500.csv"
VECTOR="mean"
EFFECT="boost"

# Define the models and their corresponding max layer numbers
MODELS=(
"EleutherAI/pythia-70m-deduped"
"EleutherAI/pythia-410m-deduped"
)

MAX_LAYERS=(
5
23
)

# Calculate the model index and layer number based on the array task ID
if [ $SLURM_ARRAY_TASK_ID -lt ${MAX_LAYERS[0]} ]; then
    MODEL_IDX=0
    LAYER_NUM=$((SLURM_ARRAY_TASK_ID + 1))  # +1 because we want layers 1-5, not 0-4
else
    MODEL_IDX=1
    LAYER_NUM=$(($SLURM_ARRAY_TASK_ID - ${MAX_LAYERS[0]} + 1))  # +1 to start from layer 1
fi

MODEL="${MODELS[$MODEL_IDX]}"

# Log which model and layer are being processed
echo "Processing model: $MODEL, layer: $LAYER_NUM"

# Run the direction analysis
python $SCRIPT_ROOT/analyze_direction.py \
-m $MODEL \
-n $NEURON_FILE \
--layer_num $LAYER_NUM \
--effect $EFFECT \
--vector $VECTOR \
--resume