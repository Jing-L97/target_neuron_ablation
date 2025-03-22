#!/bin/bash
#SBATCH --job-name=70tz
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --mem=70G
#SBATCH --exclude=puck5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=40:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/surprisal/70tz_%a.log
#SBATCH --array=0-2

# Define constants for better readability and maintenance
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/surprisal"
WORD="context/stas/c4-en-10k/5/merged.json"
MODEL="EleutherAI/pythia-70m-deduped"
MODE="longtail"
ABLATION="zero"

# Define the neuron files in an array
NEURON_FILES=(
    "500_10.csv"
    "500_50.csv"
    "500_500.csv"
)

# Use the SLURM array task ID to select the appropriate neuron file
NEURON_FILE="${NEURON_FILES[$SLURM_ARRAY_TASK_ID]}"

# Log which file is being processed
echo "Processing neuron file: $NEURON_FILE"

# Run the surprisal computation with the selected neuron file
python $SCRIPT_ROOT/compute_surprisal.py \
    -m $MODEL \
    -w $WORD \
    -n $NEURON_FILE \
    -a $ABLATION \
    --vector $MODE \
    --resume