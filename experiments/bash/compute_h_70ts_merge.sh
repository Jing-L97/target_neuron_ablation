#!/bin/bash
#SBATCH --job-name=70ms_merge
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/surprisal/70ms_merge%a.log
#SBATCH --array=0-1

# Define constants for better readability and maintenance
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/surprisal"
WORD="context/stas/c4-en-10k/5/merged.json"
MODEL="EleutherAI/pythia-70m-deduped"
VECTOR="longtail"
ABLATION="scaled"

# Define the neuron files in an array
NEURON_FILES=(
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
    --vector $VECTOR \
    --resume

