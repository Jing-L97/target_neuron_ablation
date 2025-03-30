#!/bin/bash
#SBATCH --job-name=tm_merge
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --exclude=puck6
#SBATCH --cpus-per-task=16
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/surprisal/tm_merge%a.log
#SBATCH --array=0-1

# Define constants for better readability and maintenance
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/surprisal"
WORD="context/stas/c4-en-10k/5/merged.json"
MODEL=
VECTOR="longtail"
ABLATION="mean"
EFFECT="boost"
NEURON_FILE="500_500.csv"
# Define the neuron files in an array
MODELS=(
    "EleutherAI/pythia-70m-deduped"
    "EleutherAI/pythia-410m-deduped"
)

# Use the SLURM array task ID to select the appropriate neuron file
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

# Log which file is being processed
echo "Processing model: $MODEL"

# Run the surprisal computation with the selected neuron file
python $SCRIPT_ROOT/compute_surprisal.py \
    -m $MODEL \
    -w $WORD \
    -n $NEURON_FILE \
    -a $ABLATION \
    --effect $EFFECT \
    --vector $VECTOR \
    --resume

