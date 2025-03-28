#!/bin/bash
#SBATCH --job-name=sel410
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --mem=70G
#SBATCH --exclude=puck5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/ablation/sel410_%a.log
#SBATCH --array=0-3

SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/ablations"
MODEL="EleutherAI/pythia-410m-deduped"
EFFECT="boost"

# Define the neuron files in an array
TOP_NS=(
    10
    100
    50
    500
)

# Use the SLURM array task ID to select the appropriate neuron file
TOP_N="${TOP_NS[$SLURM_ARRAY_TASK_ID]}"

# Log which file is being processed
echo "Processing neuron file: $NEURON_FILE"


python $SCRIPT_ROOT/analyze_unigram.py -m $MODEL --effect $EFFECT --top_n $TOP_N
