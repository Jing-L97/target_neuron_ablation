#!/bin/bash
#SBATCH --job-name=save_entropy
#SBATCH --export=ALL
#SBATCH --partition=cpu
#SBATCH --mem=100G
#SBATCH --cpus-per-task=10
#SBATCH --time=1:15:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/ablation/save_entropy.log


# Script and config paths
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/ablation"

# Run the script with the appropriate configuration
python $SCRIPT_ROOT/ablate_neuron.py --config_name "config_unigram_ablations_410.yaml" --resume