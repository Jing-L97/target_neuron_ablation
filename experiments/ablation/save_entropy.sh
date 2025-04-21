#!/bin/bash
#SBATCH --job-name=save_entropy
#SBATCH --export=ALL
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --time=8:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/ablation/save_entropy.log


# Script and config paths
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/ablation"

# Run the script with the appropriate configuration
python $SCRIPT_ROOT/save_entropy.py --config_name "config_unigram_ablations_410.yaml" --resume