#!/bin/bash
#SBATCH --job-name=abl_1.4b
#SBATCH --export=ALL
#SBATCH --partition=erc-cristia
#SBATCH --mem=180G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/ablation/pythia_1.4b.log


# Script and config paths
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/ablation"

# Define the configuration files
CONFIG_NAME="config_unigram_ablations_1.4B.yaml"


INTERVAL=1
START=138

# Run the script with the appropriate configuration
python $SCRIPT_ROOT/ablate_neuron.py --interval $INTERVAL --config_name $CONFIG_NAME --start $START --resume --last_N_step 3

