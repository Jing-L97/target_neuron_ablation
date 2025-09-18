#!/bin/bash
#SBATCH --job-name=abl_160m
#SBATCH --export=ALL
#SBATCH --partition=erc-cristia
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/ablation/pythia_160m_step.log


# Script and config paths
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/ablation"

# Define the configuration files
CONFIG_NAME="config_unigram_ablations_160M.yaml"


INTERVAL=20
START=14
END=143
# Run the script with the appropriate configuration
python $SCRIPT_ROOT/ablate_neuron.py --interval $INTERVAL --config_name $CONFIG_NAME --start $START --end $END --resume 

