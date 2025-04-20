#!/bin/bash
#SBATCH --job-name=abl_tail
#SBATCH --export=ALL
#SBATCH --partition=cpu
#SBATCH --mem=120G
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/ablation/tail50_%a.log


# Script and config paths
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/ablation"

# Run the script with the appropriate configuration
python $SCRIPT_ROOT/ablate_neuron.py --config_name "config_unigram_ablations_70.yaml"