#!/bin/bash
#SBATCH --job-name=hyperplane_debug
#SBATCH --export=ALL
#SBATCH --partition=cpu
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/classify/hyper_group_%a.log


# Script root path
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/classify"

# Run the analysis script
python "$SCRIPT_ROOT/train.py" 