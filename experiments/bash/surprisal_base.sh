#!/bin/bash
#SBATCH --job-name=h_debug
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/surprisal/pythia_debug.log


SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/surprisal"
python $SCRIPT_ROOT/base_surprisal.py --debug
