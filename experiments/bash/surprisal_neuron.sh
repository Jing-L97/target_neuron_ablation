#!/bin/bash
#SBATCH --job-name=h_neuron_debug
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/surprisal/pythia_70_neuron_debug.log


SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/surprisal"
python $SCRIPT_ROOT/neuron_surprisal.py -m EleutherAI/pythia-70m-deduped --debug
