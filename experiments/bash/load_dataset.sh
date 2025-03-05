#!/bin/bash
#SBATCH --job-name=load_data
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --mem=60G
#SBATCH --cpus-per-task=6
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/surprisal/load_data.log


SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/preprocess"

python $SCRIPT_ROOT/filter_freq.py 

