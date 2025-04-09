#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/surprisal/preprocess.log


SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/preprocess"

python $SCRIPT_ROOT/prepare_context.py -w freq/EleutherAI/pythia-410m/longtail_50.csv
