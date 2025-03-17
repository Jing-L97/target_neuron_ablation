#!/bin/bash
#SBATCH --job-name=abl410
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --exclude=puck5
#SBATCH --mem=240G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40
#SBATCH --time=1-20:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/ablation/abl410.log


SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/ablations"
python $SCRIPT_ROOT/ablate_unigram.py
