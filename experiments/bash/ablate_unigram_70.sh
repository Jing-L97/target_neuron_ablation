#!/bin/bash
#SBATCH --job-name=abl70
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --exclude=puck5
#SBATCH --mem=140G
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --time=1-20:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/ablation/abl70.log


SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/ablations"
python $SCRIPT_ROOT/ablate_unigram.py --config config_unigram_ablations_70.yaml
