#!/bin/bash
#SBATCH --job-name=abl70
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --exclude=puck5
#SBATCH --mem=140G
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/ablation/abl70_%a.log
#SBATCH --array=0-2

SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/ablation"


# Run the script with the appropriate parameters
python $SCRIPT_ROOT/ablate_unigram.py --config config_unigram_ablations_70.yaml --debug