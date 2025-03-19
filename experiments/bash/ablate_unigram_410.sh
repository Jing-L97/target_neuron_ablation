#!/bin/bash
#SBATCH --job-name=abl410
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --mem=500G
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/ablation/abl410.log


SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/ablations"

START=0
END=155

# Run the script with the appropriate parameters
python $SCRIPT_ROOT/ablate_unigram.py --start $START --end $END --config config_unigram_ablations.yaml