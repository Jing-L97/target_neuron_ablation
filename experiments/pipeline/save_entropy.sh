#!/bin/bash
#SBATCH --job-name=save_entropy
#SBATCH --export=ALL
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --time=8:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/ablation/save_entropy.log

MODEL=
# preprocess the results
python src/scripts/preprocess/select_longtail.py --model gpt2

# run ablation experiment
