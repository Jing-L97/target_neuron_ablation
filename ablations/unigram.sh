#!/bin/bash
#SBATCH --job-name=unigram
#SBATCH --export=ALL
#SBATCH --exclude=puck5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=2-00:00:00
#SBATCH --output=../logs/unigram.log



FILE_ROOT=/scratch2/jliu/Generative_replay/neuron/confidence-regulation-neurons
python $FILE_ROOT/ablations/run_and_store_unigram_results.py
