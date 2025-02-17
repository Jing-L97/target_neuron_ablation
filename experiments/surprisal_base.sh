#!/bin/bash
#SBATCH --job-name=h_base
#SBATCH --export=ALL
#SBATCH --partition=cpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --time=2-00:00:00
#SBATCH --output=../logs/h_base.log



FILE_ROOT=/scratch2/jliu/Generative_replay/neuron/confidence-regulation-neurons
python $FILE_ROOT/develop_surprisal.py
