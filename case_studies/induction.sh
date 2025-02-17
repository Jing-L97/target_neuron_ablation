#!/bin/bash
#SBATCH --job-name=induction
#SBATCH --export=ALL
#SBATCH --exclude=puck5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=2-00:00:00
# Array Number of Jobs to run in Parallel
# Given via CMD arguments (because it varies depending on the number of jobs)
#SBATCH --output=../logs/induction.log



FILE_ROOT=/scratch2/jliu/Generative_replay/neuron/confidence-regulation-neurons
python $FILE_ROOT/case_studies/induction.py