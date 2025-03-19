#!/bin/bash
#SBATCH --job-name=h_cdi_b
#SBATCH --export=ALL
#SBATCH --partition=cpu
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/surprisal/h_cdi_b.log


SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/format"

python $SCRIPT_ROOT/format_surprisal.py --neuron base --eval cdi_childes
