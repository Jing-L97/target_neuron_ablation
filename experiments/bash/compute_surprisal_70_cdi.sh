#!/bin/bash
#SBATCH --job-name=h70_cdi2
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/surprisal/h70_cdi2.log


SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/surprisal"
MODEL="EleutherAI/pythia-70m-deduped"

WORD="context/stas/c4-en-10k/5/cdi_childes.json"
python $SCRIPT_ROOT/compute_surprisal.py -m $MODEL -w $WORD -n 500_1000.csv -a random --resume



