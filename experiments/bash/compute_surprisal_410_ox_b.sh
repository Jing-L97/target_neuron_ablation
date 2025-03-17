#!/bin/bash
#SBATCH --job-name=h410_ox
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/surprisal/h410_ox.log


SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/surprisal"
MODEL="EleutherAI/pythia-410m-deduped"
WORD="context/stas/c4-en-10k/5/oxford-understand.json"


python $SCRIPT_ROOT/compute_surprisal.py -m $MODEL -w $WORD -a base --resume

