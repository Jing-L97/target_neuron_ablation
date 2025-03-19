#!/bin/bash
#SBATCH --job-name=410z_ox_50
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --mem=70G
#SBATCH --exclude=puck5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=4:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/surprisal/410z_ox_50.log


SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/surprisal"
MODEL="EleutherAI/pythia-410m-deduped"

WORD="context/stas/c4-en-10k/5/oxford-understand.json"


python $SCRIPT_ROOT/compute_surprisal.py -m $MODEL -w $WORD -n 500_50.csv -a zero --resume 
