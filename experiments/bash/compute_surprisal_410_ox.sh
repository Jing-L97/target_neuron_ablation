#!/bin/bash
#SBATCH --job-name=h410_ox2
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --exclude=puck5
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/surprisal/h410_ox2.log


SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/surprisal"
MODEL="EleutherAI/pythia-410m-deduped"
WORD="context/stas/c4-en-10k/5/oxford-understand.json"


python $SCRIPT_ROOT/compute_surprisal.py -m $MODEL -w $WORD -n 500_1000.csv -a zero --resume
python $SCRIPT_ROOT/compute_surprisal.py -m $MODEL -w $WORD -n 500_1000.csv -a random --resume

python $SCRIPT_ROOT/compute_surprisal.py -m $MODEL -w $WORD -n 500_500.csv -a random --resume
python $SCRIPT_ROOT/compute_surprisal.py -m $MODEL -w $WORD -n 500_500.csv -a zero --resume