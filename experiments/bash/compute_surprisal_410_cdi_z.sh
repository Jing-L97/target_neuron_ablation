#!/bin/bash
#SBATCH --job-name=h410z_cdi
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/surprisal/h410z_cdi.log


SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/surprisal"
MODEL="EleutherAI/pythia-410m-deduped"
WORD="context/stas/c4-en-10k/5/cdi_childes.json"

python $SCRIPT_ROOT/compute_surprisal.py -m $MODEL -w $WORD -n 500_10.csv -a zero --resume
python $SCRIPT_ROOT/compute_surprisal.py -m $MODEL -w $WORD -n 500_50.csv -a zero --resume
python $SCRIPT_ROOT/compute_surprisal.py -m $MODEL -w $WORD -n 500_100.csv -a zero --resume
python $SCRIPT_ROOT/compute_surprisal.py -m $MODEL -w $WORD -n 500_500.csv -a zero --resume
python $SCRIPT_ROOT/compute_surprisal.py -m $MODEL -w $WORD -n 500_1000.csv -a zero --resume


