#!/bin/bash
#SBATCH --job-name=h_410_cdi
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/surprisal/h_410_cdi.log


SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/surprisal"
MODEL="EleutherAI/pythia-410m-deduped"
WORD="context/stas/c4-en-10k/5/cdi_childes.json"

python $SCRIPT_ROOT/compute_surprisal.py -m $MODEL -w $WORD -n 500_10.csv -a zero
python $SCRIPT_ROOT/compute_surprisal.py -m $MODEL -w $WORD -n 500_100.csv -a zero
