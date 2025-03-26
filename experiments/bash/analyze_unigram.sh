#!/bin/bash
#SBATCH --job-name=sel_neuron
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=3:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/ablation/longtail.log


SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/ablations"

python $SCRIPT_ROOT/analyze_unigram.py -m EleutherAI/pythia-410m-deduped --top_n 10
python $SCRIPT_ROOT/analyze_unigram.py -m EleutherAI/pythia-410m-deduped --top_n 50
python $SCRIPT_ROOT/analyze_unigram.py -m EleutherAI/pythia-410m-deduped --top_n 500
python $SCRIPT_ROOT/analyze_unigram.py -m EleutherAI/pythia-410m-deduped --top_n 1000