#!/bin/bash
#SBATCH --job-name=abl_analysis
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --exclude=puck5
#SBATCH --cpus-per-task=10
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/ablation/50_100.log


SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/ablations"

python $SCRIPT_ROOT/analyze_unigram.py -m EleutherAI/pythia-70m-deduped --top_n 500
python $SCRIPT_ROOT/analyze_unigram.py -m EleutherAI/pythia-70m-deduped --top_n 1000
python $SCRIPT_ROOT/analyze_unigram.py -m EleutherAI/pythia-410m-deduped --top_n 500
python $SCRIPT_ROOT/analyze_unigram.py -m EleutherAI/pythia-410m-deduped --top_n 1000