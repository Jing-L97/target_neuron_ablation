#!/bin/bash
#SBATCH --job-name=70ts_tail
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/surprisal/70ts_tail.log

# Define constants for better readability and maintenance
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/surprisal"
WORD="context/stas/c4-en-10k/5/longtail_words.json"
MODEL="EleutherAI/pythia-70m-deduped"
VECTOR="longtail"
ABLATION="scaled"


# Run the surprisal computation with the selected neuron file
python $SCRIPT_ROOT/compute_surprisal.py \
    -m $MODEL \
    -w $WORD \
    -n "500_500.csv" \
    -a $ABLATION \
    --vector $VECTOR \
    --resume

