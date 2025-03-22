#!/bin/bash
#SBATCH --job-name=h_70
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --mem=70G
#SBATCH --exclude=puck5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/surprisal/h%a.log
#SBATCH --array=0-15

SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/surprisal"

WORD="context/stas/c4-en-10k/5/merged.json"


MODEL="EleutherAI/pythia-70m-deduped"

MODE="longtail"

ABLATION="zero"
python $SCRIPT_ROOT/compute_surprisal.py -m $MODEL -w $WORD -n 500_10.csv -a $ABLATION --vector $MODE --resume 
python $SCRIPT_ROOT/compute_surprisal.py -m $MODEL -w $WORD -n 500_50.csv -a $ABLATION --vector $MODE --resume 
python $SCRIPT_ROOT/compute_surprisal.py -m $MODEL -w $WORD -n 500_100.csv -a $ABLATION --vector $MODE --resume 
python $SCRIPT_ROOT/compute_surprisal.py -m $MODEL -w $WORD -n 500_500.csv -a $ABLATION --vector $MODE --resume 

ABLATION="mean"
python $SCRIPT_ROOT/compute_surprisal.py -m $MODEL -w $WORD -n 500_10.csv -a $ABLATION --vector $MODE --resume 
python $SCRIPT_ROOT/compute_surprisal.py -m $MODEL -w $WORD -n 500_50.csv -a $ABLATION --vector $MODE --resume 
python $SCRIPT_ROOT/compute_surprisal.py -m $MODEL -w $WORD -n 500_100.csv -a $ABLATION --vector $MODE --resume 
python $SCRIPT_ROOT/compute_surprisal.py -m $MODEL -w $WORD -n 500_500.csv -a $ABLATION --vector $MODE --resume 



MODE="mean"

ABLATION="mean"
python $SCRIPT_ROOT/compute_surprisal.py -m $MODEL -w $WORD -n 500_10.csv -a $ABLATION --vector $MODE --resume 
python $SCRIPT_ROOT/compute_surprisal.py -m $MODEL -w $WORD -n 500_50.csv -a $ABLATION --vector $MODE --resume 
python $SCRIPT_ROOT/compute_surprisal.py -m $MODEL -w $WORD -n 500_100.csv -a $ABLATION --vector $MODE --resume 
python $SCRIPT_ROOT/compute_surprisal.py -m $MODEL -w $WORD -n 500_500.csv -a $ABLATION --vector $MODE --resume 






MODEL="EleutherAI/pythia-410m-deduped"


MODE="mean"

ABLATION="mean"
python $SCRIPT_ROOT/compute_surprisal.py -m $MODEL -w $WORD -n 500_10.csv -a $ABLATION --vector $MODE --resume 
python $SCRIPT_ROOT/compute_surprisal.py -m $MODEL -w $WORD -n 500_50.csv -a $ABLATION --vector $MODE --resume 
python $SCRIPT_ROOT/compute_surprisal.py -m $MODEL -w $WORD -n 500_100.csv -a $ABLATION --vector $MODE --resume 
python $SCRIPT_ROOT/compute_surprisal.py -m $MODEL -w $WORD -n 500_500.csv -a $ABLATION --vector $MODE --resume 
