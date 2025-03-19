#!/bin/bash
#SBATCH --job-name=abl410
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --mem=500G
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/ablation/abl410%a.log
#SBATCH --array=0-1

SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/ablations"

# Define start and end values for each array task
if [ "$SLURM_ARRAY_TASK_ID" -eq "0" ]; then
    START=0
    END=70
elif [ "$SLURM_ARRAY_TASK_ID" -eq "1" ]; then
    START=70
    END=155

fi

# Run the script with the appropriate parameters
python $SCRIPT_ROOT/ablate_unigram.py --start $START --end $END --config config_unigram_ablations.yaml


