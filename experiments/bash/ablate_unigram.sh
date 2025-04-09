#!/bin/bash
#SBATCH --job-name=abl70
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --exclude=puck5
#SBATCH --mem=140G
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/ablation/abl70_%a.log
#SBATCH --array=0-2

SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/ablation"

# Define start and end values for each array task
if [ "$SLURM_ARRAY_TASK_ID" -eq "0" ]; then
    START=0
    END=50
elif [ "$SLURM_ARRAY_TASK_ID" -eq "1" ]; then
    START=50
    END=100
elif [ "$SLURM_ARRAY_TASK_ID" -eq "2" ]; then
    START=100
    END=155

fi

# Run the script with the appropriate parameters
python $SCRIPT_ROOT/ablate_unigram.py --start $START --end $END --config config_unigram_ablations_70.yaml --debug
