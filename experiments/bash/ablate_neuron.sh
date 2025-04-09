#!/bin/bash
#SBATCH --job-name=abl_tail
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --exclude=puck5
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/ablation/tail50_%a.log
#SBATCH --array=0-1

# Script and config paths
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/ablation"

# Define the configuration files
CONFIG_NAMES=(
  "config_unigram_ablations_70.yaml"
  "config_unigram_ablations_410.yaml"
)

# Map array task ID to configuration file
CONFIG_IDX=$SLURM_ARRAY_TASK_ID
CONFIG_NAME="${CONFIG_NAMES[$CONFIG_IDX]}"

echo "Running job array task $SLURM_ARRAY_TASK_ID with configuration: $CONFIG_NAME"

# Run the script with the appropriate configuration
python $SCRIPT_ROOT/ablate_neuron.py --config_name $CONFIG_NAME --resume