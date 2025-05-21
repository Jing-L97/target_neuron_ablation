#!/bin/bash
#SBATCH --job-name=abl_gpt2xl
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/ablation/gpt2xl.log


# Script and config paths
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/ablation"

# Define the configuration files
CONFIG_NAMES=(
  "config_unigram_ablations_gpt2-xl.yaml"
)

# Map array task ID to configuration file
CONFIG_IDX=$SLURM_ARRAY_TASK_ID
CONFIG_NAME="${CONFIG_NAMES[$CONFIG_IDX]}"

echo "Running job array task $SLURM_ARRAY_TASK_ID with configuration: $CONFIG_NAME"

# Run the script with the appropriate configuration
python $SCRIPT_ROOT/ablate_neuron_model.py --config_name $CONFIG_NAME --resume