#!/bin/bash
#SBATCH --job-name=htsr_gptxl
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/analysis/htsr_gptxl_%a.log
#SBATCH --array=0-2  # Update if number of combinations changes



output=/scratch2/jliu/Generative_replay/neuron/logs/analysis/geometry_gpt2L_%a.log

# Define constants
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/analysis"

STEP_MODE="single"
MODEL="gpt2-large"
TOP_N=35
# Run the analysis script
python "$SCRIPT_ROOT/activation_geometry.py" \
  -m "$MODEL" \
  --top_n "$TOP_N" \
  --step_mode "$STEP_MODE" \
  --resume


MODEL="gpt2-xl"
TOP_N=30
# Run the analysis script
python "$SCRIPT_ROOT/activation_geometry.py" \
  -m "$MODEL" \
  --top_n "$TOP_N" \
  --step_mode "$STEP_MODE" \
  --resume



STEP_MODE="multi"
MODEL="EleutherAI/pythia-1B-deduped"
VECTOR="longtail_0_50"
TOP_N=50
# Run the analysis script
python "$SCRIPT_ROOT/activation_geometry.py" \
  -m "$MODEL" \
  --top_n "$TOP_N" \
  --step_mode "$STEP_MODE" \
  --vector "$VECTOR" \
  --resume



STEP_MODE="multi"
MODEL="EleutherAI/pythia-2.8B-deduped"
VECTOR="longtail_0_50"
TOP_N=50
# Run the analysis script
python "$SCRIPT_ROOT/activation_geometry.py" \
  -m "$MODEL" \
  --top_n "$TOP_N" \
  --step_mode "$STEP_MODE" \
  --vector "$VECTOR" \
  --resume
