#!/bin/bash
#SBATCH --job-name=KL_geometry
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --exclude=puck5
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/analysis/KL_geometry_%a.log
#SBATCH --array=0-11

# Define constants
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/analysis"
Sel_by_Med="True"

# Define arrays
MODELS=(
  "EleutherAI/pythia-70m-deduped"
  "EleutherAI/pythia-410m-deduped"
)
EFFECTS=(
  "suppress"
  "boost"
)
VECTORS=(
  "longtail_50"
)
TOP_NS=(
  10
  50
  100
)

HEURISTICS=(
  "KL"
)

# Calculate total combinations
TOTAL_MODELS=${#MODELS[@]}
TOTAL_EFFECTS=${#EFFECTS[@]}
TOTAL_VECTORS=${#VECTORS[@]}
TOTAL_TOP_NS=${#TOP_NS[@]}
TOTAL_HEURISTICS=${#HEURISTICS[@]}
TOTAL_COMBINATIONS=$((TOTAL_MODELS * TOTAL_EFFECTS * TOTAL_VECTORS * TOTAL_TOP_NS * TOTAL_HEURISTICS))

# Safety check
if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]]; then
  echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($TOTAL_COMBINATIONS)"
  exit 1
fi

# Calculate indices for each parameter
model_idx=$(( SLURM_ARRAY_TASK_ID / (TOTAL_EFFECTS * TOTAL_VECTORS * TOTAL_TOP_NS) ))
effect_idx=$(( (SLURM_ARRAY_TASK_ID / (TOTAL_VECTORS * TOTAL_TOP_NS)) % TOTAL_EFFECTS ))
vector_idx=$(( (SLURM_ARRAY_TASK_ID / TOTAL_TOP_NS) % TOTAL_VECTORS ))
top_n_idx=$(( SLURM_ARRAY_TASK_ID % TOTAL_TOP_NS ))
heuristic_idx=$(( SLURM_ARRAY_TASK_ID % TOTAL_HEURISTICS ))

# Select parameters
MODEL="${MODELS[$model_idx]}"
EFFECT="${EFFECTS[$effect_idx]}"
VECTOR="${VECTORS[$vector_idx]}"
TOP_N="${TOP_NS[$top_n_idx]}"
HEURISTIC="${HEURISTICS[$heuristic_idx]}"

# Log what's happening
echo "Processing parameters:"
echo "  Model: $MODEL"
echo "  Effect: $EFFECT"
echo "  Vector: $VECTOR"
echo "  Top_N: $TOP_N"
echo "  Heuristic: $HEURISTIC"
echo "  Task ID: $SLURM_ARRAY_TASK_ID of $TOTAL_COMBINATIONS"

# Run the Python analysis
python "$SCRIPT_ROOT/analyze_activation_geometry.py" \
  -m "$MODEL" \
  --effect "$EFFECT" \
  --vector "$VECTOR" \
  --top_n "$TOP_N" \
  --heuristic  "$HEURISTIC" 

echo "Analysis complete for combination $SLURM_ARRAY_TASK_ID"