#!/bin/bash
#SBATCH --job-name=sel_group
#SBATCH --export=ALL
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=30:00:00
#SBATCH --output=/scratch2/jliu/Generative_replay/neuron/logs/selection/sel_group_%a.log
#SBATCH --array=0-279%24  # adjust depending on total combinations

SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/selection"

# Define arrays
EFFECTS=("boost","suppress")

TOP_NS=(-1)
MODELS=("gpt2" 
        "gpt2-medium" 
        "gpt2-large" 
        "gpt2-xl"
        "EleutherAI/pythia-1B-deduped"
        "EleutherAI/pythia-1.4B-deduped"
        "EleutherAI/pythia-2.8B-deduped"
        # "EleutherAI/pythia-70m-deduped"
        # "EleutherAI/pythia-160m-deduped"
        # "EleutherAI/pythia-410m-deduped"

        )


# Define min/max percentile pairs as "MAX,MIN"
PCT_PAIRS=(
    "0,5" 
    "95,100"
    "0,10" 
    "90,100"
    "0,15" 
    "85,100"
    "0,20" 
    "80,100"
    "0,25" 
    "75,100"
    "0,30" 
    "70,100"
    "0,35" 
    "65,100"
    "0,40" 
    "60,100"
    "0,45" 
    "55,100"
    "0,50" 
    "50,100"
    )  # format: max_percentile,min_percentile

# Total combinations
TOTAL_COMBINATIONS=$((${#EFFECTS[@]}  * ${#TOP_NS[@]} * ${#MODELS[@]} * ${#PCT_PAIRS[@]}))

if [[ $SLURM_ARRAY_TASK_ID -ge $TOTAL_COMBINATIONS ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total combinations ($TOTAL_COMBINATIONS)"
    exit 1
fi

# Compute indices
EFFECT_IDX=$(( SLURM_ARRAY_TASK_ID / (${#VECTORS[@]} * ${#TOP_NS[@]} * ${#MODELS[@]} * ${#PCT_PAIRS[@]}) ))

TOP_N_IDX=$(( (SLURM_ARRAY_TASK_ID / (${#MODELS[@]} * ${#PCT_PAIRS[@]})) % ${#TOP_NS[@]} ))
MODEL_IDX=$(( (SLURM_ARRAY_TASK_ID / ${#PCT_PAIRS[@]}) % ${#MODELS[@]} ))
PCT_PAIR_IDX=$(( SLURM_ARRAY_TASK_ID % ${#PCT_PAIRS[@]} ))

# Assign values
EFFECT="${EFFECTS[$EFFECT_IDX]}"

TOP_N="${TOP_NS[$TOP_N_IDX]}"
MODEL="${MODELS[$MODEL_IDX]}"
MAX_RANK_PERCENTILE=$(echo "${PCT_PAIRS[$PCT_PAIR_IDX]}" | cut -d',' -f1)
MIN_RANK_PERCENTILE=$(echo "${PCT_PAIRS[$PCT_PAIR_IDX]}" | cut -d',' -f2)

# Log info
echo "Processing combination:"
echo " Model: $MODEL"
echo " Top N: $TOP_N"
echo " Max Rank Percentile: $MAX_RANK_PERCENTILE"
echo " Min Rank Percentile: $MIN_RANK_PERCENTILE"

# Run the analysis with final Python arguments
python $SCRIPT_ROOT/sel_neuron.py \
    -m "$MODEL" \
    --effect "$EFFECT" \
    --top_n "$TOP_N" \
    --min_rank_pct "$MIN_RANK_PERCENTILE" \
    --max_rank_pct "$MAX_RANK_PERCENTILE" 

