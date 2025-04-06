# Experiments

## Neuron identification
Step 1: Run the causal mediation analyses on all the neurons in the last layer


```
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/ablations"

# Run the script with the appropriate parameters
python $SCRIPT_ROOT/ablate_unigram.py --start $START --end $END --config config_unigram_ablations_70.yaml
```

Step 2: Select neurons based on the given heuristics

```

SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/ablations"

# Define the input arrays
EFFECTS=(
    "suppress"
    "boost"
)

VECTORS=(
    "longtail"
)

TOP_NS=(
    10
    50
)

MODELS=(
    "EleutherAI/pythia-70m-deduped"
    "EleutherAI/pythia-410m-deduped"
)

# Run the analysis with the selected combination
python $SCRIPT_ROOT/analyze_unigram.py \
    -m "$MODEL" \
    --effect "$EFFECT" \
    --top_n "$TOP_N" \
    --vector "$VECTOR"
```


## Neuron intervention

```

# Define constants for better readability and maintenance
SCRIPT_ROOT="/scratch2/jliu/Generative_replay/neuron/target_neuron_ablation/src/scripts/surprisal"

# Define constants
EFFECTS=(
  "suppress"
  "boost"
)

# Define the input arrays
VECTORS=(
  "longtail"
)

# Fixed word variable
WORD="context/stas/c4-en-10k/5/longtail_words.json"

NEURON_FILES=(
  "500_10.csv"
  "500_50.csv"
)

ABLATIONS=(
  "mean"
  "zero"
)

# Fixed model variable
MODEL="EleutherAI/pythia-70m-deduped"


# Run the surprisal computation with the selected combination
python $SCRIPT_ROOT/compute_surprisal.py \
  -m "$MODEL" \
  -w "$WORD" \
  -n "$NEURON_FILE" \
  -a "$ABLATION" \
  --effect "$EFFECT" \
  --vector "$VECTOR" \
  --resume
```


## Evaluation metrics
# Build longtail word set
Step 1: selet long-tail words based on the given threhsold, implement with this script:

```
python src/scripts/preprocess/select_longtail.py
```

Step 2: selet context from the given corpus, implement with this script:

```
python src/scripts/preprocess/select_longtail.py 
        -w freq/EleutherAI/pythia-410m/longtail_words.csv
```
