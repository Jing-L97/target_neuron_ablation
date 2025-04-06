# Neuron intervention for long-tail lifting

## Setup

To install the required packages, run:

    pip install -r requirements.txt

For accessing gated repositories (e.g., LLaMA), add your HuggingFace token to `./ablations/hf_token.txt`.

## Ablation Experiments

The `scripts/ablations/` folder contains code for performing neuron ablation experiments to identify token frequency neurons. We use Hydra for parameter configuration. The config files are available in `experiments/config/`

- `ablate_unigram.py`: Runs mean ablations to quantify the total vs direct effect for token frequency neurons.

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


- `analyze_unigram.py`: Select the token frequency neurons based on the mediation effect and KL divergence. 


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

- `scale_neuron/`: Analyze null space of common tokens and the neuron scaling effect


## Metric
The `scripts/surprisal/` folder contains code for computing the target surprisal-based metrics. 
- `prepare_context.py`: Extract the target context of the given  word list
- `compute_surprisal.py`: Compute surprisal conditioned on the given text, across differnt training steps. 

### Build longtail word set
Step 1: selet long-tail words based on the given threhsold, implement with this script:

```
python src/scripts/preprocess/select_longtail.py
```

Step 2: selet context from the given corpus, implement with this script:

```
python src/scripts/preprocess/select_longtail.py 
        -w freq/EleutherAI/pythia-410m/longtail_words.csv
```


## Citing this Work
If you find this work useful in your research, please consider citing our paper:

    @article{
    }

