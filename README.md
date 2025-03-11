# Neuron intervention for long-tail lifting

## Setup

To install the required packages, run:

    pip install -r requirements.txt

For accessing gated repositories (e.g., LLaMA), add your HuggingFace token to `./ablations/hf_token.txt`.

## Ablation Experiments

The `scripts/ablations/` folder contains code for performing neuron ablation experiments to identify token frequency neurons. We use Hydra for parameter configuration. The config files are available in `experiments/config/`

- `ablate_unigram.py`: Runs mean ablations to quantify the total vs direct effect for token frequency neurons.
- `analyze_unigram.py`: Select the token frequency neurons based on the mediation effect and KL divergence. 
- `scale_neuron/`: Analyze null space of common tokens and the neuron scaling effect


## Metric
The `scripts/surprisal/` folder contains code for computing the target surprisal-based metrics. 
- `prepare_context.py`: Extract the target context of the given  word list
- `compute_surprisal.py`: Compute surprisal conditioned on the given text, across differnt training steps. 


## Citing this Work
If you find this work useful in your research, please consider citing our paper:

    @article{
    }

