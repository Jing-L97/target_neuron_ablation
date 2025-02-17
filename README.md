# Confidence Regulation Neurons in Language Models

This repository contains the code for the NeurIPS 2024 paper “Confidence Regulation Neurons in Language Models.”

![Sample Figure](./assets/)

## Setup

To install the required packages, run:

    pip install -r requirements.txt

For accessing gated repositories (e.g., LLaMA), add your HuggingFace token to `./ablations/hf_token.txt`.

## Ablation Experiments

The `ablations/` folder contains code for performing neuron ablation experiments to identify entropy and token frequency neurons. We use Hydra for parameter configuration. The config files are available in ablations/config/. These scripts should be executed from the `ablations` folder.


- `run_and_store_ablation_results.py`: Runs mean ablations to quantify the total vs direct effect for entropy neurons.
- `run_and_store_unigram_results.py`: Runs mean ablations to quantify the total vs direct effect for token frequency neurons.
- `load_results.py` and `load_unigram_results.py`: For visualizing the results of the mean ablation experiments.
- `datasets/`: Contains `.npy` files with token counts for OpenWebText and The Pile, used to compute the token frequency distribution.

## Citing this Work
If you find this work useful in your research, please consider citing our paper:

    @article{
    }

