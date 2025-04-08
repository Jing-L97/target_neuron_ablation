import argparse
import gc
from pathlib import Path

import numpy as np
import torch
import transformer_lens
from datasets import load_dataset
from transformers import AutoTokenizer

from neuron_analyzer import settings
from neuron_analyzer.null_space import NullSpaceScaler


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract word surprisal across different training steps.")
    parser.add_argument(
        "--freq_path",
        type=Path,
        default="src/unigram/pythia-unigrams.npy",
        help="Relative path to the target words",
    )

    parser.add_argument(
        "-m", "--model_name", type=str, default="EleutherAI/pythia-70m-deduped", help="Target model name"
    )
    parser.add_argument("-o", "--out_path", type=str, default="surprisal/nullspace/abl.csv", help="Target model name")
    parser.add_argument(
        "-a",
        "--ablate",
        type=str,
        default="base",
        choices=["base", "zero", "random"],
        help="Neuron options for computing surprisal",
    )
    parser.add_argument("--debug", action="store_true", help="Compute the first few 5 lines if enabled")
    parser.add_argument("--resume", action="store_true", help="Resume from the existing checkpoint")
    return parser.parse_args()


def main():
    # Set device
    args = parse_args()
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Load the target model
    print(f"Loading {args.model_name} model...")
    model = transformer_lens.HookedTransformer.from_pretrained(args.model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load a subset of the C4 dataset
    print("Loading dataset subset...")
    dataset = load_dataset("stas/c4-en-10k", split="train[:1000]")  # Use only first 1000 examples

    # load unifgram freq as a torch tensor
    token_frequencies = torch.tensor(np.load(settings.PATH.dataset_root / args.freq_path), dtype=torch.float32)
    print(f"Unigram freq has been loaded from {settings.PATH.dataset_root / args.freq_path}")
    # Tokenize a small subset for unigram statistics
    print("Tokenizing dataset...")

    # Scale rare token weights
    print("Scaling rare token weights...")
    freq_scaler = NullSpaceScaler(
        model,
        tokenizer,
        dataset,
        token_frequencies,
        top_k_percent=0.05,
        rare_threshold=None,
        base_scaling_factor=1.5,
        scaling_method="linear",
        variance_threshold=0.95,
        exponent=1.0,
    )
    results = freq_scaler.run_pipeline()
    # save the results to the target file
    results.to_csv(settings.PATH.result_dir / args.out_path)


if __name__ == "__main__":
    main()
