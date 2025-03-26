import argparse
import sys

sys.path.append("../")
import logging
import typing as t
from warnings import simplefilter

import pandas as pd
import torch
from transformers import AutoTokenizer

from neuron_analyzer import settings
from neuron_analyzer.abl_util import get_pile_unigram_distribution
from neuron_analyzer.freq import ZipfThresholdAnalyzer
from neuron_analyzer.preprocess import filter_words

T = t.TypeVar("T")

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for step range."""
    parser = argparse.ArgumentParser(description="Extract word surprisal across different training steps.")
    parser.add_argument("--model", default="EleutherAI/pythia-410m", help="train data of the model")
    return parser.parse_args()


class TokenSelector:
    """Class to handle token selection for long-tail analysis."""

    def __init__(self, model: str, logger: t.Optional[logging.Logger] = None):
        """Initialize the token selector."""
        # Set up logger
        self.logger = logger or logging.getLogger(__name__)
        # Initialize parameters
        self.model = model
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)

        # Initialize these attributes explicitly
        self.unigram_distrib = None
        self.longtail_threshold = None
        self.longtail_indices = None

        # Handle model-specific vocabulary details
        if "pythia" in self.model:
            self.true_vocab_size = 50277
            self.pad_to_match_w_u = True
        elif "phi-2" in self.model:
            self.true_vocab_size = 50295
            self.pad_to_match_w_u = True
        else:  # Default for other models like GPT-2
            self.true_vocab_size = len(self.tokenizer)
            self.pad_to_match_w_u = False

    def load_unigram(self) -> torch.Tensor:
        """Load unigram distribution based on model type."""
        if "pythia" in self.model:
            self.logger.info("Loading unigram distribution for pythia...")
            unigram_distrib = get_pile_unigram_distribution(
                device=self.device, file_path=settings.PATH.unigram_dir / "pythia-unigrams.npy"
            )
        elif "gpt" in self.model:
            self.logger.info("Loading unigram distribution for gpt2...")
            unigram_distrib = get_pile_unigram_distribution(
                device=self.device,
                file_path=settings.PATH.unigram_dir / "gpt2-small-unigrams_openwebtext-2M_rows_500000.npy",
                pad_to_match_W_U=False,
            )
        else:
            raise Exception(f"No unigram distribution for {self.model}")

        self.unigram_distrib = unigram_distrib
        return unigram_distrib

    def get_tail_threshold(self, unigram_distrib: torch.Tensor) -> float:
        """Calculate threshold for long-tail ablation mode."""
        window_size = 2000
        analyzer = ZipfThresholdAnalyzer(unigram_distrib, window_size=window_size)
        threshold_stats = analyzer.analyze_zipf_anomalies(verbose=False)
        longtail_threshold = threshold_stats["elbow_info"]["elbow_probability"]

        self.longtail_threshold = longtail_threshold
        self.unigram_distrib = unigram_distrib  # Ensure this is set

        # Now identify long-tail tokens
        self._identify_longtail_indices()

        self.logger.info(f"Calculating long-tail threshold using Zipf's law with window size {window_size}.")
        return longtail_threshold

    def _identify_longtail_indices(self) -> list[int]:
        """Identify token indices in the long tail based on threshold."""
        if self.unigram_distrib is None or self.longtail_threshold is None:
            raise ValueError("Unigram distribution and threshold must be calculated first")

        # Only consider indices within true vocabulary size when using padded distributions
        if self.pad_to_match_w_u:
            mask = torch.zeros_like(self.unigram_distrib, dtype=torch.bool)
            mask[: self.true_vocab_size] = True
            mask = mask & (self.unigram_distrib < self.longtail_threshold)
            self.longtail_indices = torch.where(mask)[0].cpu().tolist()
        else:
            # For non-padded distributions, consider all indices
            self.longtail_indices = torch.where(self.unigram_distrib < self.longtail_threshold)[0].cpu().tolist()

        self.logger.info(
            f"Identified {len(self.longtail_indices)} long-tail tokens out of {self.true_vocab_size} true vocab tokens"
        )
        return self.longtail_indices

    def decode_longtail_tokens(self) -> list[str]:
        """Decode only the long-tail tokens from the tokenizer."""
        if self.longtail_indices is None:
            raise ValueError("Long-tail indices not calculated. Call get_tail_threshold first.")

        # Use batch processing to handle large vocabularies
        batch_size = 1000
        decoded_tokens = []

        # Process in batches to avoid potential memory issues
        for i in range(0, len(self.longtail_indices), batch_size):
            batch = self.longtail_indices[i : i + batch_size]
            try:
                decoded_batch = self.tokenizer.batch_decode(batch, skip_special_tokens=True)
                decoded_tokens.extend(decoded_batch)
            except Exception as e:
                self.logger.error(f"Error decoding batch starting at index {i}: {e}")
                # Continue with next batch instead of failing completely

        return decoded_tokens

    def get_longtail_words(self) -> list[str]:
        """Encapsulated method to perform the entire process."""
        # Step 1: Load unigram distribution
        self.logger.info("Loading unigram distribution...")
        unigram_distrib = self.load_unigram()

        # Step 2: Calculate threshold and identify long-tail tokens
        self.logger.info("Calculating threshold and identifying long-tail tokens...")
        self.get_tail_threshold(unigram_distrib)

        # Step 3: Decode and return the long-tail tokens
        self.logger.info("Decoding long-tail tokens...")
        words = self.decode_longtail_tokens()

        # Step 4: Filter correct words
        self.logger.info("Filtering true words...")
        words = filter_words(words)
        word_df = pd.DataFrame(words)
        word_df.columns = ["word"]
        word_df["word"] = word_df["word"].str.lower()
        self.logger.info(f"Found {len(words)} long-tail words")
        return word_df


def main():
    """Main entry point that handles both CLI args and Hydra config."""
    args = parse_args()

    # intialize the process class
    token_selector = TokenSelector(model=args.model, logger=logger)
    word_df = token_selector.get_longtail_words()
    out_path = settings.PATH.dataset_root / "freq" / args.model / "longtail_words.csv"
    word_df.to_csv(out_path)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
