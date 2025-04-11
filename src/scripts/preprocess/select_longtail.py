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
from neuron_analyzer.analysis.freq import ZipfThresholdAnalyzer
from neuron_analyzer.load_util import load_unigram
from neuron_analyzer.preprocess.preprocess import filter_words

T = t.TypeVar("T")

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for step range."""
    parser = argparse.ArgumentParser(description="Extract longtail words from longtail tokens.")
    parser.add_argument("--model", default="EleutherAI/pythia-410m", help="train data of the model")
    parser.add_argument("--tail_threshold", type=float, default=50, help="prop of longtail")
    parser.add_argument("--apply_elbow", type=bool, default=False, help="whether to check the elbow")
    return parser.parse_args()


class TokenSelector:
    """Class to handle token selection for long-tail analysis."""

    def __init__(
        self,
        model: str,
        device: str,
        apply_elbow: bool,
        tail_threshold: float = 0.5,
    ):
        """Initialize the token selector."""
        # Initialize parameters
        self.model = model
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.tail_threshold = tail_threshold
        self.apply_elbow = apply_elbow

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

    def decode_longtail_tokens(self, longtail_threshold) -> list[str]:
        """Decode only the long-tail tokens from the tokenizer."""
        # get longtail indices
        self._identify_longtail_indices(longtail_threshold)
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
                logger.error(f"Error decoding batch starting at index {i}: {e}")

        return decoded_tokens

    def _identify_longtail_indices(self, longtail_threshold) -> list[int]:
        """Identify token indices in the long tail based on threshold."""
        # Only consider indices within true vocabulary size when using padded distributions
        if self.pad_to_match_w_u:
            mask = torch.zeros_like(self.unigram_distrib, dtype=torch.bool)
            mask[: self.true_vocab_size] = True
            mask = mask & (self.unigram_distrib < self.tail_threshold)
            self.longtail_indices = torch.where(mask)[0].cpu().tolist()
        else:
            # For non-padded distributions, consider all indices
            self.longtail_indices = torch.where(self.unigram_distrib < longtail_threshold)[0].cpu().tolist()

        logger.info(
            f"Identified {len(self.longtail_indices)} long-tail tokens out of {self.true_vocab_size} true vocab tokens"
        )
        return self.longtail_indices

    def get_longtail_words(self) -> list[str]:
        """Encapsulated method to perform the entire process."""
        # Step 1: Load unigram distribution
        logger.info("Loading unigram distribution...")
        self.unigram_distrib, _ = load_unigram(self.model, self.device)

        # Step 2: Calculate threshold and identify long-tail tokens
        logger.info("Calculating threshold and identifying long-tail tokens...")
        analyzer = ZipfThresholdAnalyzer(
            unigram_distrib=self.unigram_distrib,
            tail_threshold=self.tail_threshold,
            apply_elbow=self.apply_elbow,
        )
        longtail_threshold, _ = analyzer.get_tail_threshold()

        # Step 3: Decode and return the long-tail tokens
        logger.info("Decoding long-tail tokens...")
        words = self.decode_longtail_tokens(longtail_threshold)

        # Step 4: Filter correct words
        logger.info("Filtering true words...")
        words = filter_words(words)
        word_df = pd.DataFrame(words)
        word_df.columns = ["word"]
        # bpe tokens are case sensitive
        # word_df["word"] = word_df["word"].str.lower()
        logger.info(f"Found {len(words)} long-tail words")
        return word_df

    def get_save_dir(self, word_df) -> None:
        """Get the savepath based on current configurations."""
        ablation_name = "longtail_elbow.csv" if self.apply_elbow else f"longtail_{self.tail_threshold}.csv"
        out_path = settings.PATH.dataset_root / "freq" / self.model / ablation_name
        word_df.to_csv(out_path)
        logger.info(f"Results saved to {out_path}")


def main():
    """Main entry point that handles both CLI args and Hydra config."""
    args = parse_args()
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # intialize the process class
    token_selector = TokenSelector(
        model=args.model,
        device=device,
        apply_elbow=args.apply_elbow,
        tail_threshold=args.tail_threshold,
    )
    word_df = token_selector.get_longtail_words()
    token_selector.get_save_dir(word_df)


if __name__ == "__main__":
    main()
