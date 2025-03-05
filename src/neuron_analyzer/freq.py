import typing as t

import numpy as np
import torch
from transformers import AutoTokenizer


class UnigramAnalyzer:
    """Class for analyzing unigram frequencies of words based on model-specific unigram counts."""
    # Model-specific constants
    MODEL_CONFIGS = {
        "pythia": {"W_U_SIZE": 50304, "TRUE_VOCAB_SIZE": 50277},
        "phi-2": {"W_U_SIZE": 51200, "TRUE_VOCAB_SIZE": 50295}
    }

    def __init__(
        self,
        model_name: str = "pythia-410m",
        unigram_file_path: t.Optional[str] = None,
        tokenizer: t.Optional[AutoTokenizer] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the UnigramAnalyzer with model and tokenizer information."""
        self.model_name = model_name
        self.device = device

        # Determine model type and set vocabulary sizes
        if "pythia" in model_name:
            self.model_type = "pythia"
        elif "phi-2" in model_name:
            self.model_type = "phi-2"
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.W_U_SIZE = self.MODEL_CONFIGS[self.model_type]["W_U_SIZE"]
        self.TRUE_VOCAB_SIZE = self.MODEL_CONFIGS[self.model_type]["TRUE_VOCAB_SIZE"]
        self.token_discrepancy = self.W_U_SIZE - self.TRUE_VOCAB_SIZE
        self.unigram_file_path = unigram_file_path
        self.tokenizer = tokenizer
        # Load and prepare unigram data
        self._load_unigram_data()

    def _load_unigram_data(self) -> None:
        """Load unigram data from file and prepare distributions."""
        # Load the unigram counts from the .npy file
        self.unigram_count = np.load(self.unigram_file_path)

        # Pad the unigram count array if needed
        if len(self.unigram_count) < self.W_U_SIZE:
            self.unigram_count = np.concatenate([self.unigram_count, [0] * self.token_discrepancy])

        # Calculate unigram distribution (normalized frequency)
        self.unigram_distrib = self.unigram_count + 1  # Add 1 for Laplace smoothing
        self.unigram_distrib = self.unigram_distrib / self.unigram_distrib.sum()
        self.unigram_distrib = torch.tensor(self.unigram_distrib, dtype=torch.float32).to(self.device)

    def get_unigram_freq(self, word: str) -> list[tuple[int, int, float]]:
        """Get the unigram count and frequency for a given word."""
        # Encode the word to get token IDs
        token_ids = self.tokenizer.encode(word, add_special_tokens=False)
        # Get counts and frequencies for each token in the word
        results = []
        for token_id in token_ids:
            # Ensure token_id is within bounds
            if 0 <= token_id < len(self.unigram_count):
                count = int(self.unigram_count[token_id])
                frequency = float(self.unigram_distrib[token_id].cpu().item())
            else:
                count = 0
                frequency = 0.0

            results.append((token_id, count, frequency))

        return results

    def get_word_unigram_stats(self, word: str) -> dict:
        """Get comprehensive unigram statistics for a word."""
        token_results = self.get_unigram_freq(word)
        # Calculate aggregated stats
        total_count = sum(count for _, count, _ in token_results) / len(token_results)
        avg_frequency = sum(freq for _, _, freq in token_results) / len(token_results) if token_results else 0.0

        return {
            "total_tokens": len(token_results),
            "total_count": total_count,
            "avg_frequency": avg_frequency
        }

