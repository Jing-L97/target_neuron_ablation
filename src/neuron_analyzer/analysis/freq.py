import logging
import os
import sys
import typing as t

import numpy as np
import torch
from scipy import stats
from transformers import AutoTokenizer

from neuron_analyzer.load_util import extract_tail_threshold, load_unigram
from neuron_analyzer.settings import get_dtype

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Only show errors, not warnings
# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


#######################################################
# Analyze freq distribution and select zipf threshold


class ZipfThresholdAnalyzer:
    def __init__(
        self,
        unigram_distrib,
        window_size: int = 2000,
        head_portion: float = 0.2,
        min_freq: t.Any = None,  # percentage or "elbow"
        max_freq: t.Any = None,  # percentage or "elbow"
        residual_significance_threshold: float = 1.5,
        min_tokens_threshold: int = 10,
    ):
        """Initialize Zipf threshold analyzer with configurable parameters."""

        def parse_value(val):
            if isinstance(val, str):
                try:
                    return float(val)
                except ValueError:
                    return val
            return val

        self.unigram_distrib = unigram_distrib
        self.head_portion = head_portion
        self.min_freq_input = parse_value(min_freq)
        self.max_freq_input = parse_value(max_freq)
        self.residual_significance_threshold = residual_significance_threshold
        self.min_tokens_threshold = min_tokens_threshold
        self.window_size = window_size
        self._sorted_probs = None
        self._ranks = None
        self._sorted_indices = None

    def _preprocess_distribution(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess the input distribution."""
        if self._sorted_probs is not None:
            return self._sorted_probs, self._ranks, self._sorted_indices

        if torch.is_tensor(self.unigram_distrib):
            unigram_probs = self.unigram_distrib.cpu().numpy()
        else:
            unigram_probs = np.asarray(self.unigram_distrib)

        unigram_probs = unigram_probs / unigram_probs.sum()
        sorted_indices = np.argsort(-unigram_probs)
        sorted_probs = unigram_probs[sorted_indices]
        ranks = np.arange(1, len(sorted_probs) + 1)

        self._sorted_probs = sorted_probs
        self._ranks = ranks
        self._sorted_indices = sorted_indices

        return sorted_probs, ranks, sorted_indices

    def find_elbow_point(
        self,
        log_ranks: np.ndarray,
        log_probs: np.ndarray,
        threshold_multiplier: float = 3.0,
        min_rank_percentile: float = 0.5,
    ) -> dict[str, t.Any]:
        """Find the elbow point in the log-log Zipf distribution."""
        if len(log_ranks) < self.window_size * 3:
            return {"elbow_detected": False, "message": "Not enough data points for elbow detection"}

        derivatives = []
        indices = []
        min_rank_idx = max(int(len(log_ranks) * min_rank_percentile), self.window_size * 2)

        for i in range(min_rank_idx, len(log_ranks) - self.window_size):
            window_fit = stats.linregress(log_ranks[i : i + self.window_size], log_probs[i : i + self.window_size])
            derivatives.append(window_fit.slope)
            indices.append(i)

        derivatives = np.array(derivatives)
        slope_changes = np.abs(np.diff(derivatives))
        mean_change = np.mean(slope_changes)
        std_change = np.std(slope_changes)
        threshold = mean_change + threshold_multiplier * std_change
        significant_changes = np.where(slope_changes > threshold)[0]

        if len(significant_changes) == 0:
            return {"elbow_detected": False, "message": "No significant elbow point detected"}

        max_change_idx = significant_changes[np.argmax(slope_changes[significant_changes])]
        elbow_idx = indices[max_change_idx + 1]

        elbow_rank = np.exp(log_ranks[elbow_idx])
        elbow_prob = np.exp(log_probs[elbow_idx])

        return {
            "elbow_detected": True,
            "rank": int(elbow_rank),
            "probability": elbow_prob,
            "log_rank": log_ranks[elbow_idx],
            "log_probability": log_probs[elbow_idx],
            "index": elbow_idx,
            "slope_before": derivatives[max_change_idx],
            "slope_after": derivatives[max_change_idx + 1],
        }

    def find_prop_point(self, prop: float) -> dict[str, t.Any]:
        """Select tokens based on a proportion (0-1) and return frequency info."""
        sorted_probs, ranks, sorted_indices = self._preprocess_distribution()
        total_tokens = len(sorted_probs)
        idx = int(total_tokens * prop)
        idx = min(max(idx, 0), total_tokens - 1)
        freq = sorted_probs[idx]
        return {"frequency": freq, "rank": ranks[idx], "index": sorted_indices[idx]}

    def analyze_zipf_anomalies(self, verbose: bool = False) -> dict[str, t.Any]:
        """Analyze Zipf distribution and handle both proportion and elbow inputs."""
        sorted_probs, ranks, sorted_indices = self._preprocess_distribution()

        if len(sorted_probs) < self.min_tokens_threshold:
            raise ValueError(
                f"Insufficient tokens. Need at least {self.min_tokens_threshold}, found {len(sorted_probs)}"
            )

        head_cutoff = max(2, int(len(sorted_probs) * self.head_portion))
        log_ranks = np.log(ranks)
        log_probs = np.log(sorted_probs)
        head_fit = stats.linregress(log_ranks[:head_cutoff], log_probs[:head_cutoff])

        results = {"zipf_alpha": -head_fit.slope, "zipf_fit_r_value": head_fit.rvalue}
        threshold_info = {}

        # Compute elbow once for integration
        elbow_info = self.find_elbow_point(log_ranks, log_probs)

        def build_threshold_data(freq_input):
            """Build a threshold dict containing both proportion and elbow info if applicable."""
            data = {}
            if freq_input == "elbow":
                data["elbow"] = elbow_info
            elif freq_input is not None:
                prop = float(freq_input) / 100.0
                data["proportion"] = self.find_prop_point(prop)
            else:
                logger.info("Wrong freq info! Existing")
                sys.exit()
            return data

        min_data = build_threshold_data(self.min_freq_input)
        max_data = build_threshold_data(self.max_freq_input)

        # Ensure min <= max based on main frequency value (proportion if exists, else elbow)
        def get_main_value(d):
            if "proportion" in d:
                return d["proportion"]["frequency"]
            if "elbow" in d:
                return d["elbow"]["probability"]
            return None

        min_val = get_main_value(min_data)
        max_val = get_main_value(max_data)
        if min_val is not None and max_val is not None and min_val > max_val:
            logger.info(f"Reversing max {max_data} with min {min_data}")
            min_data, max_data = max_data, min_data

        threshold_info["min"] = min_data
        threshold_info["max"] = max_data
        results["threshold_info"] = threshold_info

        return results

    def get_tail_threshold(self) -> tuple[float | None, float | None, dict | None]:
        stats = self.analyze_zipf_anomalies(verbose=False)
        min_freq, max_freq = extract_tail_threshold(stats)
        return min_freq, max_freq, stats


#######################################################
# Analyze unigram distribution


class UnigramAnalyzer:
    """Class for analyzing unigram frequencies of words based on model-specific unigram counts."""

    def __init__(self, device: str, unigram_distrib=None, unigram_count=None, model_name: str = "pythia-410m"):
        """Initialize the UnigramAnalyzer with model and tokenizer information."""
        self.model_name = model_name
        self.device = device
        if unigram_distrib is None:
            self.unigram_distrib, self.unigram_count = load_unigram(
                self.model_name, self.device, dtype=get_dtype(model_name)
            )
        else:
            self.unigram_distrib, self.unigram_count = unigram_distrib, unigram_count

    def get_unigram_freq(self, word: str) -> list[tuple[int, int, float]]:
        """Get the unigram count and frequency for a given word."""
        # encode word
        token_ids = self._encode_word(word)
        # Get counts and frequencies for each token in the word
        results = []
        for token_id in token_ids:
            count, frequency = self._extract_freq(token_id)
            results.append((token_id, count, frequency))
        return results

    def get_word_unigram_stats(self, word: str) -> dict:
        """Get comprehensive unigram statistics for a word."""
        token_results = self.get_unigram_freq(word)
        # Calculate aggregated stats
        total_count = sum(count for _, count, _ in token_results) / len(token_results)
        avg_frequency = sum(freq for _, _, freq in token_results) / len(token_results) if token_results else 0.0

        return {"total_tokens": len(token_results), "total_count": total_count, "avg_frequency": avg_frequency}

    def _encode_word(self, word: str) -> int | list[int]:
        """Encode the word to get token IDs."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self.tokenizer.encode(word, add_special_tokens=False)

    def _extract_freq(self, token_id: int) -> tuple[int, float]:
        """Extract frequency from token id."""
        # Ensure token_id is within bounds
        if 0 <= token_id < len(self.unigram_count):
            return int(self.unigram_count[token_id]), float(self.unigram_distrib[token_id].cpu().item())
        return 0, 0.0
