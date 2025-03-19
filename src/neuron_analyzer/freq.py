import typing as t

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from transformers import AutoTokenizer


class UnigramAnalyzer:
    """Class for analyzing unigram frequencies of words based on model-specific unigram counts."""

    # Model-specific constants
    MODEL_CONFIGS = {
        "pythia": {"W_U_SIZE": 50304, "TRUE_VOCAB_SIZE": 50277},
        "phi-2": {"W_U_SIZE": 51200, "TRUE_VOCAB_SIZE": 50295},
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

        return {"total_tokens": len(token_results), "total_count": total_count, "avg_frequency": avg_frequency}


def calculate_zipf_threshold(unigram_distrib, head_portion=0.3, error_threshold=0.5):
    """Calculate long-tail threshold based on deviation from Zipf's law."""

    # Convert to numpy if tensor
    if torch.is_tensor(unigram_distrib):
        unigram_probs = unigram_distrib.cpu().numpy()
    else:
        unigram_probs = unigram_distrib

    # Make sure it's normalized
    if abs(unigram_probs.sum() - 1.0) > 1e-5:
        unigram_probs = unigram_probs / unigram_probs.sum()

    # Sort by frequency (descending)
    sorted_indices = np.argsort(-unigram_probs)
    sorted_probs = unigram_probs[sorted_indices]
    ranks = np.arange(1, len(sorted_probs) + 1)

    # Filter out zeros for log calculation
    nonzero_mask = sorted_probs > 0
    nonzero_ranks = ranks[nonzero_mask]
    nonzero_probs = sorted_probs[nonzero_mask]

    log_ranks = np.log(nonzero_ranks)
    log_probs = np.log(nonzero_probs)

    # Fit Zipf's law to the head of the distribution
    head_cutoff = int(len(nonzero_ranks) * head_portion)
    fit = stats.linregress(log_ranks[:head_cutoff], log_probs[:head_cutoff])
    alpha = -fit.slope

    # Predict token frequencies based on the fit
    predicted_log_probs = fit.slope * log_ranks + fit.intercept
    predicted_probs = np.exp(predicted_log_probs)

    # Calculate relative error between actual and predicted
    relative_error = np.abs(nonzero_probs - predicted_probs) / predicted_probs

    # Find where error exceeds the threshold
    deviation_indices = np.where(relative_error > error_threshold)[0]

    if len(deviation_indices) > 0:
        first_major_deviation = deviation_indices[0]
        threshold = nonzero_probs[first_major_deviation]
        deviation_rank = nonzero_ranks[first_major_deviation]
    else:
        # Fallback if no major deviation found
        threshold = np.percentile(nonzero_probs, 70)
        deviation_rank = None

    # Calculate percentage of tokens considered long-tail
    long_tail_count = (unigram_probs < threshold).sum()
    long_tail_percentage = long_tail_count / len(unigram_probs) * 100

    stats_dict = {
        "threshold": threshold,
        "alpha": alpha,
        "zipf_fit_slope": fit.slope,
        "zipf_fit_intercept": fit.intercept,
        "zipf_fit_r_value": fit.rvalue,
        "long_tail_token_count": long_tail_count,
        "long_tail_percentage": long_tail_percentage,
        "deviation_rank": deviation_rank,
    }

    return threshold, stats_dict


class ZipfThresholdAnalyzer:
    def __init__(
        self,
        unigram_distrib,
        window_size: int = 2000, # sliding window size for elbow point detection
        head_portion: float = 0.2,  # Modified to 20% for tail analysis
        tail_threshold: float = 0.8,  # New parameter for tail region
        residual_significance_threshold: float = 1.5,  # Threshold for identifying anomalous words
        min_tokens_threshold: int = 10,  # Increased for more robust analysis
    ):
        """Initialize Zipf threshold analyzer with configurable parameters."""
        self.unigram_distrib = unigram_distrib
        self.head_portion = head_portion
        self.tail_threshold = tail_threshold
        self.residual_significance_threshold = residual_significance_threshold
        self.min_tokens_threshold = min_tokens_threshold
        self.window_size = window_size

    def _preprocess_distribution(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess the input distribution for Zipf law analysis."""
        # Normalize and convert to numpy
        if torch.is_tensor(self.unigram_distrib):
            unigram_probs = self.unigram_distrib.cpu().numpy()
        else:
            unigram_probs = np.asarray(self.unigram_distrib)

        # Ensure probability distribution sums to 1
        unigram_probs = unigram_probs / unigram_probs.sum()

        # Sort probabilities in descending order
        sorted_indices = np.argsort(-unigram_probs)
        sorted_probs = unigram_probs[sorted_indices]
        ranks = np.arange(1, len(sorted_probs) + 1)

        return sorted_probs, ranks, sorted_indices

    def find_elbow_point(
        self,
        log_ranks: np.ndarray,
        log_probs: np.ndarray,
        threshold_multiplier: float = 3.0,
        min_rank_percentile: float = 0.5,
    ) -> dict[str, t.Any]:
        # We need enough points for a meaningful analysis
        if len(log_ranks) < self.window_size * 3:
            return {"elbow_detected": False, "message": "Not enough data points for elbow detection"}

        # Calculate first derivatives (slopes) using sliding windows
        derivatives = []
        indices = []
        # Skip the very head of the distribution as it might have its own variations
        min_rank_idx = max(int(len(log_ranks) * min_rank_percentile), self.window_size * 2)

        for i in range(min_rank_idx, len(log_ranks) - self.window_size):
            # Calculate slope in the current window
            window_fit = stats.linregress(log_ranks[i : i + self.window_size], log_probs[i : i + self.window_size])
            derivatives.append(window_fit.slope)
            indices.append(i)

        derivatives = np.array(derivatives)

        # Compute differences between adjacent slopes to find rapid changes
        slope_changes = np.abs(np.diff(derivatives))

        # Detect significant changes (beyond several standard deviations)
        mean_change = np.mean(slope_changes)
        std_change = np.std(slope_changes)
        threshold = mean_change + threshold_multiplier * std_change

        # Find points exceeding the threshold
        significant_changes = np.where(slope_changes > threshold)[0]

        if len(significant_changes) == 0:
            return {"elbow_detected": False, "message": "No significant elbow point detected"}

        # Find the most significant change
        max_change_idx = significant_changes[np.argmax(slope_changes[significant_changes])]
        elbow_idx = indices[max_change_idx + 1]  # +1 because of np.diff

        # Get actual rank and probability at the elbow point
        elbow_rank = np.exp(log_ranks[elbow_idx])
        elbow_prob = np.exp(log_probs[elbow_idx])

        return {
            "elbow_detected": True,
            "elbow_rank": int(elbow_rank),
            "elbow_probability": elbow_prob,
            "elbow_log_rank": log_ranks[elbow_idx],
            "elbow_log_probability": log_probs[elbow_idx],
            "elbow_index": elbow_idx,
            "slope_before": derivatives[max_change_idx],
            "slope_after": derivatives[max_change_idx + 1],
        }

    def analyze_zipf_anomalies(
        self, detect_elbow: bool = True, verbose: bool = False
    ) -> dict[str, t.Any]:
        """Analyze Zipf law distribution and identify anomalous words."""
        # Preprocess distribution
        sorted_probs, ranks, sorted_indices = self._preprocess_distribution()
        # Validate sufficient data
        if len(sorted_probs) < self.min_tokens_threshold:
            raise ValueError(
                f"Insufficient tokens. Need at least {self.min_tokens_threshold}, found {len(sorted_probs)}"
            )
        # Fit Zipf's law to the head of the distribution
        head_cutoff = max(2, int(len(sorted_probs) * self.head_portion))
        tail_start = max(2, int(len(sorted_probs) * self.tail_threshold))
        # Take log of ranks and probabilities
        log_ranks = np.log(ranks)
        log_probs = np.log(sorted_probs)
        # Fit Zipf law to the head (power law)
        head_fit = stats.linregress(log_ranks[:head_cutoff], log_probs[:head_cutoff])

        # Calculate residuals for the tail region
        tail_residuals = log_probs[tail_start:] - (head_fit.slope * log_ranks[tail_start:] + head_fit.intercept)

        # Compute residual statistics
        residual_mean = np.mean(tail_residuals)
        residual_std = np.std(tail_residuals)

        # Prepare results
        results = {
            "zipf_alpha": -head_fit.slope,  # Power law exponent
            "zipf_fit_r_value": head_fit.rvalue,
            "tail_residual_mean": residual_mean,
            "tail_residual_std": residual_std,
            "tail_start_rank": tail_start,
        }

        # Detect elbow point if requested
        elbow_info = None
        if detect_elbow:
            elbow_info = self.find_elbow_point(log_ranks, log_probs)
            results["elbow_info"] = elbow_info

        # Optional visualization
        if verbose:
            self._visualize_zipf_analysis(
                log_ranks, log_probs, head_fit, head_cutoff, tail_start, tail_residuals, elbow_info
            )

        return results

    def _visualize_zipf_analysis(
        self,
        log_ranks: np.ndarray,
        log_probs: np.ndarray,
        head_fit: t.Any,
        head_cutoff: int,
        tail_start: int,
        tail_residuals: np.ndarray,
        elbow_info: dict[str, t.Any] = None,
    ):
        plt.figure(figsize=(15, 5))

        # Log-Log Distribution Plot
        plt.subplot(1, 3, 1)
        plt.scatter(log_ranks, log_probs, label="Actual", alpha=0.7)
        plt.plot(
            log_ranks[:head_cutoff],
            head_fit.slope * log_ranks[:head_cutoff] + head_fit.intercept,
            color="red",
            label="Zipf Fit (Head)",
        )

        # Mark the elbow point if detected
        if elbow_info and elbow_info.get("elbow_detected", False):
            elbow_idx = elbow_info["elbow_index"]
            plt.scatter(
                log_ranks[elbow_idx],
                log_probs[elbow_idx],
                color="green",
                s=100,
                marker="X",
                label=f"Elbow Point (Rank {int(np.exp(log_ranks[elbow_idx]))})",
            )

        plt.title("Log-Log Distribution")
        plt.xlabel("Log Rank")
        plt.ylabel("Log Probability")
        plt.legend()

        # Tail Residuals Plot
        plt.subplot(1, 3, 2)
        plt.scatter(log_ranks[tail_start:], tail_residuals, label="Tail Residuals", alpha=0.7)

        # Mark the elbow point on the residuals plot if in range
        if elbow_info and elbow_info.get("elbow_detected", False):
            elbow_idx = elbow_info["elbow_index"]
            if elbow_idx >= tail_start:
                elbow_residual_idx = elbow_idx - tail_start
                plt.scatter(
                    log_ranks[elbow_idx],
                    tail_residuals[elbow_residual_idx],
                    color="green",
                    s=100,
                    marker="X",
                    label="Elbow Point",
                )

        plt.title("Tail Residuals")
        plt.xlabel("Log Rank")
        plt.ylabel("Residual")
        plt.legend()
