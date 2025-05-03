import json
import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import scipy
import torch

from neuron_analyzer.load_util import cleanup

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HeavyTailedAnalysisCore:
    """Core analysis methods for Heavy-Tailed Self-Regularization theory.
    This class contains the shared analytical methods that can be applied
    to both weight space and activation space.
    """

    @staticmethod
    def eigenspectrum_analysis(correlation_matrix: np.ndarray) -> dict[str, Any]:
        """Compute eigenspectrum of the provided correlation matrix."""
        if correlation_matrix.shape[0] < 2:
            return {"eigenvalues": None, "error": "Matrix too small for eigenspectrum analysis"}

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(correlation_matrix)

        # Sort in descending order
        eigenvalues = np.sort(eigenvalues)[::-1]

        return {
            "eigenvalues": eigenvalues.tolist(),
            "max_eigenvalue": float(eigenvalues[0]),
            "min_eigenvalue": float(eigenvalues[-1]),
            "sum_eigenvalues": float(np.sum(eigenvalues)),
            "num_eigenvalues": int(len(eigenvalues)),
        }

    @staticmethod
    def esd_shape_analysis(correlation_matrix: np.ndarray, Q: float | None = None) -> dict[str, Any]:
        """Analyze the shape of the Empirical Spectral Density (ESD).
        Measures deviation from Marchenko-Pastur law and identifies spike separation.

        Args:
            correlation_matrix: Correlation matrix to analyze
            Q: Shape parameter for MP law. If None, estimated from data.

        Returns:
            Dictionary with shape analysis metrics

        """
        result = {}

        # Get eigenvalues
        eig_result = HeavyTailedAnalysisCore.eigenspectrum_analysis(correlation_matrix)
        if eig_result["eigenvalues"] is None:
            return {"error": "Matrix too small for ESD shape analysis"}

        eigenvalues = np.array(eig_result["eigenvalues"])
        n = correlation_matrix.shape[0]  # Number of rows
        d = correlation_matrix.shape[1] if correlation_matrix.shape[0] != correlation_matrix.shape[1] else n

        # Estimate Q if not provided (with safety check)
        if Q is None:
            Q = n / max(d, 1)  # Avoid division by zero

        # Compute Marchenko-Pastur (MP) law parameters with numerical stability
        sigma_squared = max(np.mean(eigenvalues), 1e-10)  # Ensure minimal value

        # Safe calculation of lambda bounds
        safe_Q = max(Q, 1e-10)  # Ensure Q is not too small
        lambda_plus = sigma_squared * (1 + np.sqrt(1 / safe_Q)) ** 2
        lambda_minus = sigma_squared * (1 - np.sqrt(1 / safe_Q)) ** 2 if Q > 1 else 1e-10

        # Ensure minimum value for lambda_minus
        lambda_minus = max(lambda_minus, 1e-10)

        # Generate MP law distribution (discretized) with numerical safety
        # Ensure x range is positive and well-behaved
        x = np.linspace(lambda_minus, lambda_plus * 1.01, 1000)  # Add small margin for numerical stability
        mp_pdf = np.zeros_like(x)

        # Define valid x range with safety checks
        valid_x = (x >= lambda_minus) & (x <= lambda_plus)

        # Safely compute MP PDF
        try:
            safe_denom = 2 * np.pi * sigma_squared
            if safe_denom > 0:
                scale_factor = Q / safe_denom

                # Compute sqrt term safely
                sqrt_term = np.sqrt(np.maximum(0, (lambda_plus - x[valid_x]) * (x[valid_x] - lambda_minus)))

                # Compute final PDF with safe division
                mp_pdf[valid_x] = scale_factor * sqrt_term / np.maximum(x[valid_x], 1e-10)
        except Exception as e:
            logger.warning(f"Error in MP PDF calculation: {e}")
            # Fallback to uniform distribution if calculation fails
            mp_pdf[valid_x] = 1.0 / np.sum(valid_x) if np.sum(valid_x) > 0 else 0

        # Normalize MP PDF safely
        mp_pdf_sum = np.sum(mp_pdf)
        if mp_pdf_sum > 0:
            mp_pdf_norm = mp_pdf / mp_pdf_sum
        else:
            # Fallback to uniform distribution
            mp_pdf_norm = np.zeros_like(mp_pdf)
            mp_pdf_norm[valid_x] = 1.0 / np.sum(valid_x) if np.sum(valid_x) > 0 else 0

        # Generate empirical CDF of eigenvalues
        eigenvalues_sorted = np.sort(eigenvalues)
        empirical_cdf = np.arange(1, len(eigenvalues) + 1) / len(eigenvalues)

        # Generate theoretical CDF from MP law
        mp_cdf = np.cumsum(mp_pdf_norm)

        # Compute KS statistic (approximation) with error handling
        def mp_cdf_interpolated(x):
            try:
                # Find nearest index in x array
                idx = np.abs(np.subtract.outer(x, np.linspace(lambda_minus, lambda_plus, 1000))).argmin(axis=1)
                return mp_cdf[idx]
            except Exception:
                # Fallback to zeros if interpolation fails
                return np.zeros_like(x)

        mp_cdf_at_eigenvalues = mp_cdf_interpolated(eigenvalues_sorted)
        ks_statistic = np.max(np.abs(empirical_cdf - mp_cdf_at_eigenvalues))

        # Spike separation metric (if we have at least 2 eigenvalues)
        spike_separation = None
        if len(eigenvalues) >= 3:
            lambda1, lambda2 = eigenvalues[0], eigenvalues[1]
            lambda_bulk_mean = np.mean(eigenvalues[2:])
            # Safe calculation of spike separation
            denominator = lambda2 - lambda_bulk_mean
            if abs(denominator) > 1e-10:
                spike_separation = (lambda1 - lambda2) / denominator
            else:
                # If denominator is too small, use a large value to indicate separation
                spike_separation = 1000.0 if lambda1 > lambda2 else 0.0

        # Store results
        result = {
            "ks_statistic": float(ks_statistic),
            "spike_separation": float(spike_separation) if spike_separation is not None else None,
            "lambda_plus": float(lambda_plus),
            "lambda_minus": float(lambda_minus),
            "eigenvalues_max": float(eigenvalues[0]),
            "eigenvalues_bulk_mean": float(np.mean(eigenvalues[2:])) if len(eigenvalues) >= 3 else None,
            "Q": float(Q),
            "sigma_squared": float(sigma_squared),
        }

        return result

    @staticmethod
    def shape_metrics_analysis(correlation_matrix: np.ndarray, k_csn: int | None = None) -> dict[str, Any]:
        """Compute shape metrics from HT-SR theory.

        Args:
            correlation_matrix: Correlation matrix to analyze
            k_csn: k value for Hill estimator (if None, estimated using heuristic)

        Returns:
            Dictionary with shape metrics: PL Alpha Hill, Alpha Hat, Stable Rank, Entropy

        """
        result = {}

        # Get eigenvalues
        eig_result = HeavyTailedAnalysisCore.eigenspectrum_analysis(correlation_matrix)
        if eig_result["eigenvalues"] is None:
            return {"error": "Matrix too small for shape metrics analysis"}

        eigenvalues = np.array(eig_result["eigenvalues"])

        # PL Alpha Hill calculation
        # To estimate k, we use a simple heuristic (10% of data points)
        if k_csn is None:
            k_csn = max(1, int(0.1 * len(eigenvalues)))
        k_csn = min(k_csn, len(eigenvalues) - 1)  # Ensure k is valid

        # Compute Hill estimator
        try:
            log_term = np.mean(np.log(eigenvalues[:k_csn] / eigenvalues[k_csn]))
            alpha_hill = 1 / log_term if log_term > 0 else None
        except:
            alpha_hill = None

        # Alpha Hat calculation
        try:
            lambda_min = eigenvalues[-1]
            log_term = np.sum(np.log(eigenvalues / lambda_min))
            alpha_hat = 1 + len(eigenvalues) / log_term if log_term > 0 else None
        except:
            alpha_hat = None

        # Stable Rank calculation
        stable_rank = np.sum(eigenvalues) / eigenvalues[0] if eigenvalues[0] > 0 else None

        # Entropy calculation
        p = eigenvalues / np.sum(eigenvalues)
        entropy = -np.sum(p * np.log(p + 1e-10))

        # Store results
        result = {
            "pl_alpha_hill": float(alpha_hill) if alpha_hill is not None else None,
            "alpha_hat": float(alpha_hat) if alpha_hat is not None else None,
            "stable_rank": float(stable_rank) if stable_rank is not None else None,
            "entropy": float(entropy),
            "k_csn_used": int(k_csn),
        }

        return result

    @staticmethod
    def phase_transition_analysis(correlation_matrix: np.ndarray, weights: list[float] | None = None) -> dict[str, Any]:
        """Analyze phase characteristics based on Martin's "five-plus-one phase model".

        Args:
            correlation_matrix: Correlation matrix to analyze
            weights: Weights for the composite metric [w1, w2, w3]

        Returns:
            Dictionary with phase transition metrics

        """
        # Default weights if not provided
        if weights is None:
            weights = [0.4, 0.3, 0.3]

        # Get ESD shape analysis and shape metrics
        esd_result = HeavyTailedAnalysisCore.esd_shape_analysis(correlation_matrix)
        shape_result = HeavyTailedAnalysisCore.shape_metrics_analysis(correlation_matrix)

        if "error" in esd_result or "error" in shape_result:
            return {"error": "Not enough data for phase transition analysis"}

        # Extract key metrics
        ks_statistic = esd_result["ks_statistic"]
        spike_separation = esd_result["spike_separation"] if esd_result["spike_separation"] is not None else 0
        alpha_hill = shape_result["pl_alpha_hill"] if shape_result["pl_alpha_hill"] is not None else 3.0

        # Reference value for Alpha Hill (typical for random matrices)
        alpha_ref = 3.0

        # Compute composite phase metric
        w1, w2, w3 = weights
        try:
            log_spike = np.log(spike_separation + 1e-10)
        except:
            log_spike = -10  # Default value if log fails

        phase_metric = w1 * ks_statistic + w2 * log_spike + w3 * (1 - alpha_hill / alpha_ref)

        # Determine phase based on composite metric
        # These thresholds are heuristic and may need adjustment
        phase = None
        if phase_metric < -2:
            phase = "random-like"
        elif phase_metric < -1:
            phase = "bulk+spike"
        elif phase_metric < 0:
            phase = "bulk-decay"
        elif phase_metric < 1:
            phase = "heavy-tailed"
        else:
            phase = "multi-power-law"

        # Store results
        result = {
            "phase_metric": float(phase_metric),
            "phase": phase,
            "ks_contribution": float(w1 * ks_statistic),
            "spike_contribution": float(w2 * log_spike),
            "alpha_contribution": float(w3 * (1 - alpha_hill / alpha_ref)),
            "weights": weights,
        }

        return result

    @staticmethod
    def _compute_moments(eigenvalues: np.ndarray, max_moment: int = 4) -> np.ndarray:
        """Compute moments of the eigenvalue distribution."""
        moments = np.zeros(max_moment + 1)
        eigenvalues = np.array(eigenvalues)
        total = np.sum(eigenvalues)

        for p in range(max_moment + 1):
            if p == 0:
                moments[p] = 1  # 0th moment is 1
            else:
                moments[p] = np.sum(eigenvalues**p) / total

        return moments

    @staticmethod
    def _moments_to_free_cumulants(moments: np.ndarray) -> np.ndarray:
        """Convert moments to free cumulants using the moment-cumulant formula."""
        n = len(moments) - 1
        cumulants = np.zeros(n + 1)
        cumulants[0] = 1  # 0th cumulant is 1
        cumulants[1] = moments[1]  # 1st cumulant equals 1st moment

        # Use recursive formula for higher cumulants
        for p in range(2, n + 1):
            cumulant_p = moments[p]
            for j in range(1, p):
                cumulant_p -= scipy.special.comb(p - 1, j - 1) * cumulants[j] * moments[p - j]
            cumulants[p] = cumulant_p

        return cumulants

    @staticmethod
    def bulk_spike_interaction_analysis(correlation_matrix: np.ndarray, max_moment: int = 4) -> dict[str, Any]:
        """Analyze interactions between spike and bulk components using free probability theory.

        Args:
            correlation_matrix: Correlation matrix to analyze
            max_moment: Maximum moment/cumulant order to compute

        Returns:
            Dictionary with bulk-spike interaction metrics

        """
        result = {}

        # Get eigenvalues
        eig_result = HeavyTailedAnalysisCore.eigenspectrum_analysis(correlation_matrix)
        if eig_result["eigenvalues"] is None:
            return {"error": "Matrix too small for bulk-spike interaction analysis"}

        eigenvalues = np.array(eig_result["eigenvalues"])

        # Need enough eigenvalues to decompose into signal and noise
        if len(eigenvalues) < 3:
            return {"error": "Need at least 3 eigenvalues for bulk-spike decomposition"}

        # Simple decomposition: first eigenvalue as signal, rest as noise
        signal_eigenvalues = eigenvalues[:1]
        noise_eigenvalues = eigenvalues[1:]

        # Compute moments for full, signal, and noise components
        full_moments = HeavyTailedAnalysisCore._compute_moments(eigenvalues, max_moment)
        signal_moments = HeavyTailedAnalysisCore._compute_moments(signal_eigenvalues, max_moment)
        noise_moments = HeavyTailedAnalysisCore._compute_moments(noise_eigenvalues, max_moment)

        # Convert moments to free cumulants
        full_cumulants = HeavyTailedAnalysisCore._moments_to_free_cumulants(full_moments)
        signal_cumulants = HeavyTailedAnalysisCore._moments_to_free_cumulants(signal_moments)
        noise_cumulants = HeavyTailedAnalysisCore._moments_to_free_cumulants(noise_moments)

        # Compute interaction strength (skip 0th and 1st cumulants)
        interaction_strength = 0
        for p in range(2, max_moment + 1):
            interaction_strength += abs(full_cumulants[p]) - abs(signal_cumulants[p]) - abs(noise_cumulants[p])

        # Check for random-like behavior using Kurtosis
        # MP law has excess kurtosis of 2 (Gaussian would be 0)
        excess_kurtosis = (full_moments[4] / (full_moments[2] ** 2)) - 3
        mp_deviation = abs(excess_kurtosis - 2)

        # Store results
        result = {
            "interaction_strength": float(interaction_strength),
            "full_cumulants": full_cumulants[2:].tolist(),  # Skip trivial cumulants
            "signal_cumulants": signal_cumulants[2:].tolist(),
            "noise_cumulants": noise_cumulants[2:].tolist(),
            "excess_kurtosis": float(excess_kurtosis),
            "mp_deviation": float(mp_deviation),
        }

        return result


class BaseHeavyTailedAnalyzer(ABC):
    """Base class for Heavy-Tailed Self-Regularization analyses.
    This abstract class defines the common interface for weight and activation space analyzers.
    """

    def __init__(self, boost_neurons: list[int], suppress_neurons: list[int], device=None):
        """Initialize the analyzer with neuron groups."""
        self.boost_neuron_indices = boost_neurons
        self.suppress_neuron_indices = suppress_neurons
        self.device = device
        self.results = {}

        # Combine rare token neurons for some analyses
        self.rare_token_neurons = list(set(boost_neurons + suppress_neurons))

    @abstractmethod
    def compute_correlation_matrix(self, neuron_indices: list[int]) -> np.ndarray:
        """Compute correlation matrix for the specified neurons."""

    @abstractmethod
    def run_all_analyses(self) -> dict[str, Any]:
        """Run all HTSR analyses and return results."""

    def esd_shape_analysis(self, neuron_indices: list[int]) -> dict[str, Any]:
        """Run ESD shape analysis for the given neuron indices."""
        correlation_matrix = self.compute_correlation_matrix(neuron_indices)
        return HeavyTailedAnalysisCore.esd_shape_analysis(correlation_matrix)

    def shape_metrics_analysis(self, neuron_indices: list[int]) -> dict[str, Any]:
        """Run shape metrics analysis for the given neuron indices."""
        correlation_matrix = self.compute_correlation_matrix(neuron_indices)
        return HeavyTailedAnalysisCore.shape_metrics_analysis(correlation_matrix)

    def phase_transition_analysis(self, neuron_indices: list[int]) -> dict[str, Any]:
        """Run phase transition analysis for the given neuron indices."""
        correlation_matrix = self.compute_correlation_matrix(neuron_indices)
        return HeavyTailedAnalysisCore.phase_transition_analysis(correlation_matrix)

    def bulk_spike_interaction_analysis(self, neuron_indices: list[int]) -> dict[str, Any]:
        """Run bulk-spike interaction analysis for the given neuron indices."""
        correlation_matrix = self.compute_correlation_matrix(neuron_indices)
        return HeavyTailedAnalysisCore.bulk_spike_interaction_analysis(correlation_matrix)


class WeightSpaceHeavyTailedAnalyzer(BaseHeavyTailedAnalyzer):
    """Analyzer for Heavy-Tailed Self-Regularization properties in weight space.
    This class analyzes weight matrices directly with no activation matrices.
    """

    def __init__(
        self,
        model,
        layer_num: int,
        boost_neuron_indices: list[int],
        suppress_neuron_indices: list[int],
        excluded_neuron_indices: list[int] = None,
        num_random_groups: int = 2,
        use_mixed_precision: bool = False,
        device: str = None,
    ):
        """Initialize the analyzer with model and neuron groups.

        Args:
            model: The neural network model to analyze
            layer_num: The layer number to analyze
            boost_neurons: List of neuron indices that boost specific tokens
            suppress_neurons: List of neuron indices that suppress specific tokens
            excluded_neuron_indices: List of neuron indices to exclude from analysis
            num_random_groups: Number of random neuron groups to create for comparison
            use_mixed_precision: Whether to use mixed precision calculations
            device: The device to perform calculations on (CPU or GPU)

        """
        super().__init__(boost_neuron_indices, suppress_neuron_indices, device)
        self.model = model
        self.layer_num = layer_num
        self.excluded_neuron_indices = excluded_neuron_indices or []
        self.num_random_groups = num_random_groups
        self.use_mixed_precision = use_mixed_precision

        # Get common neurons and create random groups
        self.common_neurons, self.random_indices = self._get_common_neurons()

        # Create neuron group dictionary
        self.neuron_groups = {
            "boost": self.boost_neuron_indices,
            "suppress": self.suppress_neuron_indices,
            "rare_token": self.rare_token_neurons,
        }

        # Add random groups to the neuron groups dictionary
        for i in range(len(self.random_indices)):
            self.neuron_groups[f"random_{i + 1}"] = self.random_indices[i]

        # Include a "common" sample of similar size to rare token neurons
        sample_size = max(len(boost_neuron_indices), len(suppress_neuron_indices))
        if self.common_neurons and len(self.common_neurons) > sample_size:
            np.random.shuffle(self.common_neurons)
            self.neuron_groups["common"] = self.common_neurons[:sample_size]
        else:
            self.neuron_groups["common"] = self.common_neurons

        # Set up PyTorch dtype
        self.dtype = torch.float16 if self.use_mixed_precision else torch.float32
        logger.info(f"Using device: {self.device}, Mixed precision: {self.use_mixed_precision}")

        # Initialize weight matrices cache
        self.weight_matrices = {}

    def _get_common_neurons(self) -> tuple[list[int], list[list[int]]]:
        """Generate non-overlapping random neuron groups that don't overlap with boost or suppress neurons.

        Returns:
            Tuple containing:
            - List of all common neurons (not boost or suppress)
            - List of lists, where each inner list is a group of random neurons

        """
        # Get layer to determine total neurons
        layer_path = f"gpt_neox.layers.{self.layer_num}.mlp.dense_h_to_4h"
        layer_dict = dict(self.model.named_modules())
        layer = layer_dict[layer_path]
        total_neurons = layer.weight.shape[0]

        # Get all neuron indices
        all_neuron_indices = list(range(total_neurons))

        # Set parameters
        group_size = max(len(self.boost_neuron_indices), len(self.suppress_neuron_indices))

        # Define special indices to exclude (boost and suppress)
        special_indices = set(self.boost_neuron_indices + self.suppress_neuron_indices + self.excluded_neuron_indices)

        # Get non-special neurons (those that are neither boost nor suppress)
        non_special_indices = [idx for idx in all_neuron_indices if idx not in special_indices]

        # Initialize list to store random groups
        random_indices = []

        # Check if we have enough neurons for the desired number of random groups
        if len(non_special_indices) < self.num_random_groups * group_size:
            logger.warning(
                f"Not enough neurons for {self.num_random_groups} non-overlapping random groups of size {group_size}."
            )
            # Not enough neurons - split them evenly into the required number of groups
            np.random.shuffle(non_special_indices)
            split_points = [
                len(non_special_indices) * i // self.num_random_groups for i in range(self.num_random_groups + 1)
            ]

            for i in range(self.num_random_groups):
                random_indices.append(non_special_indices[split_points[i] : split_points[i + 1]])
        else:
            # We have enough neurons - create proper random groups
            np.random.shuffle(non_special_indices)
            for i in range(self.num_random_groups):
                start_idx = i * group_size
                end_idx = (i + 1) * group_size
                if end_idx <= len(non_special_indices):
                    random_indices.append(non_special_indices[start_idx:end_idx])
                else:
                    # If we don't have enough neurons, just use what's left
                    random_indices.append(non_special_indices[start_idx:])

        return non_special_indices, random_indices

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert a PyTorch tensor to NumPy array."""
        return tensor.detach().cpu().numpy()

    def extract_neuron_weights(self, neuron_indices: list[int]) -> np.ndarray:
        """Extract weight vectors for specified neurons in a layer.

        Args:
            neuron_indices: List of neuron indices to extract weights for

        Returns:
            NumPy array of shape (len(neuron_indices), input_dim) containing weights

        """
        # Cache key for the neuron group
        cache_key = tuple(sorted(neuron_indices))

        # Return cached result if available
        if cache_key in self.weight_matrices:
            return self.weight_matrices[cache_key]

        # Get layer weights
        layer_path = f"gpt_neox.layers.{self.layer_num}.mlp.dense_h_to_4h"
        layer_dict = dict(self.model.named_modules())
        layer = layer_dict[layer_path]

        # Get weight matrix
        W = layer.weight.detach().cpu().numpy()

        # Extract weights for specific neurons
        W_neurons = W[neuron_indices]

        # Ensure we return a 2D array
        if len(W_neurons.shape) == 1:
            W_neurons = W_neurons.reshape(1, -1)

        # Store in cache
        self.weight_matrices[cache_key] = W_neurons

        return W_neurons

    def compute_correlation_matrix(self, neuron_indices: list[int]) -> np.ndarray:
        """Compute the correlation matrix for the specified neurons using weights.

        Args:
            neuron_indices: List of neuron indices to compute correlations for

        Returns:
            NumPy array of shape (len(neuron_indices), len(neuron_indices)) with correlation coefficients

        """
        # Extract weight vectors for the specified neurons
        weight_matrix = self.extract_neuron_weights(neuron_indices)

        # Skip calculation if not enough neurons
        if weight_matrix.shape[0] <= 1:
            # Return identity matrix as a fallback
            return np.eye(max(1, weight_matrix.shape[0]))

        # Center each neuron's weights
        centered = weight_matrix - np.mean(weight_matrix, axis=1, keepdims=True)

        # Check for zero variance
        std_values = np.std(weight_matrix, axis=1, keepdims=True)
        # Avoid division by zero
        std_values[std_values < 1e-8] = 1.0

        # Normalize
        normalized = centered / std_values

        # Compute correlation matrix
        n_features = normalized.shape[1]
        correlation_matrix = np.dot(normalized, normalized.T) / n_features

        # Fix potential numerical issues
        correlation_matrix = np.clip(correlation_matrix, -1.0, 1.0)

        # Ensure the diagonal is exactly 1.0
        np.fill_diagonal(correlation_matrix, 1.0)

        return correlation_matrix

    def run_all_analyses(self) -> dict[str, Any]:
        """Run all HTSR analyses for weight space and return results as a dictionary.

        Returns:
            Dictionary containing analysis results for each neuron group

        """
        results = {}

        # For each neuron group
        for group_name, neuron_indices in self.neuron_groups.items():
            if not neuron_indices or len(neuron_indices) < 2:
                results[group_name] = {"error": f"Not enough neurons in {group_name} group for analysis"}
                continue

            group_results = {}

            # Compute correlation matrix once for this group
            try:
                correlation_matrix = self.compute_correlation_matrix(neuron_indices)
            except Exception as e:
                logger.error(f"Error computing correlation matrix for {group_name}: {e}")
                results[group_name] = {"error": f"Failed to compute correlation matrix: {e!s}"}
                continue

            # Run each analysis using the computed correlation matrix
            try:
                group_results["esd_shape"] = HeavyTailedAnalysisCore.esd_shape_analysis(correlation_matrix)
            except Exception as e:
                group_results["esd_shape"] = {"error": str(e)}

            try:
                group_results["shape_metrics"] = HeavyTailedAnalysisCore.shape_metrics_analysis(correlation_matrix)
            except Exception as e:
                group_results["shape_metrics"] = {"error": str(e)}

            try:
                group_results["phase_transition"] = HeavyTailedAnalysisCore.phase_transition_analysis(
                    correlation_matrix
                )
            except Exception as e:
                group_results["phase_transition"] = {"error": str(e)}

            try:
                group_results["bulk_spike_interaction"] = HeavyTailedAnalysisCore.bulk_spike_interaction_analysis(
                    correlation_matrix
                )
            except Exception as e:
                group_results["bulk_spike_interaction"] = {"error": str(e)}

            results[group_name] = group_results

        # Add comparisons
        results["comparisons"] = self._get_comparisons(results)

        return results

    def _get_comparisons(self, group_results: dict[str, Any]) -> dict[str, Any]:
        """Generate comparison metrics between neuron groups.

        Args:
            group_results: Dictionary with results for each neuron group

        Returns:
            Dictionary with comparison metrics

        """
        comparisons = {}

        # Skip if there are errors or missing results
        if "boost" not in group_results or "suppress" not in group_results or "common" not in group_results:
            return {"error": "Missing data for comparisons"}

        # ESD Shape comparisons
        try:
            esd_comparisons = {
                "ks_boost_vs_common": group_results["boost"]["esd_shape"]["ks_statistic"]
                / (group_results["common"]["esd_shape"]["ks_statistic"] + 1e-10),
                "ks_suppress_vs_common": group_results["suppress"]["esd_shape"]["ks_statistic"]
                / (group_results["common"]["esd_shape"]["ks_statistic"] + 1e-10),
                "ks_suppress_vs_boost": group_results["suppress"]["esd_shape"]["ks_statistic"]
                / (group_results["boost"]["esd_shape"]["ks_statistic"] + 1e-10),
                "spike_boost_vs_common": group_results["boost"]["esd_shape"]["spike_separation"]
                / (group_results["common"]["esd_shape"]["spike_separation"] + 1e-10),
                "spike_suppress_vs_common": group_results["suppress"]["esd_shape"]["spike_separation"]
                / (group_results["common"]["esd_shape"]["spike_separation"] + 1e-10),
                "spike_suppress_vs_boost": group_results["suppress"]["esd_shape"]["spike_separation"]
                / (group_results["boost"]["esd_shape"]["spike_separation"] + 1e-10),
            }
            comparisons["esd_shape"] = esd_comparisons
        except:
            comparisons["esd_shape"] = {"error": "Failed to calculate ESD shape comparisons"}

        # Shape metrics comparisons
        try:
            shape_comparisons = {
                "alpha_hill_boost_vs_common": group_results["boost"]["shape_metrics"]["pl_alpha_hill"]
                / (group_results["common"]["shape_metrics"]["pl_alpha_hill"] + 1e-10),
                "alpha_hill_suppress_vs_common": group_results["suppress"]["shape_metrics"]["pl_alpha_hill"]
                / (group_results["common"]["shape_metrics"]["pl_alpha_hill"] + 1e-10),
                "alpha_hill_suppress_vs_boost": group_results["suppress"]["shape_metrics"]["pl_alpha_hill"]
                / (group_results["boost"]["shape_metrics"]["pl_alpha_hill"] + 1e-10),
                "stable_rank_boost_vs_common": group_results["boost"]["shape_metrics"]["stable_rank"]
                / (group_results["common"]["shape_metrics"]["stable_rank"] + 1e-10),
                "stable_rank_suppress_vs_common": group_results["suppress"]["shape_metrics"]["stable_rank"]
                / (group_results["common"]["shape_metrics"]["stable_rank"] + 1e-10),
                "stable_rank_suppress_vs_boost": group_results["suppress"]["shape_metrics"]["stable_rank"]
                / (group_results["boost"]["shape_metrics"]["stable_rank"] + 1e-10),
                "entropy_boost_vs_common": group_results["boost"]["shape_metrics"]["entropy"]
                / (group_results["common"]["shape_metrics"]["entropy"] + 1e-10),
                "entropy_suppress_vs_common": group_results["suppress"]["shape_metrics"]["entropy"]
                / (group_results["common"]["shape_metrics"]["entropy"] + 1e-10),
                "entropy_suppress_vs_boost": group_results["suppress"]["shape_metrics"]["entropy"]
                / (group_results["boost"]["shape_metrics"]["entropy"] + 1e-10),
            }
            comparisons["shape_metrics"] = shape_comparisons
        except:
            comparisons["shape_metrics"] = {"error": "Failed to calculate shape metrics comparisons"}

        # Bulk-spike interaction comparisons
        try:
            interaction_comparisons = {
                "interaction_boost_vs_common": group_results["boost"]["bulk_spike_interaction"]["interaction_strength"]
                / (group_results["common"]["bulk_spike_interaction"]["interaction_strength"] + 1e-10),
                "interaction_suppress_vs_common": group_results["suppress"]["bulk_spike_interaction"][
                    "interaction_strength"
                ]
                / (group_results["common"]["bulk_spike_interaction"]["interaction_strength"] + 1e-10),
                "interaction_suppress_vs_boost": group_results["suppress"]["bulk_spike_interaction"][
                    "interaction_strength"
                ]
                / (group_results["boost"]["bulk_spike_interaction"]["interaction_strength"] + 1e-10),
            }
            comparisons["bulk_spike_interaction"] = interaction_comparisons
        except:
            comparisons["bulk_spike_interaction"] = {"error": "Failed to calculate bulk-spike interaction comparisons"}

        return comparisons


class ActivationSpaceHeavyTailedAnalyzer(BaseHeavyTailedAnalyzer):
    """Analyzer for Heavy-Tailed Self-Regularization properties in activation space."""

    def __init__(
        self,
        activation_data: pd.DataFrame,
        boost_neurons: list[int],
        suppress_neurons: list[int],
        activation_column: str = "activation",
        component_column: str = "component_name",
        token_column: str = "str_tokens",
        context_column: str = "context",
        device=None,
        use_mixed_precision: bool = True,
        random_sample_size: int = None,
    ):
        """Initialize the analyzer with activation data and neuron groups."""
        super().__init__(boost_neurons, suppress_neurons, device)
        self.data = activation_data
        self.activation_column = activation_column
        self.component_column = component_column
        self.token_column = token_column
        self.context_column = context_column
        self.use_mixed_precision = use_mixed_precision

        # Create ID column for token-context pairs
        self.data["token_context_id"] = (
            self.data[token_column].astype(str) + "_" + self.data[context_column].astype(str)
        )

        # Get all unique neurons
        self.all_neuron_indices = self.data[self.component_column].astype(int).unique()

        # Get common neurons and create random samples
        self.common_neurons, self.sampled_common_neurons = self._get_common_neurons(random_sample_size)

        # Create neuron group dictionary
        self.neuron_groups = {
            "boost": self.boost_neurons,
            "suppress": self.suppress_neurons,
            "rare_token": self.rare_token_neurons,
            "common": self.sampled_common_neurons,
        }

        # Set up PyTorch dtype
        self.dtype = torch.float16 if self.use_mixed_precision else torch.float32
        logger.info(f"Using device: {self.device}, Mixed precision: {self.use_mixed_precision}")

        # Initialize activation matrices cache
        self.activation_matrices = {}

    def _get_common_neurons(self, sample_size: int | None = None) -> tuple[list[int], list[int]]:
        """Get neurons that are neither boosting nor suppressing and create a sample."""
        # Get neurons that are neither boosting nor suppressing
        all_special = set(self.boost_neurons + self.suppress_neurons)
        common_neurons = [idx for idx in self.all_neuron_indices if idx not in all_special]

        # Sample a subset of similar size to the rare token neurons if requested
        if sample_size is None:
            sample_size = len(self.rare_token_neurons)

        if len(common_neurons) > sample_size:
            # Shuffle the common neurons
            np.random.shuffle(common_neurons)
            sampled_common_neurons = common_neurons[:sample_size]
        else:
            sampled_common_neurons = common_neurons

        return common_neurons, sampled_common_neurons

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert a PyTorch tensor to NumPy array."""
        return tensor.detach().cpu().numpy()

    def _create_activation_matrix(self, neuron_indices: list[int]) -> np.ndarray:
        """Create an activation matrix where rows are token-context pairs and columns are neurons."""
        # Cache key for the neuron group
        cache_key = tuple(sorted(neuron_indices))

        # Return cached result if available
        if cache_key in self.activation_matrices:
            return self.activation_matrices[cache_key]

        # Filter data to only include specified neurons
        filtered_data = self.data[self.data[self.component_column].isin(neuron_indices)]

        # Pivot to create matrix with token-context pairs as rows and neurons as columns
        pivot_table = filtered_data.pivot_table(
            index="token_context_id",
            columns=self.component_column,
            values=self.activation_column,
            aggfunc="first",  # In case of duplicates, take the first value
        )

        # Handle missing values if any
        pivot_table = pivot_table.fillna(0)

        # Store in cache
        matrix = pivot_table.values
        self.activation_matrices[cache_key] = matrix

        return matrix

    def compute_correlation_matrix(self, neuron_indices: list[int]) -> np.ndarray:
        """Compute the correlation matrix for the specified neurons using activations."""
        # Get activation matrix (neurons as columns, token-contexts as rows)
        activation_matrix = self._create_activation_matrix(neuron_indices)

        # Transpose to get neurons as rows
        neuron_activations = activation_matrix.T

        # Skip calculation if not enough neurons or contexts
        if neuron_activations.shape[0] <= 1 or neuron_activations.shape[1] == 0:
            # Return identity matrix as a fallback
            return np.eye(max(1, neuron_activations.shape[0]))

        # Center each neuron's activations
        centered = neuron_activations - np.mean(neuron_activations, axis=1, keepdims=True)

        # Check for zero variance
        std_values = np.std(neuron_activations, axis=1, keepdims=True)
        # Avoid division by zero
        std_values[std_values < 1e-8] = 1.0

        # Normalize
        normalized = centered / std_values

        # Compute correlation matrix
        n_samples = normalized.shape[1]
        correlation_matrix = np.dot(normalized, normalized.T) / n_samples

        # Fix potential numerical issues
        correlation_matrix = np.clip(correlation_matrix, -1.0, 1.0)

        return correlation_matrix

    def compute_enhanced_correlation_matrix(self, neuron_indices: list[int], use_gpu: bool = True) -> np.ndarray:
        """Compute correlation matrix with enhanced efficiency using GPU if available.
        This method is an alternative to compute_correlation_matrix for larger datasets.
        """
        # Get activation matrix
        activation_matrix = self._create_activation_matrix(neuron_indices)

        # If GPU is available and requested, use it
        if use_gpu and torch.cuda.is_available() and self.device:
            # Convert to PyTorch tensor
            tensor = torch.tensor(activation_matrix, dtype=self.dtype).to(self.device)

            # Transpose to get neurons as rows
            neuron_tensor = tensor.t()

            # Center each neuron's activations
            centered = neuron_tensor - torch.mean(neuron_tensor, dim=1, keepdim=True)

            # Compute standard deviation
            std_values = torch.std(neuron_tensor, dim=1, keepdim=True)
            # Avoid division by zero
            std_values[std_values < 1e-8] = 1.0

            # Normalize
            normalized = centered / std_values

            # Compute correlation matrix
            n_samples = normalized.shape[1]
            correlation_tensor = torch.mm(normalized, normalized.t()) / n_samples

            # Move to CPU and convert to numpy
            correlation_matrix = self._to_numpy(correlation_tensor)

            # Clean up GPU memory
            del tensor, neuron_tensor, centered, normalized, correlation_tensor
            cleanup()
        else:
            # Fall back to NumPy implementation
            correlation_matrix = self.compute_correlation_matrix(neuron_indices)

        # Fix potential numerical issues
        correlation_matrix = np.clip(correlation_matrix, -1.0, 1.0)

        return correlation_matrix

    def run_analyses(self, use_gpu: bool = True) -> dict[str, Any]:
        """Run all HTSR analyses for activation space and return results as a dictionary."""
        results = {}

        # For each neuron group
        for group_name, neuron_indices in self.neuron_groups.items():
            if not neuron_indices or len(neuron_indices) < 2:
                results[group_name] = {"error": f"Not enough neurons in {group_name} group for analysis"}
                continue

            group_results = {}

            # Compute correlation matrix once for this group
            try:
                correlation_matrix = self.compute_enhanced_correlation_matrix(neuron_indices, use_gpu)
            except Exception as e:
                logger.error(f"Error computing correlation matrix for {group_name}: {e}")
                results[group_name] = {"error": f"Failed to compute correlation matrix: {e!s}"}
                continue

            # Run each analysis using the computed correlation matrix
            try:
                group_results["esd_shape"] = HeavyTailedAnalysisCore.esd_shape_analysis(correlation_matrix)
            except Exception as e:
                group_results["esd_shape"] = {"error": str(e)}

            try:
                group_results["shape_metrics"] = HeavyTailedAnalysisCore.shape_metrics_analysis(correlation_matrix)
            except Exception as e:
                group_results["shape_metrics"] = {"error": str(e)}

            try:
                group_results["phase_transition"] = HeavyTailedAnalysisCore.phase_transition_analysis(
                    correlation_matrix
                )
            except Exception as e:
                group_results["phase_transition"] = {"error": str(e)}

            try:
                group_results["bulk_spike_interaction"] = HeavyTailedAnalysisCore.bulk_spike_interaction_analysis(
                    correlation_matrix
                )
            except Exception as e:
                group_results["bulk_spike_interaction"] = {"error": str(e)}

            results[group_name] = group_results

        # Add comparisons
        results["comparisons"] = self._get_comparisons(results)

        # Cleanup memory
        cleanup()

        return results

    def _get_comparisons(self, group_results: dict[str, Any]) -> dict[str, Any]:
        """Generate comparison metrics between neuron groups."""
        comparisons = {}

        # Skip if there are errors or missing results
        if "boost" not in group_results or "suppress" not in group_results or "common" not in group_results:
            return {"error": "Missing data for comparisons"}

        # ESD Shape comparisons
        try:
            esd_comparisons = {
                "ks_boost_vs_common": group_results["boost"]["esd_shape"]["ks_statistic"]
                / (group_results["common"]["esd_shape"]["ks_statistic"] + 1e-10),
                "ks_suppress_vs_common": group_results["suppress"]["esd_shape"]["ks_statistic"]
                / (group_results["common"]["esd_shape"]["ks_statistic"] + 1e-10),
                "ks_suppress_vs_boost": group_results["suppress"]["esd_shape"]["ks_statistic"]
                / (group_results["boost"]["esd_shape"]["ks_statistic"] + 1e-10),
                "spike_boost_vs_common": group_results["boost"]["esd_shape"]["spike_separation"]
                / (group_results["common"]["esd_shape"]["spike_separation"] + 1e-10),
                "spike_suppress_vs_common": group_results["suppress"]["esd_shape"]["spike_separation"]
                / (group_results["common"]["esd_shape"]["spike_separation"] + 1e-10),
                "spike_suppress_vs_boost": group_results["suppress"]["esd_shape"]["spike_separation"]
                / (group_results["boost"]["esd_shape"]["spike_separation"] + 1e-10),
            }
            comparisons["esd_shape"] = esd_comparisons
        except:
            comparisons["esd_shape"] = {"error": "Failed to calculate ESD shape comparisons"}

        # Shape metrics comparisons
        try:
            shape_comparisons = {
                "alpha_hill_boost_vs_common": group_results["boost"]["shape_metrics"]["pl_alpha_hill"]
                / (group_results["common"]["shape_metrics"]["pl_alpha_hill"] + 1e-10),
                "alpha_hill_suppress_vs_common": group_results["suppress"]["shape_metrics"]["pl_alpha_hill"]
                / (group_results["common"]["shape_metrics"]["pl_alpha_hill"] + 1e-10),
                "alpha_hill_suppress_vs_boost": group_results["suppress"]["shape_metrics"]["pl_alpha_hill"]
                / (group_results["boost"]["shape_metrics"]["pl_alpha_hill"] + 1e-10),
                "stable_rank_boost_vs_common": group_results["boost"]["shape_metrics"]["stable_rank"]
                / (group_results["common"]["shape_metrics"]["stable_rank"] + 1e-10),
                "stable_rank_suppress_vs_common": group_results["suppress"]["shape_metrics"]["stable_rank"]
                / (group_results["common"]["shape_metrics"]["stable_rank"] + 1e-10),
                "stable_rank_suppress_vs_boost": group_results["suppress"]["shape_metrics"]["stable_rank"]
                / (group_results["boost"]["shape_metrics"]["stable_rank"] + 1e-10),
                "entropy_boost_vs_common": group_results["boost"]["shape_metrics"]["entropy"]
                / (group_results["common"]["shape_metrics"]["entropy"] + 1e-10),
                "entropy_suppress_vs_common": group_results["suppress"]["shape_metrics"]["entropy"]
                / (group_results["common"]["shape_metrics"]["entropy"] + 1e-10),
                "entropy_suppress_vs_boost": group_results["suppress"]["shape_metrics"]["entropy"]
                / (group_results["boost"]["shape_metrics"]["entropy"] + 1e-10),
            }
            comparisons["shape_metrics"] = shape_comparisons
        except:
            comparisons["shape_metrics"] = {"error": "Failed to calculate shape metrics comparisons"}

        # Bulk-spike interaction comparisons
        try:
            interaction_comparisons = {
                "interaction_boost_vs_common": group_results["boost"]["bulk_spike_interaction"]["interaction_strength"]
                / (group_results["common"]["bulk_spike_interaction"]["interaction_strength"] + 1e-10),
                "interaction_suppress_vs_common": group_results["suppress"]["bulk_spike_interaction"][
                    "interaction_strength"
                ]
                / (group_results["common"]["bulk_spike_interaction"]["interaction_strength"] + 1e-10),
                "interaction_suppress_vs_boost": group_results["suppress"]["bulk_spike_interaction"][
                    "interaction_strength"
                ]
                / (group_results["boost"]["bulk_spike_interaction"]["interaction_strength"] + 1e-10),
            }
            comparisons["bulk_spike_interaction"] = interaction_comparisons
        except:
            comparisons["bulk_spike_interaction"] = {"error": "Failed to calculate bulk-spike interaction comparisons"}

        return comparisons

    def export_results_json(self, file_path: str = None, use_gpu: bool = True) -> str:
        """Run all analyses and export results to JSON file."""
        results = self.run_analyses(use_gpu)

        if file_path:
            with open(file_path, "w") as f:
                json.dump(results, f, indent=2)
            return f"Results saved to {file_path}"
        return json.dumps(results, indent=2)
