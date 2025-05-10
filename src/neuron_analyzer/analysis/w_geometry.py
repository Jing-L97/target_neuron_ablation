#!/usr/bin/env python
import logging

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cosine
from scipy.stats import ttest_ind

from neuron_analyzer.selection.neuron import generate_random_indices

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

#######################################################
# Neuron group subspace direction analysis


class WeightGeometricAnalyzer:
    """Analyzer for geometric properties of weight vectors in neural networks."""

    def __init__(
        self,
        model,
        layer_num: int,
        boost_neuron_indices: list[int],
        suppress_neuron_indices: list[int],
        excluded_neuron_indices: list[int] = None,
        num_random_groups: int = 2,
    ):
        """Initialize with model and neuron groups.

        Args:
            model: The neural network model
            layer_num: Layer number to analyze
            boost_neuron_indices: Indices of boost neurons
            suppress_neuron_indices: Indices of suppress neurons
            excluded_neuron_indices: Additional indices to exclude from random groups
            num_random_groups: Number of random control groups to create

        """
        self.boost_neuron_indices = boost_neuron_indices
        self.suppress_neuron_indices = suppress_neuron_indices
        self.model = model
        self.layer_num = layer_num
        self.num_random_groups = num_random_groups
        self.excluded_neuron_indices = excluded_neuron_indices or []

        # Get common neurons and create random groups
        self.random_indices = self._get_common_neurons()

        # Store neuron indices in a dictionary for easy access
        self.neuron_indices = {
            "boost": self.boost_neuron_indices,
            "suppress": self.suppress_neuron_indices,
        }

        # Add random groups to the neuron indices dictionary
        for i in range(len(self.random_indices)):
            self.neuron_indices[f"random_{i + 1}"] = self.random_indices[i]

        # Initialize results dictionaries
        self.dimensionality_results: dict = {}
        self.orthogonality_results: dict = {}
        self.comparative_results: dict = {}

    def _get_common_neurons(self) -> list[list[int]]:
        """Generate non-overlapping random neuron groups that don't overlap with boost or suppress neurons.

        Returns:
            List of lists containing random neuron indices

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

        # Get random indices using the helper function
        random_indices = generate_random_indices(
            all_neuron_indices, special_indices, group_size, self.num_random_groups
        )
        return random_indices

    def extract_neuron_weights(self, neuron_indices: list[int]) -> np.ndarray:
        """Extract weight vectors for specified neurons in a layer.

        Args:
            neuron_indices: Indices of neurons to extract weights for

        Returns:
            Array of weight vectors for the specified neurons

        """
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

        return W_neurons

    def _safe_ttest(self, value: float, comparison_values: list[float]) -> tuple[float, float, bool, str]:
        """Safely perform a t-test handling edge cases and potential errors.

        Args:
            value: Single value for comparison
            comparison_values: List of values to compare against

        Returns:
            Tuple containing t-statistic, p-value, significance flag, and comparison direction

        """
        if not comparison_values:
            return 0.0, 1.0, False, "unknown"

        # If all values are identical, no statistical test is needed
        if all(v == comparison_values[0] for v in comparison_values) and value == comparison_values[0]:
            return 0.0, 1.0, False, "equal"

        try:
            # Use one-sample t-test against the mean
            mean_comparison = np.mean(comparison_values)
            std_comparison = np.std(comparison_values)

            # If standard deviation is zero, we can't perform t-test
            if std_comparison == 0:
                return 0.0, 1.0, False, "higher" if value > mean_comparison else "lower"

            # Use independent t-test with proper handling
            tstat, pvalue = ttest_ind([value], comparison_values, equal_var=False)

            # Determine the direction of the difference
            comparison = "higher" if value > mean_comparison else "lower"

            return float(tstat), float(pvalue), bool(pvalue < 0.05), comparison

        except Exception as e:
            print(f"Error performing t-test: {e}")
            return 0.0, 1.0, False, "unknown"

    def analyze_dimensionality(self, variance_threshold: float = 0.95) -> dict:
        """Analyze the dimensionality of each neuron group's weight space.

        Args:
            variance_threshold: Threshold for explained variance to determine effective dimensionality

        Returns:
            Dictionary containing dimensionality analysis results

        """
        results = {}

        # Analyze each neuron group
        for group_name, neuron_indices in self.neuron_indices.items():
            W_group = self.extract_neuron_weights(neuron_indices)

            # Perform SVD on the neuron group
            U, S, Vh = np.linalg.svd(W_group, full_matrices=False)

            # Calculate normalized singular values
            S_norm = S / S.sum() if S.sum() > 0 else S

            # Calculate cumulative explained variance
            cum_var = np.cumsum(S_norm)

            # Estimate effective dimensionality (number of dimensions for variance_threshold variance)
            effective_dim = (
                np.argmax(cum_var >= variance_threshold) + 1 if np.any(cum_var >= variance_threshold) else len(S)
            )

            # Store metrics
            result = {
                "effective_dim": int(effective_dim),
                "total_dim": len(neuron_indices),
                "dim_prop": float(effective_dim / len(neuron_indices)),
                "explained_variance_ratio": S_norm,
                "cumulative_variance": cum_var,
                "right_singular_vectors": Vh,
            }

            results[group_name] = result

        # Compare boost vs random_1 and suppress vs random_1
        if "boost" in results and "random_1" in results:
            tstat, pvalue, is_significant, comparison = self._safe_ttest(
                results["boost"]["effective_dim"], [results["random_1"]["effective_dim"]]
            )
            results["boost_vs_random_1"] = {
                "ttest_stat": float(tstat),
                "ttest_p": float(pvalue),
                "is_significantly_different": bool(is_significant),
                "comparison": comparison,
            }

        if "suppress" in results and "random_1" in results:
            tstat, pvalue, is_significant, comparison = self._safe_ttest(
                results["suppress"]["effective_dim"], [results["random_1"]["effective_dim"]]
            )
            results["suppress_vs_random_1"] = {
                "ttest_stat": float(tstat),
                "ttest_p": float(pvalue),
                "is_significantly_different": bool(is_significant),
                "comparison": comparison,
            }

        # Compare random_1 vs random_2
        if "random_1" in results and "random_2" in results:
            tstat, pvalue, is_significant, comparison = self._safe_ttest(
                results["random_1"]["effective_dim"], [results["random_2"]["effective_dim"]]
            )
            results["random_1_vs_random_2"] = {
                "ttest_stat": float(tstat),
                "ttest_p": float(pvalue),
                "is_significantly_different": bool(is_significant),
                "comparison": comparison,
            }

        # Compare boost vs suppress
        if "boost" in results and "suppress" in results:
            tstat, pvalue, is_significant, comparison = self._safe_ttest(
                results["boost"]["effective_dim"], [results["suppress"]["effective_dim"]]
            )
            results["boost_vs_suppress"] = {
                "ttest_stat": float(tstat),
                "ttest_p": float(pvalue),
                "is_significantly_different": bool(is_significant),
                "comparison": comparison,
            }

        self.dimensionality_results = results
        return results

    def compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity value (-1 to 1)

        """
        # Handle zero vectors
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0

        # Calculate cosine similarity (1 - cosine distance)
        return 1 - cosine(vec1, vec2)

    def vector_angle_degrees(self, cosine_similarity: float) -> float:
        """Convert cosine similarity to angle in degrees.

        Args:
            cosine_similarity: Cosine similarity value (-1 to 1)

        Returns:
            Angle in degrees (0 to 180)

        """
        # Clip to handle floating point precision issues
        sim_clipped = np.clip(cosine_similarity, -1.0, 1.0)
        return np.degrees(np.arccos(sim_clipped))

    def _calculate_orthogonality_metrics(self, weight_vectors: np.ndarray) -> dict:
        """Calculate comprehensive orthogonality metrics within a group of weight vectors.

        Args:
            weight_vectors: Array of weight vectors [n_neurons, n_features]

        Returns:
            Dictionary containing orthogonality metrics

        """
        n_vectors = weight_vectors.shape[0]
        if n_vectors <= 1:
            return {
                "mean_cosine_similarity": 0.0,
                "median_cosine_similarity": 0.0,
                "max_cosine_similarity": 0.0,
                "min_cosine_similarity": 0.0,
                "angle_distribution": [],
                "mean_angle_degrees": 0.0,
                "pct_near_orthogonal": 0.0,
            }

        # Calculate all pairwise cosine similarities and angles
        cosine_similarities = []
        angles = []

        for i in range(n_vectors):
            for j in range(i + 1, n_vectors):
                cos_sim = self.compute_cosine_similarity(weight_vectors[i], weight_vectors[j])
                cosine_similarities.append(cos_sim)
                angles.append(self.vector_angle_degrees(cos_sim))

        # Convert to numpy arrays for easier calculations
        cosine_similarities = np.array(cosine_similarities)
        angles = np.array(angles)

        # Calculate metrics
        result = {
            "mean_cosine_similarity": float(np.mean(cosine_similarities)),
            "median_cosine_similarity": float(np.median(cosine_similarities)),
            "max_cosine_similarity": float(np.max(cosine_similarities)),
            "min_cosine_similarity": float(np.min(cosine_similarities)),
            "angle_distribution": angles.tolist(),
            "mean_angle_degrees": float(np.mean(angles)),
            "pct_near_orthogonal": float(((angles >= 80) & (angles <= 100)).mean()),
        }

        return result

    def _calculate_between_orthogonality_metrics(
        self, weight_vectors_1: np.ndarray, weight_vectors_2: np.ndarray
    ) -> dict:
        """Calculate orthogonality metrics between two groups of weight vectors.

        Args:
            weight_vectors_1: Array of weight vectors for first group [n1_neurons, n_features]
            weight_vectors_2: Array of weight vectors for second group [n2_neurons, n_features]

        Returns:
            Dictionary containing cross-group orthogonality metrics

        """
        if weight_vectors_1.shape[0] == 0 or weight_vectors_2.shape[0] == 0:
            return {
                "mean_cross_cosine_similarity": 0.0,
                "median_cross_cosine_similarity": 0.0,
                "max_cross_cosine_similarity": 0.0,
                "min_cross_cosine_similarity": 0.0,
                "cross_angle_distribution": [],
                "mean_cross_angle_degrees": 0.0,
                "pct_cross_near_orthogonal": 0.0,
            }

        # Calculate all pairwise cosine similarities and angles between the groups
        cross_cosine_similarities = []
        cross_angles = []

        for i in range(weight_vectors_1.shape[0]):
            for j in range(weight_vectors_2.shape[0]):
                cos_sim = self.compute_cosine_similarity(weight_vectors_1[i], weight_vectors_2[j])
                cross_cosine_similarities.append(cos_sim)
                cross_angles.append(self.vector_angle_degrees(cos_sim))

        # Convert to numpy arrays
        cross_cosine_similarities = np.array(cross_cosine_similarities)
        cross_angles = np.array(cross_angles)

        # Calculate metrics
        result = {
            "mean_cross_cosine_similarity": float(np.mean(cross_cosine_similarities)),
            "median_cross_cosine_similarity": float(np.median(cross_cosine_similarities)),
            "max_cross_cosine_similarity": float(np.max(cross_cosine_similarities)),
            "min_cross_cosine_similarity": float(np.min(cross_cosine_similarities)),
            "cross_angle_distribution": cross_angles.tolist(),
            "mean_cross_angle_degrees": float(np.mean(cross_angles)),
            "pct_cross_near_orthogonal": float(((cross_angles >= 80) & (cross_angles <= 100)).mean()),
        }

        return result

    def analyze_orthogonality(self) -> dict:
        """Analyze orthogonality within and between neuron groups.

        Returns:
            Dictionary containing orthogonality analysis results

        """
        results = {"within": {}, "between": {}}

        # Extract weights for all groups first
        group_weights = {}
        for group_name, indices in self.neuron_indices.items():
            group_weights[group_name] = self.extract_neuron_weights(indices)

        # Within-group orthogonality
        for group_name, weight_vectors in group_weights.items():
            results["within"][group_name] = self._calculate_orthogonality_metrics(weight_vectors)

        # Between-group orthogonality
        # Define the pairs to analyze
        pairs = [("boost", "random_1"), ("suppress", "random_1"), ("boost", "suppress"), ("random_1", "random_2")]

        for group1, group2 in pairs:
            if group1 in group_weights and group2 in group_weights:
                pair_name = f"{group1}_vs_{group2}"
                results["between"][pair_name] = self._calculate_between_orthogonality_metrics(
                    group_weights[group1], group_weights[group2]
                )

        # Statistical comparisons for within-group metrics
        comparisons = [("boost", "random_1"), ("suppress", "random_1"), ("boost", "suppress"), ("random_1", "random_2")]

        statistical_results = {}
        for group1, group2 in comparisons:
            if group1 in results["within"] and group2 in results["within"]:
                # Compare mean angle degrees
                tstat, pvalue, is_significant, comparison = self._safe_ttest(
                    results["within"][group1]["mean_angle_degrees"],
                    [results["within"][group2]["mean_angle_degrees"]],
                )

                statistical_results[f"{group1}_vs_{group2}_angle"] = {
                    "ttest_stat": float(tstat),
                    "ttest_p": float(pvalue),
                    "is_significantly_different": bool(is_significant),
                    "comparison": comparison,
                }

                # Compare percentage of near-orthogonal angles
                tstat, pvalue, is_significant, comparison = self._safe_ttest(
                    results["within"][group1]["pct_near_orthogonal"], [results["within"][group2]["pct_near_orthogonal"]]
                )

                statistical_results[f"{group1}_vs_{group2}_orthogonal"] = {
                    "ttest_stat": float(tstat),
                    "ttest_p": float(pvalue),
                    "is_significantly_different": bool(is_significant),
                    "comparison": comparison,
                }

        results["statistical_tests"] = statistical_results
        self.orthogonality_results = results

        return results

    def _filter_large_arrays(self, results: dict) -> dict:
        """Filter out large arrays from results dict to make it more manageable.

        Args:
            results: Dictionary containing analysis results

        Returns:
            Filtered dictionary with large arrays removed

        """
        if isinstance(results, dict):
            filtered = {}
            for k, v in results.items():
                if isinstance(v, np.ndarray) and v.size > 1000:
                    # Skip large arrays
                    continue
                if isinstance(v, dict):
                    # Recursively filter nested dictionaries
                    filtered[k] = self._filter_large_arrays(v)
                else:
                    filtered[k] = v
            return filtered
        return results

    def run_all_analyses(self) -> dict:
        """Run all geometric analyses and compile comprehensive results.

        Returns:
            Dictionary containing all analysis results

        """
        dimensionality_results = self.analyze_dimensionality()
        orthogonality_results = self.analyze_orthogonality()

        # Compile summary findings
        summary = {}

        # Dimensionality findings
        if (
            "boost_vs_random_1" in dimensionality_results
            and dimensionality_results["boost_vs_random_1"]["is_significantly_different"]
        ):
            if dimensionality_results["boost_vs_random_1"]["comparison"] == "lower":
                summary["boost_dimensionality"] = (
                    "The boost neuron group has significantly lower dimensionality than random groups, "
                    "suggesting coordinated/synergistic weight patterns."
                )
            else:
                summary["boost_dimensionality"] = (
                    "The boost neuron group has significantly higher dimensionality than random groups, "
                    "suggesting diverse and independent weight patterns."
                )
        else:
            summary["boost_dimensionality"] = (
                "The boost neuron group's dimensionality is not significantly different from random groups."
            )

        if (
            "suppress_vs_random_1" in dimensionality_results
            and dimensionality_results["suppress_vs_random_1"]["is_significantly_different"]
        ):
            if dimensionality_results["suppress_vs_random_1"]["comparison"] == "lower":
                summary["suppress_dimensionality"] = (
                    "The suppress neuron group has significantly lower dimensionality than random groups, "
                    "suggesting coordinated/synergistic weight patterns."
                )
            else:
                summary["suppress_dimensionality"] = (
                    "The suppress neuron group has significantly higher dimensionality than random groups, "
                    "suggesting diverse and independent weight patterns."
                )
        else:
            summary["suppress_dimensionality"] = (
                "The suppress neuron group's dimensionality is not significantly different from random groups."
            )

        # Orthogonality findings
        for group in ["boost", "suppress"]:
            if (
                f"{group}_vs_random_1_angle" in orthogonality_results["statistical_tests"]
                and orthogonality_results["statistical_tests"][f"{group}_vs_random_1_angle"][
                    "is_significantly_different"
                ]
            ):
                if orthogonality_results["statistical_tests"][f"{group}_vs_random_1_angle"]["comparison"] == "higher":
                    summary[f"{group}_orthogonality"] = (
                        f"The {group} neuron group shows significantly higher angles between weight vectors than random groups, "
                        "suggesting more orthogonal functionality."
                    )
                else:
                    summary[f"{group}_orthogonality"] = (
                        f"The {group} neuron group shows significantly lower angles between weight vectors than random groups, "
                        "suggesting more aligned/coordinated patterns."
                    )
            else:
                summary[f"{group}_orthogonality"] = (
                    f"The {group} neuron group's weight vector alignments are not significantly different from random groups."
                )

        # Between boost and suppress
        if (
            "boost_vs_suppress" in dimensionality_results
            and dimensionality_results["boost_vs_suppress"]["is_significantly_different"]
        ):
            if dimensionality_results["boost_vs_suppress"]["comparison"] == "higher":
                summary["boost_vs_suppress_dimensionality"] = (
                    "Boost neurons show significantly higher weight space dimensionality than suppress neurons."
                )
            else:
                summary["boost_vs_suppress_dimensionality"] = (
                    "Boost neurons show significantly lower weight space dimensionality than suppress neurons."
                )
        else:
            summary["boost_vs_suppress_dimensionality"] = (
                "Boost and suppress neurons have similar weight space dimensionality."
            )

        # Create combined results
        combined_results = {
            "dimensionality": self._filter_large_arrays(dimensionality_results),
            "orthogonality": self._filter_large_arrays(orthogonality_results),
            "summary": summary,
            "neuron_indices": self.neuron_indices,
        }

        self.comparative_results = combined_results
        return combined_results


#######################################################
# Neuron direction analysis


def analyze_neuron_directions(model, layer_num=-1, chunk_size=1024, device=None) -> pd.DataFrame:
    """Analyze orthogonality between all neurons in a layer with optimized computation."""
    # Get weight matrix directly on the device where the model is
    input_layer_path = f"gpt_neox.layers.{layer_num}.mlp.dense_h_to_4h"
    layer_dict = dict(model.named_modules())
    input_layer = layer_dict[input_layer_path]

    # Get the weight matrix directly on the appropriate device
    W_in = input_layer.weight.detach()
    if W_in.device != device:
        W_in = W_in.to(device)

    # Get dimensions
    intermediate_size, hidden_size = W_in.shape
    # Normalize all neuron directions
    norm = torch.norm(W_in, dim=1, keepdim=True)
    normalized_directions = W_in / (norm + 1e-8)

    # Compute the cosine similarity matrix more efficiently
    # We can use chunking to avoid memory issues for large matrices
    cosine_sim_matrix = torch.zeros((intermediate_size, intermediate_size), device=device)

    # Calculate only the lower triangular part (including diagonal)
    for i in range(0, intermediate_size, chunk_size):
        end_i = min(i + chunk_size, intermediate_size)
        chunk_i = normalized_directions[i:end_i]

        # Calculate similarity with all neurons up to and including this chunk
        for j in range(0, end_i, chunk_size):
            end_j = min(j + chunk_size, end_i)  # Only calculate lower triangle
            chunk_j = normalized_directions[j:end_j]

            # Compute cosine similarity for this block
            block_sim = torch.matmul(chunk_i, chunk_j.T)

            # Fill in the corresponding part of the matrix
            cosine_sim_matrix[i:end_i, j:end_j] = block_sim

    # Make the matrix symmetric by copying the lower triangle to the upper triangle
    indices = torch.triu_indices(intermediate_size, intermediate_size, 1, device=device)
    cosine_sim_matrix[indices[1], indices[0]] = cosine_sim_matrix[indices[0], indices[1]]

    # Set diagonal to zero (self-similarity is not relevant for orthogonality analysis)
    cosine_sim_matrix.fill_diagonal_(0)

    # Move to CPU for DataFrame conversion
    cosine_sim_matrix_cpu = cosine_sim_matrix.cpu()

    # Convert to DataFrame with neuron indices as both row and column labels
    cosine_df = pd.DataFrame(
        cosine_sim_matrix_cpu.numpy(), index=list(range(intermediate_size)), columns=list(range(intermediate_size))
    )

    return cosine_df


def get_stat(cosine_df, neuron_idx: list, threshold: float = 0.1) -> pd.DataFrame:
    """Compute neuron statistics based on cosine similarity matrix."""
    # Ensure all neurons are in the dataframe
    all_neurons = set(cosine_df.index)
    valid_neurons = [n for n in neuron_idx if n in all_neurons]

    if len(valid_neurons) < len(neuron_idx):
        missing = set(neuron_idx) - all_neurons
        print(f"Warning: {len(missing)} neurons not found in the matrix: {missing}")

    if not valid_neurons:
        raise ValueError("No valid neurons found in the cosine similarity matrix")

    # Create a mask for neurons in the list and outside the list
    neuron_set = set(valid_neurons)
    all_neurons = set(cosine_df.index)
    neurons_outside = list(all_neurons - neuron_set)

    # Prepare results container
    results = []

    # Process each neuron
    for neuron in valid_neurons:
        # Get cosine similarities
        similarities = cosine_df.loc[neuron]

        # Within list statistics (excluding self)
        within_list = [similarities[n] for n in valid_neurons if n != neuron]
        if within_list:
            within_mean = sum(abs(v) for v in within_list) / len(within_list)
            within_max = max(abs(v) for v in within_list)
            within_min = min(abs(v) for v in within_list)
            within_null_count = sum(1 for v in within_list if abs(v) < threshold)
            within_null_percent = (within_null_count / len(within_list)) * 100 if within_list else 0
        else:
            within_mean = within_max = within_min = within_null_count = within_null_percent = 0

        # Outside list statistics
        outside_list = [similarities[n] for n in neurons_outside]
        if outside_list:
            outside_mean = sum(abs(v) for v in outside_list) / len(outside_list)
            outside_max = max(abs(v) for v in outside_list)
            outside_min = min(abs(v) for v in outside_list)
            outside_null_count = sum(1 for v in outside_list if abs(v) < threshold)
            outside_null_percent = (outside_null_count / len(outside_list)) * 100
        else:
            outside_mean = outside_max = outside_min = outside_null_count = outside_null_percent = 0

        # Add to results (two rows per neuron - within and outside)
        results.append(
            {
                "neuron_idx": neuron,
                "type": "within_list",
                "mean_abs_cosine": within_mean,
                "max_abs_cosine": within_max,
                "min_abs_cosine": within_min,
                "null_space_count": within_null_count,
                "null_space_percent": within_null_percent,
                "total_comparisons": len(within_list),
            }
        )

        results.append(
            {
                "neuron_idx": neuron,
                "type": "outside_list",
                "mean_abs_cosine": outside_mean,
                "max_abs_cosine": outside_max,
                "min_abs_cosine": outside_min,
                "null_space_count": outside_null_count,
                "null_space_percent": outside_null_percent,
                "total_comparisons": len(outside_list),
            }
        )

    # Convert to DataFrame
    stat_df = pd.DataFrame(results)

    # Add interpretation column
    def interpret_orthogonality(row):
        if row["null_space_percent"] > 75:
            return "highly_orthogonal"
        if row["null_space_percent"] > 50:
            return "moderately_orthogonal"
        if row["null_space_percent"] > 25:
            return "slightly_orthogonal"
        return "not_orthogonal"

    stat_df["orthogonality_category"] = stat_df.apply(interpret_orthogonality, axis=1)

    return stat_df
