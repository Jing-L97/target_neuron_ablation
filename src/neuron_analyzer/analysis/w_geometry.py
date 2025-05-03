#!/usr/bin/env python
import logging

import numpy as np
import pandas as pd
import torch
from scipy.linalg import subspace_angles
from scipy.stats import ttest_ind

from neuron_analyzer.selection.neuron import generate_random_indices

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

#######################################################
# Neuron group subspace direction analysis


class WeightGeometricAnalyzer:
    def __init__(
        self,
        model,
        layer_num: int,
        boost_neuron_indices: list[int],
        suppress_neuron_indices: list[int],
        excluded_neuron_indices: list[int] = None,
        num_random_groups: int = 2,
    ):
        """Initialize with model and neuron groups."""
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

    def _get_common_neurons(self) -> tuple[list[int], list[list[int]]]:
        """Generate non-overlapping random neuron groups that don't overlap with boost or suppress neurons."""
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
        special_indices = set(self.boost_neuron_indices + self.suppress_neuron_indices)

        # Get non-special neurons (those that are neither boost nor suppress)
        random_indices = generate_random_indices(
            all_neuron_indices, special_indices, group_size, self.num_random_groups
        )
        return random_indices

    def extract_neuron_weights(self, neuron_indices: list[int]) -> np.ndarray:
        """Extract weight vectors for specified neurons in a layer."""
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
        """Safely perform a t-test handling edge cases and potential errors."""
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
        """Analyze the dimensionality of each neuron group's weight space."""
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

    def _calculate_orthogonality_metrics(self, Vh: np.ndarray, dim: int) -> dict:
        """Calculate orthogonality metrics within a single group's subspace."""
        if Vh is None or dim <= 1:
            return {
                "mean_angle_degrees": 0.0,
                "median_angle_degrees": 0.0,
                "min_angle_degrees": 0.0,
                "max_angle_degrees": 0.0,
                "pct_near_orthogonal": 0.0,
                "is_self_pair": True,
            }

        # Make sure we don't exceed available dimensions
        dim = min(dim, Vh.shape[0])

        # Use effective dimensions to define the subspace
        V = Vh[:dim].T  # Column vectors spanning the subspace

        # Calculate angles between all pairs of basis vectors within the subspace
        full_angles_degrees = []

        for i in range(V.shape[1]):
            v1 = V[:, i]
            v1_norm = np.linalg.norm(v1)

            for j in range(i + 1, V.shape[1]):  # Only need upper triangle
                v2 = V[:, j]
                v2_norm = np.linalg.norm(v2)

                # Avoid division by zero
                if v1_norm > 0 and v2_norm > 0:
                    # Calculate dot product
                    dot_product = np.dot(v1, v2) / (v1_norm * v2_norm)
                    # Clamp to avoid numerical errors
                    dot_product = max(min(dot_product, 1.0), -1.0)
                    # Calculate angle in degrees
                    angle_degrees = np.degrees(np.arccos(dot_product))
                    full_angles_degrees.append(angle_degrees)

        if not full_angles_degrees:
            return {
                "mean_angle_degrees": 0.0,
                "median_angle_degrees": 0.0,
                "min_angle_degrees": 0.0,
                "max_angle_degrees": 0.0,
                "pct_near_orthogonal": 0.0,
                "is_self_pair": True,
            }

        full_angles_degrees = np.array(full_angles_degrees)

        # Calculate metrics
        result = {
            "mean_angle_degrees": float(np.mean(full_angles_degrees)),
            "median_angle_degrees": float(np.median(full_angles_degrees)),
            "min_angle_degrees": float(np.min(full_angles_degrees)),
            "max_angle_degrees": float(np.max(full_angles_degrees)),
            "pct_near_orthogonal": float(((full_angles_degrees >= 80) & (full_angles_degrees <= 100)).mean() * 100),
            "pct_obtuse_angles": float((full_angles_degrees > 90).mean() * 100),
            "pct_acute_angles": float((full_angles_degrees < 90).mean() * 100),
            "is_self_pair": True,
        }

        return result

    def _calculate_between_orthogonality_metrics(
        self, Vh_1: np.ndarray, Vh_2: np.ndarray, dim_1: int, dim_2: int
    ) -> dict:
        """Calculate orthogonality metrics between two different groups of neurons."""
        if Vh_1 is None or Vh_2 is None:
            return {
                "mean_cross_angle_degrees": 0.0,
                "median_cross_angle_degrees": 0.0,
                "min_cross_angle_degrees": 0.0,
                "max_cross_angle_degrees": 0.0,
                "pct_cross_near_orthogonal": 0.0,
            }

        # Make sure we don't exceed available dimensions
        dim_1 = min(dim_1, Vh_1.shape[0])
        dim_2 = min(dim_2, Vh_2.shape[0])

        # Use effective dimensions to define subspaces
        V_1 = Vh_1[:dim_1].T  # Column vectors spanning subspace 1
        V_2 = Vh_2[:dim_2].T  # Column vectors spanning subspace 2

        # Compute principal angles between subspaces (these are always ≤ 90°)
        try:
            principal_angles = subspace_angles(V_1, V_2)
            principal_angles_degrees = np.degrees(principal_angles)
        except Exception as e:
            print(f"Error calculating subspace angles: {e}")
            principal_angles_degrees = np.array([])

        # Calculate full directional angles between all pairs of basis vectors
        full_angles_degrees = []

        for i in range(V_1.shape[1]):
            v1 = V_1[:, i]
            v1_norm = np.linalg.norm(v1)

            for j in range(V_2.shape[1]):
                v2 = V_2[:, j]
                v2_norm = np.linalg.norm(v2)

                # Avoid division by zero
                if v1_norm > 0 and v2_norm > 0:
                    # Calculate dot product
                    dot_product = np.dot(v1, v2) / (v1_norm * v2_norm)
                    # Clamp to avoid numerical errors
                    dot_product = max(min(dot_product, 1.0), -1.0)
                    # Calculate angle in degrees
                    angle_degrees = np.degrees(np.arccos(dot_product))
                    full_angles_degrees.append(angle_degrees)

        if not full_angles_degrees:
            return {
                "mean_cross_angle_degrees": 0.0,
                "median_cross_angle_degrees": 0.0,
                "min_cross_angle_degrees": 0.0,
                "max_cross_angle_degrees": 0.0,
                "pct_cross_near_orthogonal": 0.0,
            }

        full_angles_degrees = np.array(full_angles_degrees)

        # Calculate metrics
        result = {
            # Principal angles metrics (if available)
            "principal_mean_angle_degrees": float(np.mean(principal_angles_degrees))
            if len(principal_angles_degrees) > 0
            else 0.0,
            "principal_median_angle_degrees": float(np.median(principal_angles_degrees))
            if len(principal_angles_degrees) > 0
            else 0.0,
            "principal_min_angle_degrees": float(np.min(principal_angles_degrees))
            if len(principal_angles_degrees) > 0
            else 0.0,
            "principal_max_angle_degrees": float(np.max(principal_angles_degrees))
            if len(principal_angles_degrees) > 0
            else 0.0,
            # Full directional angles
            "mean_cross_angle_degrees": float(np.mean(full_angles_degrees)),
            "median_cross_angle_degrees": float(np.median(full_angles_degrees)),
            "min_cross_angle_degrees": float(np.min(full_angles_degrees)),
            "max_cross_angle_degrees": float(np.max(full_angles_degrees)),
            # Percentage of angles in different ranges
            "pct_cross_near_orthogonal": float(
                ((full_angles_degrees >= 80) & (full_angles_degrees <= 100)).mean() * 100
            ),
            "pct_cross_obtuse_angles": float((full_angles_degrees > 90).mean() * 100),
            "pct_cross_acute_angles": float((full_angles_degrees < 90).mean() * 100),
            # Flag for pair type
            "is_self_pair": False,
        }

        return result

    def analyze_orthogonality(self) -> dict:
        """Analyze orthogonality within and between neuron groups."""
        results = {"within": {}, "between": {}}

        # Within-group orthogonality
        for group_name, result in self.dimensionality_results.items():
            if group_name in self.neuron_indices:  # Skip comparison results like "boost_vs_random_1"
                Vh = result.get("right_singular_vectors")
                effective_dim = result.get("effective_dim")

                if Vh is not None and effective_dim is not None:
                    results["within"][group_name] = self._calculate_orthogonality_metrics(Vh, effective_dim)

        # Between-group orthogonality
        # Define the pairs to analyze
        pairs = [("boost", "random_1"), ("suppress", "random_1"), ("boost", "suppress"), ("random_1", "random_2")]

        for group1, group2 in pairs:
            if (
                group1 in self.dimensionality_results
                and group2 in self.dimensionality_results
                and "right_singular_vectors" in self.dimensionality_results[group1]
                and "right_singular_vectors" in self.dimensionality_results[group2]
            ):
                Vh_1 = self.dimensionality_results[group1]["right_singular_vectors"]
                Vh_2 = self.dimensionality_results[group2]["right_singular_vectors"]
                dim_1 = self.dimensionality_results[group1]["effective_dim"]
                dim_2 = self.dimensionality_results[group2]["effective_dim"]

                results["between"][f"{group1}_vs_{group2}"] = self._calculate_between_orthogonality_metrics(
                    Vh_1, Vh_2, dim_1, dim_2
                )

        # Statistical comparisons
        # Compare within-group metrics
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
        """Filter out large arrays from results dict to make it more manageable."""
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
        """Run all geometric analyses and compile comprehensive results."""
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
