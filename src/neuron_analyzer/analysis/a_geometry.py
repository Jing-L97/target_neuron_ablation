import logging
import typing as t
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA

from neuron_analyzer.load_util import cleanup
from neuron_analyzer.selection.neuron import generate_random_indices

logger = logging.getLogger(__name__)


T = t.TypeVar("T")
ComputationType = t.Literal["within", "between"]
GroupType = t.Literal["boost", "suppress", "random_1", "random_2"]


class ActivationGeometricAnalyzer:
    """Analyzes the geometric properties of neuron activations with optimized memory usage."""

    def __init__(
        self,
        activation_data: pd.DataFrame,
        boost_neuron_indices: list[int],
        suppress_neuron_indices: list[int],
        excluded_neuron_indices: list[int],
        activation_column: str = "activation",
        token_column: str = "str_tokens",
        context_column: str = "context",
        component_column: str = "component_name",
        num_random_groups: int = 2,
        device: str | None = None,
        use_mixed_precision: bool = True,
    ):
        """Initialize the analyzer with activation data and neuron groups."""
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        self.dtype = torch.float16 if self.use_mixed_precision else torch.float32
        logger.info(f"Using device: {self.device}, Mixed precision: {self.use_mixed_precision}")

        self.data = activation_data
        self.boost_neuron_indices = boost_neuron_indices
        self.suppress_neuron_indices = suppress_neuron_indices
        self.excluded_neuron_indices = excluded_neuron_indices
        self.activation_column = activation_column
        self.token_column = token_column
        self.context_column = context_column
        self.component_column = component_column
        self.num_random_groups = num_random_groups

        # Create a unique token-context identifier
        self.data["token_context_id"] = (
            self.data[token_column].astype(str) + "_" + self.data[context_column].astype(str)
        )

        # Extract unique token-context pairs and components
        self.token_contexts = self.data["token_context_id"].unique()
        self.all_neuron_indices = self.data[self.component_column].astype(int).unique()

        # Generate non-overlapping random groups
        self.random_groups, self.random_indices = self._generate_random_groups()

        # Create activation matrices for each group
        self.boost_activation_matrix = self._create_activation_matrix(boost_neuron_indices)
        self.suppress_activation_matrix = self._create_activation_matrix(suppress_neuron_indices)
        self.random_1_activation_matrix = self._create_activation_matrix(self.random_indices[0])
        self.random_2_activation_matrix = self._create_activation_matrix(self.random_indices[1])

        # Convert matrices to PyTorch tensors
        self.boost_activation_tensor = torch.tensor(self.boost_activation_matrix, dtype=self.dtype).to(self.device)
        self.suppress_activation_tensor = torch.tensor(self.suppress_activation_matrix, dtype=self.dtype).to(
            self.device
        )
        self.random_1_activation_tensor = torch.tensor(self.random_1_activation_matrix, dtype=self.dtype).to(
            self.device
        )
        self.random_2_activation_tensor = torch.tensor(self.random_2_activation_matrix, dtype=self.dtype).to(
            self.device
        )

        # Results storage
        self.dimensionality_results = {}
        self.orthogonality_results = {}
        self.coactivation_results = {}
        self.comparative_results = {}

        # Store activation matrices in a dictionary for easy access
        self.activation_matrices = {
            "boost": self.boost_activation_matrix,
            "suppress": self.suppress_activation_matrix,
            "random_1": self.random_1_activation_matrix,
            "random_2": self.random_2_activation_matrix,
        }

        # Store activation tensors in a dictionary for easy access
        self.activation_tensors = {
            "boost": self.boost_activation_tensor,
            "suppress": self.suppress_activation_tensor,
            "random_1": self.random_1_activation_tensor,
            "random_2": self.random_2_activation_tensor,
        }

        # Store neuron indices in a dictionary for easy access
        self.neuron_indices = {
            "boost": self.boost_neuron_indices,
            "suppress": self.suppress_neuron_indices,
            "random_1": self.random_indices[0],
            "random_2": self.random_indices[1],
        }

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert a PyTorch tensor to NumPy array."""
        return tensor.detach().cpu().numpy()

    def _create_activation_matrix(self, neuron_indices: list[int]) -> np.ndarray:
        """Create an activation matrix where rows are token-context pairs and columns are neurons."""
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
        return pivot_table.values

    def get_token_groups(self) -> dict[str, list[str]]:
        """Group token-context IDs by their token value."""
        token_groups = {}
        for token_context in self.token_contexts:
            token = token_context.split("_")[0]
            if token not in token_groups:
                token_groups[token] = []
            token_groups[token].append(token_context)
        return token_groups

    def get_context_groups(self) -> dict[str, list[str]]:
        """Group token-context IDs by their context value."""
        context_groups = {}
        for token_context in self.token_contexts:
            context = token_context.split("_")[1]
            if context not in context_groups:
                context_groups[context] = []
            context_groups[context].append(token_context)
        return context_groups

    def _generate_random_groups(self) -> tuple[list[list[int]], list[list[int]]]:
        """Generate non-overlapping random neuron groups that don't overlap with boost or suppress neurons."""
        group_size = max(len(self.boost_neuron_indices), len(self.suppress_neuron_indices))
        special_indices = set(self.boost_neuron_indices + self.suppress_neuron_indices + self.excluded_neuron_indices)

        random_indices = generate_random_indices(
            self.all_neuron_indices, special_indices, group_size, self.num_random_groups
        )
        # Create activation matrices for the random groups
        random_groups = [self._create_activation_matrix(indices) for indices in random_indices]

        return random_groups, random_indices

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
            logger.warning(f"Error performing t-test: {e}")
            return 0.0, 1.0, False, "unknown"

    def analyze_dimensionality(self, variance_threshold: float = 0.95) -> dict[str, Any]:
        """Analyze the dimensionality of each neuron group's activation space."""
        results = {}

        # Analyze each group
        for group_name, tensor in self.activation_tensors.items():
            if tensor.shape[0] > 0 and tensor.shape[1] > 0:
                # Normalize the activation tensor
                means = torch.mean(tensor, dim=0)
                stds = torch.std(tensor, dim=0)
                # Replace zero stds with 1 to avoid division by zero
                stds[stds == 0] = 1.0
                normalized_tensor = (tensor - means) / stds

                # Convert to CPU for PCA
                normalized_matrix = self._to_numpy(normalized_tensor)

                # Run PCA
                pca = PCA()
                pca.fit(normalized_matrix)

                # Calculate effective dimensionality
                explained_variance_ratio = pca.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance_ratio)
                effective_dim = np.argmax(cumulative_variance >= variance_threshold) + 1

                group_results = {
                    "effective_dim": int(effective_dim),
                    "total_dim": len(self.neuron_indices[group_name]),
                    "dim_prop": float(effective_dim / len(self.neuron_indices[group_name])),
                    "explained_variance_ratio": explained_variance_ratio,
                    "cumulative_variance": cumulative_variance,
                }

                results[group_name] = group_results

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

        self.dimensionality_results = results

        # Clear some memory
        cleanup()

        return results

    def _calculate_orthogonality_metrics(self, activation_matrix: np.ndarray) -> dict[str, Any]:
        """Calculate orthogonality metrics for a given activation matrix."""
        # Transpose to get neuron x token-context matrix for analyzing neuron relationships
        neuron_matrix = activation_matrix.T

        # Skip calculation if not enough neurons or contexts
        if neuron_matrix.shape[0] <= 1 or neuron_matrix.shape[1] == 0:
            return {
                "cosine_similarity_matrix": np.array([[1.0]]),
                "mean_cosine_similarity": 0.0,
                "median_cosine_similarity": 0.0,
                "max_cosine_similarity": 0.0,
                "min_cosine_similarity": 0.0,
                "angle_distribution": np.array([]),
                "mean_angle_degrees": 0.0,
                "pct_near_orthogonal": 0.0,
            }

        # Normalize neuron vectors
        normalized_neurons = neuron_matrix.copy()
        norms = np.linalg.norm(normalized_neurons, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        normalized_neurons = normalized_neurons / norms

        # Compute cosine similarity matrix (inner products of normalized vectors)
        cosine_similarity = normalized_neurons @ normalized_neurons.T

        # Extract upper triangle (excluding diagonal) for distribution analysis
        upper_indices = np.triu_indices_from(cosine_similarity, k=1)

        # Handle case with only one neuron (no upper triangle)
        if len(upper_indices[0]) == 0:
            return {
                "cosine_similarity_matrix": cosine_similarity,
                "mean_cosine_similarity": 0.0,
                "median_cosine_similarity": 0.0,
                "max_cosine_similarity": 0.0,
                "min_cosine_similarity": 0.0,
                "angle_distribution": np.array([]),
                "mean_angle_degrees": 0.0,
                "pct_near_orthogonal": 0.0,
            }

        similarity_distribution = cosine_similarity[upper_indices]

        # Calculate statistics
        mean_similarity = np.mean(similarity_distribution)
        median_similarity = np.median(similarity_distribution)
        max_similarity = np.max(similarity_distribution)
        min_similarity = np.min(similarity_distribution)

        # Calculate angle distribution (in degrees)
        angle_distribution = np.degrees(np.arccos(np.clip(similarity_distribution, -1.0, 1.0)))
        mean_angle = np.mean(angle_distribution)

        # Calculate orthogonality measures
        near_orthogonal = np.mean((angle_distribution >= 80) & (angle_distribution <= 100)) * 100

        return {
            "cosine_similarity_matrix": cosine_similarity,
            "mean_cosine_similarity": float(mean_similarity),
            "median_cosine_similarity": float(median_similarity),
            "max_cosine_similarity": float(max_similarity),
            "min_cosine_similarity": float(min_similarity),
            "angle_distribution": angle_distribution,
            "mean_angle_degrees": float(mean_angle),
            "pct_near_orthogonal": float(near_orthogonal),
        }

    def _calculate_between_orthogonality_metrics(
        self, activation_matrix1: np.ndarray, activation_matrix2: np.ndarray
    ) -> dict[str, Any]:
        """Calculate orthogonality metrics between two different groups of neurons."""
        # Transpose to get neuron x token-context matrices
        neuron_matrix1 = activation_matrix1.T
        neuron_matrix2 = activation_matrix2.T

        # Skip calculation if not enough neurons or contexts
        if (
            neuron_matrix1.shape[0] == 0
            or neuron_matrix1.shape[1] == 0
            or neuron_matrix2.shape[0] == 0
            or neuron_matrix2.shape[1] == 0
        ):
            return {
                "cross_cosine_similarity_matrix": np.array([[0.0]]),
                "mean_cross_cosine_similarity": 0.0,
                "median_cross_cosine_similarity": 0.0,
                "max_cross_cosine_similarity": 0.0,
                "min_cross_cosine_similarity": 0.0,
                "cross_angle_distribution": np.array([]),
                "mean_cross_angle_degrees": 0.0,
                "pct_cross_near_orthogonal": 0.0,
            }

        # Normalize neuron vectors
        normalized_neurons1 = neuron_matrix1.copy()
        normalized_neurons2 = neuron_matrix2.copy()

        norms1 = np.linalg.norm(normalized_neurons1, axis=1, keepdims=True)
        norms2 = np.linalg.norm(normalized_neurons2, axis=1, keepdims=True)

        norms1[norms1 == 0] = 1.0  # Avoid division by zero
        norms2[norms2 == 0] = 1.0  # Avoid division by zero

        normalized_neurons1 = normalized_neurons1 / norms1
        normalized_neurons2 = normalized_neurons2 / norms2

        # Compute cross-cosine similarity matrix
        cross_cosine_similarity = normalized_neurons1 @ normalized_neurons2.T

        # Flatten the matrix for distribution analysis
        similarity_distribution = cross_cosine_similarity.flatten()

        # Calculate statistics
        mean_similarity = np.mean(similarity_distribution)
        median_similarity = np.median(similarity_distribution)
        max_similarity = np.max(similarity_distribution)
        min_similarity = np.min(similarity_distribution)

        # Calculate angle distribution (in degrees)
        angle_distribution = np.degrees(np.arccos(np.clip(similarity_distribution, -1.0, 1.0)))
        mean_angle = np.mean(angle_distribution)

        # Calculate orthogonality measures
        near_orthogonal = np.mean((angle_distribution >= 80) & (angle_distribution <= 100)) * 100

        return {
            "cross_cosine_similarity_matrix": cross_cosine_similarity,
            "mean_cross_cosine_similarity": float(mean_similarity),
            "median_cross_cosine_similarity": float(median_similarity),
            "max_cross_cosine_similarity": float(max_similarity),
            "min_cross_cosine_similarity": float(min_similarity),
            "cross_angle_distribution": angle_distribution,
            "mean_cross_angle_degrees": float(mean_angle),
            "pct_cross_near_orthogonal": float(near_orthogonal),
        }

    def analyze_orthogonality(self) -> dict[str, dict[str, Any]]:
        """Analyze orthogonality for various conditions:
        - Within the boost neurons
        - Within the suppress neurons
        - Within each random group
        - Between boost and random groups
        - Between suppress and random groups
        - Between the random groups
        """
        results = {"within": {}, "between": {}}

        # Within-group orthogonality
        for group_name, matrix in self.activation_matrices.items():
            results["within"][group_name] = self._calculate_orthogonality_metrics(matrix)

        # Between-group orthogonality
        # Define the pairs to analyze
        pairs = [("boost", "random_1"), ("suppress", "random_1"), ("boost", "suppress"), ("random_1", "random_2")]

        for group1, group2 in pairs:
            matrix1 = self.activation_matrices[group1]
            matrix2 = self.activation_matrices[group2]
            results["between"][f"{group1}_vs_{group2}"] = self._calculate_between_orthogonality_metrics(
                matrix1, matrix2
            )

        # Statistical comparisons
        # Compare within-group metrics
        comparisons = [("boost", "random_1"), ("suppress", "random_1"), ("boost", "suppress"), ("random_1", "random_2")]

        statistical_results = {}
        for group1, group2 in comparisons:
            # Compare mean cosine similarity
            tstat, pvalue, is_significant, comparison = self._safe_ttest(
                results["within"][group1]["mean_cosine_similarity"],
                [results["within"][group2]["mean_cosine_similarity"]],
            )

            statistical_results[f"{group1}_vs_{group2}_cosine"] = {
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

    def analyze_coactivation(self) -> dict[str, Any]:
        """Analyze coactivation patterns within and between neuron groups using hierarchical clustering."""
        results = {"within": {}, "between": {}}

        # Within-group coactivation
        for group_name, tensor in self.activation_tensors.items():
            # Transpose to get neurons as rows
            neuron_tensor = tensor.t()

            # Skip if not enough data
            if neuron_tensor.shape[0] <= 1 or neuron_tensor.shape[1] == 0:
                results["within"][group_name] = {
                    "correlation_matrix": np.array([[1.0]]),
                    "mean_correlation": 0.0,
                    "median_correlation": 0.0,
                    "pct_positive_correlation": 0.0,
                    "pct_strong_correlation": 0.0,
                    "n_clusters_medium": 1,
                }
                continue

            # Calculate correlation matrix using GPU
            # First normalize
            centered = neuron_tensor - torch.mean(neuron_tensor, dim=1, keepdim=True)
            std_values = torch.std(neuron_tensor, dim=1, keepdim=True)
            # Avoid division by zero
            std_values[std_values < 1e-8] = 1.0
            normalized = centered / std_values

            # Compute correlation matrix
            n_samples = normalized.shape[1]
            correlation_matrix_tensor = torch.mm(normalized, normalized.t()) / n_samples

            # Move correlation matrix to CPU for scipy operations
            correlation_matrix = self._to_numpy(correlation_matrix_tensor)

            # Free GPU memory
            del correlation_matrix_tensor, normalized, centered
            torch.cuda.empty_cache()

            # Fix potential numerical issues (values slightly outside [-1, 1])
            correlation_matrix = np.clip(correlation_matrix, -1.0, 1.0)

            # Convert correlations to distances for clustering
            # Use absolute correlation as similarity measure (1 - |corr| as distance)
            distance_matrix = 1 - np.abs(correlation_matrix)

            # Handle single neuron case
            if len(distance_matrix) <= 1:
                results["within"][group_name] = {
                    "correlation_matrix": correlation_matrix,
                    "mean_correlation": 0.0,
                    "median_correlation": 0.0,
                    "pct_positive_correlation": 0.0,
                    "pct_strong_correlation": 0.0,
                    "n_clusters_medium": 1,
                }
                continue

            # Perform hierarchical clustering
            try:
                # Convert distance matrix to condensed form for linkage
                condensed_distances = pdist(1 - np.abs(correlation_matrix))
                Z = linkage(condensed_distances, method="ward")

                # Get clusters at different thresholds
                clusters_medium = fcluster(Z, t=5, criterion="distance")
                n_clusters_medium = len(np.unique(clusters_medium))
            except Exception as e:
                logger.warning(f"Clustering error for {group_name}: {e}")
                n_clusters_medium = 1

            # Extract upper triangle for distribution analysis
            upper_indices = np.triu_indices_from(correlation_matrix, k=1)
            if len(upper_indices[0]) > 0:  # Check if we have any upper triangle elements
                correlation_distribution = correlation_matrix[upper_indices]

                # Calculate correlation statistics
                mean_correlation = float(np.mean(correlation_distribution))
                median_correlation = float(np.median(correlation_distribution))
                positive_correlation = float(np.mean(correlation_distribution > 0) * 100)
                strong_correlation = float(np.mean(np.abs(correlation_distribution) > 0.5) * 100)
            else:
                mean_correlation = 0.0
                median_correlation = 0.0
                positive_correlation = 0.0
                strong_correlation = 0.0
                correlation_distribution = np.array([])

            results["within"][group_name] = {
                "correlation_matrix": correlation_matrix,
                "mean_correlation": mean_correlation,
                "median_correlation": median_correlation,
                "pct_positive_correlation": positive_correlation,
                "pct_strong_correlation": strong_correlation,
                "n_clusters_medium": n_clusters_medium,
            }

        # Between-group coactivation
        pairs = [("boost", "random_1"), ("suppress", "random_1"), ("boost", "suppress"), ("random_1", "random_2")]

        for group1, group2 in pairs:
            # Get tensors
            tensor1 = self.activation_tensors[group1]
            tensor2 = self.activation_tensors[group2]

            # Transpose to get neurons as rows
            neuron_tensor1 = tensor1.t()
            neuron_tensor2 = tensor2.t()

            # Skip if not enough data
            if (
                neuron_tensor1.shape[0] == 0
                or neuron_tensor1.shape[1] == 0
                or neuron_tensor2.shape[0] == 0
                or neuron_tensor2.shape[1] == 0
            ):
                results["between"][f"{group1}_vs_{group2}"] = {
                    "cross_correlation_matrix": np.array([[0.0]]),
                    "mean_cross_correlation": 0.0,
                    "median_cross_correlation": 0.0,
                    "pct_positive_cross_correlation": 0.0,
                    "pct_strong_cross_correlation": 0.0,
                }
                continue

            # Calculate normalized tensors
            centered1 = neuron_tensor1 - torch.mean(neuron_tensor1, dim=1, keepdim=True)
            centered2 = neuron_tensor2 - torch.mean(neuron_tensor2, dim=1, keepdim=True)

            std1 = torch.std(neuron_tensor1, dim=1, keepdim=True)
            std2 = torch.std(neuron_tensor2, dim=1, keepdim=True)

            # Avoid division by zero
            std1[std1 < 1e-8] = 1.0
            std2[std2 < 1e-8] = 1.0

            normalized1 = centered1 / std1
            normalized2 = centered2 / std2

            # Compute cross-correlation matrix
            n_samples = normalized1.shape[1]
            cross_correlation_tensor = torch.mm(normalized1, normalized2.t()) / n_samples

            # Move to CPU
            cross_correlation = self._to_numpy(cross_correlation_tensor)

            # Free GPU memory
            del cross_correlation_tensor, normalized1, normalized2, centered1, centered2
            torch.cuda.empty_cache()

            # Fix potential numerical issues
            cross_correlation = np.clip(cross_correlation, -1.0, 1.0)

            # Calculate correlation statistics
            mean_correlation = float(np.mean(cross_correlation))
            median_correlation = float(np.median(cross_correlation))
            positive_correlation = float(np.mean(cross_correlation > 0) * 100)
            strong_correlation = float(np.mean(np.abs(cross_correlation) > 0.5) * 100)

            results["between"][f"{group1}_vs_{group2}"] = {
                "cross_correlation_matrix": cross_correlation,
                "mean_cross_correlation": mean_correlation,
                "median_cross_correlation": median_correlation,
                "pct_positive_cross_correlation": positive_correlation,
                "pct_strong_cross_correlation": strong_correlation,
            }

            # Statistical tests comparing coactivation metrics
        statistical_results = {}
        comparisons = [("boost", "random_1"), ("suppress", "random_1"), ("boost", "suppress"), ("random_1", "random_2")]

        for group1, group2 in comparisons:
            # Compare mean correlation values
            tstat, pvalue, is_significant, comparison = self._safe_ttest(
                results["within"][group1]["mean_correlation"], [results["within"][group2]["mean_correlation"]]
            )

            statistical_results[f"{group1}_vs_{group2}_correlation"] = {
                "ttest_stat": float(tstat),
                "ttest_p": float(pvalue),
                "is_significantly_different": bool(is_significant),
                "comparison": comparison,
            }

            # Compare number of clusters
            tstat, pvalue, is_significant, comparison = self._safe_ttest(
                results["within"][group1]["n_clusters_medium"], [results["within"][group2]["n_clusters_medium"]]
            )

            statistical_results[f"{group1}_vs_{group2}_clusters"] = {
                "ttest_stat": float(tstat),
                "ttest_p": float(pvalue),
                "is_significantly_different": bool(is_significant),
                "comparison": comparison,
            }

        results["statistical_tests"] = statistical_results
        self.coactivation_results = results

        # Final memory cleanup
        cleanup()

        return results

    def run_all_analyses(self) -> dict[str, dict[str, Any]]:
        """Run all geometric analyses at once and compile comprehensive results."""
        dimensionality_results = self.analyze_dimensionality()
        orthogonality_results = self.analyze_orthogonality()
        coactivation_results = self.analyze_coactivation()

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
                    "suggesting coordinated/synergistic activity."
                )
            else:
                summary["boost_dimensionality"] = (
                    "The boost neuron group has significantly higher dimensionality than random groups, "
                    "suggesting diverse and independent response patterns."
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
                    "suggesting coordinated/synergistic activity."
                )
            else:
                summary["suppress_dimensionality"] = (
                    "The suppress neuron group has significantly higher dimensionality than random groups, "
                    "suggesting diverse and independent response patterns."
                )
        else:
            summary["suppress_dimensionality"] = (
                "The suppress neuron group's dimensionality is not significantly different from random groups."
            )

        # Orthogonality findings
        for group in ["boost", "suppress"]:
            if (
                f"{group}_vs_random_1_cosine" in orthogonality_results["statistical_tests"]
                and orthogonality_results["statistical_tests"][f"{group}_vs_random_1_cosine"][
                    "is_significantly_different"
                ]
            ):
                if orthogonality_results["statistical_tests"][f"{group}_vs_random_1_cosine"]["comparison"] == "higher":
                    summary[f"{group}_orthogonality"] = (
                        f"The {group} neuron group shows significantly higher alignment than random groups, "
                        "suggesting coordinated functionality."
                    )
                else:
                    summary[f"{group}_orthogonality"] = (
                        f"The {group} neuron group shows significantly lower alignment than random groups, "
                        "suggesting more orthogonal/independent response patterns."
                    )
            else:
                summary[f"{group}_orthogonality"] = (
                    f"The {group} neuron group's alignment is not significantly different from random groups."
                )

        # Coactivation findings
        for group in ["boost", "suppress"]:
            if (
                f"{group}_vs_random_1_correlation" in coactivation_results["statistical_tests"]
                and coactivation_results["statistical_tests"][f"{group}_vs_random_1_correlation"][
                    "is_significantly_different"
                ]
            ):
                if (
                    coactivation_results["statistical_tests"][f"{group}_vs_random_1_correlation"]["comparison"]
                    == "higher"
                ):
                    summary[f"{group}_coactivation"] = (
                        f"The {group} neuron group shows significantly stronger coactivation patterns than random groups, "
                        "suggesting synergistic behavior."
                    )
                else:
                    summary[f"{group}_coactivation"] = (
                        f"The {group} neuron group shows significantly weaker coactivation patterns than random groups, "
                        "suggesting more independent behavior."
                    )
            else:
                summary[f"{group}_coactivation"] = (
                    f"The {group} neuron group's coactivation patterns are not significantly different from random groups."
                )

        # Between boost and suppress
        if (
            "boost_vs_suppress_correlation" in coactivation_results["statistical_tests"]
            and coactivation_results["statistical_tests"]["boost_vs_suppress_correlation"]["is_significantly_different"]
        ):
            if coactivation_results["statistical_tests"]["boost_vs_suppress_correlation"]["comparison"] == "higher":
                summary["boost_vs_suppress_coactivation"] = (
                    "Boost neurons show significantly stronger coactivation patterns than suppress neurons."
                )
            else:
                summary["boost_vs_suppress_coactivation"] = (
                    "Boost neurons show significantly weaker coactivation patterns than suppress neurons."
                )
        else:
            summary["boost_vs_suppress_coactivation"] = (
                "Boost and suppress neurons' coactivation patterns are not significantly different."
            )

        # Create combined results
        combined_results = {
            "dimensionality": self._filter_large_arrays(dimensionality_results),
            "orthogonality": self._filter_large_arrays(orthogonality_results),
            "coactivation": self._filter_large_arrays(coactivation_results),
            "summary": summary,
            "neuron_indices": self.neuron_indices,
        }

        self.comparative_results = combined_results

        # Final memory cleanup
        cleanup()

        return combined_results

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
