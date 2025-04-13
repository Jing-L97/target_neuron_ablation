import gc
import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class ActivationGeometricAnalyzer:
    """Analyzes the geometric properties of neuron activations with optimized memory usage."""

    def __init__(
        self,
        activation_data: pd.DataFrame,
        special_neuron_indices: list[int],
        activation_column: str = "activation",
        token_column: str = "str_tokens",
        context_column: str = "context",
        component_column: str = "component_name",
        num_random_groups: int = 1,
        device: str | None = None,
        use_mixed_precision: bool = True,
    ):
        """Initialize the analyzer with activation data and special neuron indices."""
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        self.dtype = torch.float16 if self.use_mixed_precision else torch.float32
        logger.info(f"Using device: {self.device}, Mixed precision: {self.use_mixed_precision}")

        self.data = activation_data
        self.special_neuron_indices = special_neuron_indices
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

        # Create activation matrices and convert to PyTorch tensors on the specified device
        self.special_activation_matrix = self._create_activation_matrix(special_neuron_indices)
        self.special_activation_tensor = torch.tensor(self.special_activation_matrix, dtype=self.dtype).to(self.device)

        # Generate random groups but keep them on CPU until needed
        self.random_groups = self._generate_random_groups()
        # Don't preload all random groups to GPU
        self.random_groups_tensors = None

        # Pre-compute reusable masks for upper triangular operations
        n_special_neurons = len(self.special_neuron_indices)
        if n_special_neurons > 0:
            self.triu_mask = torch.triu(
                torch.ones(n_special_neurons, n_special_neurons, device=self.device), diagonal=1
            ).bool()

        # Results storage
        self.dimensionality_results = {}
        self.orthogonality_results = {}
        self.coactivation_results = {}
        self.comparative_results = {}

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

    def _generate_random_groups(self) -> list[np.ndarray]:
        """Generate random neuron groups of the same size as the special group for comparison."""
        group_size = len(self.special_neuron_indices)
        non_special_indices = [idx for idx in self.all_neuron_indices if idx not in self.special_neuron_indices]

        random_groups = []
        for _ in range(self.num_random_groups):
            # If there aren't enough non-special neurons, sample with replacement
            if len(non_special_indices) < group_size:
                random_indices = np.random.choice(self.all_neuron_indices, size=group_size, replace=True)
            else:
                random_indices = np.random.choice(non_special_indices, size=group_size, replace=False)

            random_activation_matrix = self._create_activation_matrix(random_indices)
            random_groups.append(random_activation_matrix)

        return random_groups

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
        """Analyze the dimensionality of the neuron group's activation space.
        Uses GPU acceleration for matrix operations and normalizations.
        """
        # Normalize the activation tensor
        tensor = self.special_activation_tensor
        if tensor.shape[0] > 0 and tensor.shape[1] > 0:
            # Calculate mean and std for each neuron (column)
            means = torch.mean(tensor, dim=0)
            stds = torch.std(tensor, dim=0)
            # Replace zero stds with 1 to avoid division by zero
            stds[stds == 0] = 1.0
            normalized_tensor = (tensor - means) / stds

        # Convert to CPU for PCA (scikit-learn doesn't support GPU)
        normalized_matrix = self._to_numpy(normalized_tensor)

        # Run PCA
        pca = PCA()
        pca.fit(normalized_matrix)

        # Calculate effective dimensionality
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        effective_dim = np.argmax(cumulative_variance >= variance_threshold) + 1

        # Calculate random group dimensionality for comparison
        random_dims = []

        # Process random groups one at a time to save memory
        for i in range(self.num_random_groups):
            # Create tensor on-demand
            random_tensor = torch.tensor(self.random_groups[i], dtype=self.dtype).to(self.device)

            # Normalize
            if random_tensor.shape[0] > 0 and random_tensor.shape[1] > 0:
                r_means = torch.mean(random_tensor, dim=0)
                r_stds = torch.std(random_tensor, dim=0)
                r_stds[r_stds == 0] = 1.0
                r_normalized_tensor = (random_tensor - r_means) / r_stds

                # Convert to CPU for PCA
                r_normalized = self._to_numpy(r_normalized_tensor)

                # Free up GPU memory
                del r_normalized_tensor

                r_pca = PCA()
                r_pca.fit(r_normalized)
                r_cumulative = np.cumsum(r_pca.explained_variance_ratio_)
                r_effective_dim = np.argmax(r_cumulative >= variance_threshold) + 1
                random_dims.append(r_effective_dim)

            # Free GPU memory
            del random_tensor
            torch.cuda.empty_cache()

        # Statistical comparison
        tstat, pvalue, is_significant, comparison = self._safe_ttest(effective_dim, random_dims)

        results = {
            "effective_dim": int(effective_dim),
            "total_dim": len(self.special_neuron_indices),
            "dim_prop": float(effective_dim / len(self.special_neuron_indices)),
            "explained_variance_ratio": explained_variance_ratio,
            "cumulative_variance": cumulative_variance,
            "random_effective_dim": random_dims,
            "random_mean_dim": float(np.mean(random_dims)),
            "ttest_stat": float(tstat),
            "ttest_p": float(pvalue),
            "is_significantly_different": bool(is_significant),
            "comparison": comparison,
        }

        self.dimensionality_results = results

        # Clear some memory
        torch.cuda.empty_cache()

        return results

    def analyze_orthogonality(self) -> dict[str, Any]:
        """Analyze the orthogonality/alignment between neurons in the special group."""
        # Calculate special group orthogonality metrics
        special_metrics = self._calculate_orthogonality_metrics(self.special_activation_matrix)

        # Calculate metrics for random groups
        random_metrics = self._calculate_random_group_orthogonality()

        # Statistical comparison
        comparison_results = self._compare_orthogonality_to_random(special_metrics, random_metrics)

        # Combine all results
        results = {**special_metrics, **random_metrics, **comparison_results}

        self.orthogonality_results = results
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
            "mean_cosine_similarity": mean_similarity,
            "median_cosine_similarity": median_similarity,
            "max_cosine_similarity": max_similarity,
            "min_cosine_similarity": min_similarity,
            "angle_distribution": angle_distribution,
            "mean_angle_degrees": mean_angle,
            "pct_near_orthogonal": near_orthogonal,
        }

    def _calculate_random_group_orthogonality(self) -> dict[str, Any]:
        """Calculate orthogonality metrics for random neuron groups.

        Returns:
            Dictionary of random group metrics

        """
        random_mean_similarities = []
        random_near_orthogonal = []

        for random_matrix in self.random_groups:
            # Calculate metrics for this random group
            metrics = self._calculate_orthogonality_metrics(random_matrix)

            # Only add to our statistics if we got valid measurements
            if len(metrics["angle_distribution"]) > 0:
                random_mean_similarities.append(metrics["mean_cosine_similarity"])
                random_near_orthogonal.append(metrics["pct_near_orthogonal"])

        # If no valid random groups, provide defaults
        if not random_mean_similarities:
            random_mean_similarities = [0.0]
            random_near_orthogonal = [0.0]

        return {"random_mean_similarities": random_mean_similarities, "random_near_orthogonal": random_near_orthogonal}

    def _compare_orthogonality_to_random(
        self, special_metrics: dict[str, Any], random_metrics: dict[str, Any]
    ) -> dict[str, Any]:
        """Compare special group orthogonality metrics to random groups."""
        random_mean_similarities = random_metrics["random_mean_similarities"]
        random_near_orthogonal = random_metrics["random_near_orthogonal"]

        # Skip statistical tests if no angle distribution for special group
        if len(special_metrics["angle_distribution"]) == 0:
            return {
                "similarity_ttest": {"statistic": 0.0, "pvalue": 1.0, "is_significant": False, "comparison": "none"},
                "orthogonality_ttest": {"statistic": 0.0, "pvalue": 1.0, "is_significant": False, "comparison": "none"},
            }

        # Statistical comparison for similarity
        tstat_sim, pvalue_sim = ttest_ind(
            [special_metrics["mean_cosine_similarity"]], random_mean_similarities, equal_var=False
        )

        # Statistical comparison for orthogonality
        tstat_orth, pvalue_orth = ttest_ind(
            [special_metrics["pct_near_orthogonal"]], random_near_orthogonal, equal_var=False
        )

        return {
            "similarity_ttest": {
                "statistic": tstat_sim,
                "pvalue": pvalue_sim,
                "is_significant": pvalue_sim < 0.05,
                "comparison": "higher"
                if special_metrics["mean_cosine_similarity"] > np.mean(random_mean_similarities)
                else "lower",
            },
            "orthogonality_ttest": {
                "statistic": tstat_orth,
                "pvalue": pvalue_orth,
                "is_significant": pvalue_orth < 0.05,
                "comparison": "higher"
                if special_metrics["pct_near_orthogonal"] > np.mean(random_near_orthogonal)
                else "lower",
            },
        }

    def analyze_coactivation(self) -> dict[str, Any]:
        """Analyze coactivation patterns among neurons with hierarchical clustering."""
        # Transpose to get neurons as rows
        neuron_tensor = self.special_activation_tensor.t()

        # Calculate correlation matrix using GPU
        # First normalize
        centered = neuron_tensor - torch.mean(neuron_tensor, dim=1, keepdim=True)
        normalized = centered / (torch.std(neuron_tensor, dim=1, keepdim=True) + 1e-8)

        # Compute correlation matrix
        n_samples = normalized.shape[1]
        correlation_matrix_tensor = torch.mm(normalized, normalized.t()) / n_samples

        # Move correlation matrix to CPU for scipy operations and free GPU memory
        correlation_matrix = self._to_numpy(correlation_matrix_tensor)
        del correlation_matrix_tensor, normalized, centered
        torch.cuda.empty_cache()

        # Convert correlations to distances
        distance_matrix = 1 - np.abs(correlation_matrix)

        # Perform hierarchical clustering (CPU operation)
        Z = linkage(distance_matrix[np.triu_indices(len(distance_matrix), k=1)], method="ward")

        # Get clusters at different thresholds
        clusters_tight = fcluster(Z, t=3, criterion="distance")
        clusters_medium = fcluster(Z, t=5, criterion="distance")
        clusters_loose = fcluster(Z, t=7, criterion="distance")

        # Calculate metrics
        n_clusters_tight = len(np.unique(clusters_tight))
        n_clusters_medium = len(np.unique(clusters_medium))
        n_clusters_loose = len(np.unique(clusters_loose))

        # Extract upper triangle for distribution analysis
        upper_indices = np.triu_indices_from(correlation_matrix, k=1)
        correlation_distribution = correlation_matrix[upper_indices]

        # Calculate correlation statistics
        mean_correlation = np.mean(correlation_distribution)
        median_correlation = np.median(correlation_distribution)
        positive_correlation = np.mean(correlation_distribution > 0) * 100
        strong_correlation = np.mean(np.abs(correlation_distribution) > 0.5) * 100

        # Do the same for random groups for comparison
        random_n_clusters_medium = []
        random_mean_correlations = []

        # Process random groups sequentially to save memory
        for i in range(self.num_random_groups):
            random_tensor = torch.tensor(self.random_groups[i], dtype=self.dtype).to(self.device)
            r_neuron_tensor = random_tensor.t()

            # Calculate correlation on GPU
            r_centered = r_neuron_tensor - torch.mean(r_neuron_tensor, dim=1, keepdim=True)
            r_normalized = r_centered / (torch.std(r_neuron_tensor, dim=1, keepdim=True) + 1e-8)
            r_corr_tensor = torch.mm(r_normalized, r_normalized.t()) / n_samples

            # Move to CPU for further processing
            r_corr = self._to_numpy(r_corr_tensor)
            r_dist = 1 - np.abs(r_corr)

            # Hierarchical clustering
            r_Z = linkage(r_dist[np.triu_indices(len(r_dist), k=1)], method="ward")
            r_clusters = fcluster(r_Z, t=5, criterion="distance")
            random_n_clusters_medium.append(len(np.unique(r_clusters)))

            # Correlation statistics
            r_upper = r_corr[np.triu_indices_from(r_corr, k=1)]
            random_mean_correlations.append(np.mean(r_upper))

            # Free GPU memory
            del random_tensor, r_neuron_tensor, r_normalized, r_centered, r_corr_tensor
            torch.cuda.empty_cache()

        # Statistical comparison for clustering
        tstat_clust, pvalue_clust, is_significant_clust, comparison_clust = self._safe_ttest(
            n_clusters_medium, random_n_clusters_medium
        )

        # Statistical comparison for correlation
        tstat_corr, pvalue_corr, is_significant_corr, comparison_corr = self._safe_ttest(
            mean_correlation, random_mean_correlations
        )

        results = {
            "correlation_matrix": correlation_matrix,
            "mean_correlation": float(mean_correlation),
            "median_correlation": float(median_correlation),
            "pct_positive_correlation": float(positive_correlation),
            "pct_strong_correlation": float(strong_correlation),
            "correlation_distribution": correlation_distribution,
            "linkage": Z,
            "clusters_tight": clusters_tight,
            "clusters_medium": clusters_medium,
            "clusters_loose": clusters_loose,
            "n_clusters_tight": int(n_clusters_tight),
            "n_clusters_medium": int(n_clusters_medium),
            "n_clusters_loose": int(n_clusters_loose),
            "random_n_clusters_medium": random_n_clusters_medium,
            "random_mean_correlations": random_mean_correlations,
            "clustering_ttest": {
                "statistic": float(tstat_clust),
                "pvalue": float(pvalue_clust),
                "is_significant": bool(is_significant_clust),
                "comparison": comparison_clust,
            },
            "correlation_ttest": {
                "statistic": float(tstat_corr),
                "pvalue": float(pvalue_corr),
                "is_significant": bool(is_significant_corr),
                "comparison": comparison_corr,
            },
        }

        self.coactivation_results = results

        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()

        return results

    def run_all_analyses(self) -> dict[str, dict[str, Any]]:
        """Run all geometric analyses at once."""
        self.analyze_dimensionality()
        self.analyze_orthogonality()
        self.analyze_coactivation()

        # Compile summary findings
        summary = {}

        # Dimensionality findings
        if self.dimensionality_results["is_significantly_different"]:
            if self.dimensionality_results["comparison"] == "lower":
                summary["dimensionality"] = (
                    "The neuron group has significantly lower dimensionality than random groups, "
                    "suggesting coordinated/synergistic activity."
                )
            else:
                summary["dimensionality"] = (
                    "The neuron group has significantly higher dimensionality than random groups, "
                    "suggesting diverse and independent response patterns."
                )
        else:
            summary["dimensionality"] = (
                "The neuron group's dimensionality is not significantly different from random groups."
            )

        # Orthogonality findings
        if self.orthogonality_results["similarity_ttest"]["is_significant"]:
            if self.orthogonality_results["similarity_ttest"]["comparison"] == "higher":
                summary["orthogonality"] = (
                    "The neuron group shows significantly higher alignment than random groups, "
                    "suggesting coordinated functionality."
                )
            else:
                summary["orthogonality"] = (
                    "The neuron group shows significantly lower alignment than random groups, "
                    "suggesting more orthogonal/independent response patterns."
                )
        else:
            summary["orthogonality"] = "The neuron group's alignment is not significantly different from random groups."

        # Coactivation findings
        if self.coactivation_results["correlation_ttest"]["is_significant"]:
            if self.coactivation_results["correlation_ttest"]["comparison"] == "higher":
                summary["coactivation"] = (
                    "The neuron group shows significantly stronger coactivation patterns than random groups, "
                    "suggesting synergistic behavior."
                )
            else:
                summary["coactivation"] = (
                    "The neuron group shows significantly weaker coactivation patterns than random groups, "
                    "suggesting more independent behavior."
                )
        else:
            summary["coactivation"] = (
                "The neuron group's coactivation patterns are not significantly different from random groups."
            )

        # Add clustering findings
        if self.coactivation_results["clustering_ttest"]["is_significant"]:
            if self.coactivation_results["clustering_ttest"]["comparison"] == "fewer clusters":
                summary["clustering"] = (
                    "The neuron group forms significantly fewer functional clusters than random groups, "
                    "suggesting more coordinated organization."
                )
            else:
                summary["clustering"] = (
                    "The neuron group forms significantly more functional clusters than random groups, "
                    "suggesting more specialized subgroups."
                )
        else:
            summary["clustering"] = (
                "The neuron group's clustering structure is not significantly different from random groups."
            )

        # Create combined results without including large matrices
        combined_results = {
            "dimensionality": {
                k: v
                for k, v in self.dimensionality_results.items()
                if not (isinstance(v, np.ndarray) and v.size > 1000)
            },
            "orthogonality": {
                k: v for k, v in self.orthogonality_results.items() if not (isinstance(v, np.ndarray) and v.size > 1000)
            },
            "coactivation": {
                k: v for k, v in self.coactivation_results.items() if not (isinstance(v, np.ndarray) and v.size > 1000)
            },
            "special_neuron_indices": self.special_neuron_indices,
            "summary": summary,
        }

        self.comparative_results = combined_results

        # Final memory cleanup
        gc.collect()
        torch.cuda.empty_cache()

        return combined_results
