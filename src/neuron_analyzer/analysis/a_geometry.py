from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA


class ActivationGeometricAnalyzer:
    """Analyzes the geometric properties of neuron activations to determine if special neuron groups"""

    def __init__(
        self,
        activation_data: pd.DataFrame,
        special_neuron_indices: list[int],
        activation_column: str = "activation",
        context_column: str = "context",
        component_column: str = "component_name",
        num_random_groups: int = 1,
    ):
        """Initialize the analyzer with activation data and special neuron indices."""
        self.data = activation_data
        self.special_neuron_indices = special_neuron_indices
        self.activation_column = activation_column
        self.context_column = context_column
        self.component_column = component_column
        self.num_random_groups = num_random_groups

        # Extract unique contexts and components
        self.contexts = self.data[self.context_column].unique()
        self.all_neuron_indices = self.data[self.component_column].unique()

        # Check if special neuron indices are valid
        if not set(special_neuron_indices).issubset(set(self.all_neuron_indices)):
            raise ValueError("Some special neuron indices are not present in the data")

        # Create activation matrices
        self.special_activation_matrix = self._create_activation_matrix(special_neuron_indices)

        # Generate random groups for comparison
        self.random_groups = self._generate_random_groups()

        # Results storage
        self.dimensionality_results = {}
        self.orthogonality_results = {}
        self.coactivation_results = {}
        self.comparative_results = {}

    def _create_activation_matrix(self, neuron_indices: list[int]) -> np.ndarray:
        """Create an activation matrix where rows are contexts and columns are neurons."""
        # Filter data to only include specified neurons
        filtered_data = self.data[self.data[self.component_column].isin(neuron_indices)]

        # Pivot to create matrix with contexts as rows and neurons as columns
        pivot_table = filtered_data.pivot_table(
            index=self.context_column,
            columns=self.component_column,
            values=self.activation_column,
            aggfunc="first",  # In case of duplicates, take the first value
        )

        # Handle missing values if any
        pivot_table = pivot_table.fillna(0)

        return pivot_table.values

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

    def analyze_dimensionality(self, variance_threshold: float = 0.95) -> dict[str, Any]:
        """Analyze the dimensionality of the neuron group's activation space."""
        # Normalize the activation matrix
        normalized_matrix = self.special_activation_matrix
        if normalized_matrix.shape[0] > 0 and normalized_matrix.shape[1] > 0:
            # Calculate mean and std for each neuron (column)
            means = np.mean(normalized_matrix, axis=0)
            stds = np.std(normalized_matrix, axis=0)
            # Replace zero stds with 1 to avoid division by zero
            stds[stds == 0] = 1.0
            normalized_matrix = (normalized_matrix - means) / stds

        # Run PCA
        pca = PCA()
        pca.fit(normalized_matrix)

        # Calculate effective dimensionality
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        effective_dim = np.argmax(cumulative_variance >= variance_threshold) + 1

        # Calculate random group dimensionality for comparison
        random_dims = []
        for random_matrix in self.random_groups:
            # Normalize
            r_normalized = random_matrix
            if r_normalized.shape[0] > 0 and r_normalized.shape[1] > 0:
                r_means = np.mean(r_normalized, axis=0)
                r_stds = np.std(r_normalized, axis=0)
                r_stds[r_stds == 0] = 1.0
                r_normalized = (r_normalized - r_means) / r_stds

            r_pca = PCA()
            r_pca.fit(r_normalized)
            r_cumulative = np.cumsum(r_pca.explained_variance_ratio_)
            r_effective_dim = np.argmax(r_cumulative >= variance_threshold) + 1
            random_dims.append(r_effective_dim)

        # Statistical comparison
        tstat, pvalue = ttest_ind([effective_dim], random_dims, equal_var=False)

        results = {
            "effective_dim": effective_dim,
            "total_dim": len(self.special_neuron_indices),
            "dim_prop": effective_dim / len(self.special_neuron_indices),
            "explained_variance_ratio": explained_variance_ratio,
            "cumulative_variance": cumulative_variance,
            "random_effective_dim": random_dims,
            "random_mean_dim": np.mean(random_dims),
            "ttest_stat": tstat,
            "ttest_p": pvalue,
            "is_significantly_different": pvalue < 0.05,
            "comparison": "lower" if effective_dim < np.mean(random_dims) else "higher",
        }

        self.dimensionality_results = results
        return results

    def analyze_orthogonality(self) -> dict[str, Any]:
        """Analyze the orthogonality/alignment between neurons in the special group."""
        # Transpose to get neuron x context matrix for analyzing neuron relationships
        neuron_matrix = self.special_activation_matrix.T

        # Normalize neuron vectors
        normalized_neurons = neuron_matrix.copy()
        norms = np.linalg.norm(normalized_neurons, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        normalized_neurons = normalized_neurons / norms

        # Compute cosine similarity matrix (inner products of normalized vectors)
        cosine_similarity = normalized_neurons @ normalized_neurons.T

        # Extract upper triangle (excluding diagonal) for distribution analysis
        upper_indices = np.triu_indices_from(cosine_similarity, k=1)
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

        # Analyze random groups for comparison
        random_mean_similarities = []
        random_near_orthogonal = []

        for random_matrix in self.random_groups:
            r_neuron_matrix = random_matrix.T
            r_normalized = r_neuron_matrix.copy()
            r_norms = np.linalg.norm(r_normalized, axis=1, keepdims=True)
            r_norms[r_norms == 0] = 1.0
            r_normalized = r_normalized / r_norms

            r_cosine = r_normalized @ r_normalized.T
            r_upper = r_cosine[np.triu_indices_from(r_cosine, k=1)]
            r_angles = np.degrees(np.arccos(np.clip(r_upper, -1.0, 1.0)))

            random_mean_similarities.append(np.mean(r_upper))
            random_near_orthogonal.append(np.mean((r_angles >= 80) & (r_angles <= 100)) * 100)

        # Statistical comparison
        tstat_sim, pvalue_sim = ttest_ind([mean_similarity], random_mean_similarities, equal_var=False)

        tstat_orth, pvalue_orth = ttest_ind([near_orthogonal], random_near_orthogonal, equal_var=False)

        results = {
            "cosine_similarity_matrix": cosine_similarity,
            "mean_cosine_similarity": mean_similarity,
            "median_cosine_similarity": median_similarity,
            "max_cosine_similarity": max_similarity,
            "min_cosine_similarity": min_similarity,
            "angle_distribution": angle_distribution,
            "mean_angle_degrees": mean_angle,
            "pct_near_orthogonal": near_orthogonal,
            "random_mean_similarities": random_mean_similarities,
            "random_near_orthogonal": random_near_orthogonal,
            "similarity_ttest": {
                "statistic": tstat_sim,
                "pvalue": pvalue_sim,
                "is_significant": pvalue_sim < 0.05,
                "comparison": "higher" if mean_similarity > np.mean(random_mean_similarities) else "lower",
            },
            "orthogonality_ttest": {
                "statistic": tstat_orth,
                "pvalue": pvalue_orth,
                "is_significant": pvalue_orth < 0.05,
                "comparison": "higher" if near_orthogonal > np.mean(random_near_orthogonal) else "lower",
            },
        }

        self.orthogonality_results = results
        return results

    def analyze_coactivation_hierarchical(self):
        # Transpose to get neurons as rows
        neuron_matrix = self.special_activation_matrix.T
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(neuron_matrix)
        # Convert correlations to distances (highly correlated neurons = small distance)
        distance_matrix = 1 - np.abs(correlation_matrix)
        # Perform hierarchical clustering
        Z = linkage(distance_matrix[np.triu_indices(len(distance_matrix), k=1)], method="ward")
        # Get clusters at different thresholds
        clusters_tight = fcluster(Z, t=3, criterion="distance")
        clusters_medium = fcluster(Z, t=5, criterion="distance")
        clusters_loose = fcluster(Z, t=7, criterion="distance")

        # Calculate metrics
        n_clusters_tight = len(np.unique(clusters_tight))
        n_clusters_medium = len(np.unique(clusters_medium))
        n_clusters_loose = len(np.unique(clusters_loose))

        # Do the same for random groups for comparison
        random_n_clusters_medium = []
        for random_matrix in self.random_groups:
            r_neuron_matrix = random_matrix.T
            r_corr = np.corrcoef(r_neuron_matrix)
            r_dist = 1 - np.abs(r_corr)
            r_Z = linkage(r_dist[np.triu_indices(len(r_dist), k=1)], method="ward")
            r_clusters = fcluster(r_Z, t=5, criterion="distance")
            random_n_clusters_medium.append(len(np.unique(r_clusters)))

        # Statistical comparison
        tstat, pvalue = ttest_ind([n_clusters_medium], random_n_clusters_medium, equal_var=False)

        return {
            "linkage": Z,
            "clusters_tight": clusters_tight,
            "clusters_medium": clusters_medium,
            "clusters_loose": clusters_loose,
            "n_clusters_tight": n_clusters_tight,
            "n_clusters_medium": n_clusters_medium,
            "n_clusters_loose": n_clusters_loose,
            "random_n_clusters_medium": random_n_clusters_medium,
            "ttest_stat": tstat,
            "ttest_p": pvalue,
            "is_significant": pvalue < 0.05,
            "comparison": "fewer clusters"
            if n_clusters_medium < np.mean(random_n_clusters_medium)
            else "more clusters",
        }

    def run_all_analyses(self) -> dict[str, dict[str, Any]]:
        """Run all geometric analyses at once."""
        self.analyze_dimensionality()
        self.analyze_orthogonality()
        self.analyze_coactivation()

        combined_results = {
            "dimensionality": self.dimensionality_results,
            "orthogonality": self.orthogonality_results,
            "coactivation": self.coactivation_results,
        }

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

        combined_results["summary"] = summary
        self.comparative_results = combined_results

        return combined_results
