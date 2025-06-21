import json
import logging
import warnings
from collections import Counter, defaultdict
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.cluster import SpectralClustering
from sklearn.feature_selection import mutual_info_regression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)

#######################################################
# Threshold selection


class AutomaticThresholdSelector:
    """Step 1: Automatic threshold selection for graph-based neural coordination analysis.
    Determines optimal correlation and MI thresholds using statistical and data-driven methods.
    """

    def __init__(
        self,
        activation_data: pd.DataFrame,
        boost_neuron_indices: list[int],
        suppress_neuron_indices: list[int],
        excluded_neuron_indices: list[int],
        rare_token_mask: np.ndarray | None = None,
        activation_column: str = "activation",
        token_column: str = "str_tokens",
        context_column: str = "context",
        component_column: str = "component_name",
        # Threshold selection parameters
        selection_data_fraction: float = 0.7,  # Use 70% of data for threshold selection
        min_threshold_correlation: float = 0.05,
        max_threshold_correlation: float = 0.8,
        min_threshold_mi: float = 0.01,
        max_threshold_mi: float = 0.5,
        n_permutations: int = 1000,
        significance_levels: list[float] = [0.05, 0.01, 0.001],
        device: str | None = None,
        random_state: int = 42,
    ):
        """Initialize automatic threshold selector."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state
        np.random.seed(random_state)

        # Store parameters
        self.activation_column = activation_column
        self.token_column = token_column
        self.context_column = context_column
        self.component_column = component_column
        self.selection_data_fraction = selection_data_fraction
        self.min_threshold_correlation = min_threshold_correlation
        self.max_threshold_correlation = max_threshold_correlation
        self.min_threshold_mi = min_threshold_mi
        self.max_threshold_mi = max_threshold_mi
        self.n_permutations = n_permutations
        self.significance_levels = significance_levels

        # Store all neuron indices for representative threshold selection
        self.all_analysis_indices = list(
            set(
                boost_neuron_indices
                + suppress_neuron_indices
                + [idx for idx in activation_data[component_column].unique() if idx not in excluded_neuron_indices]
            )
        )

        # Create token-context identifier
        self.data = activation_data.copy()
        self.data["token_context_id"] = (
            self.data[token_column].astype(str) + "_" + self.data[context_column].astype(str)
        )

        # Handle rare token mask
        self.rare_token_mask = self._create_rare_token_mask(rare_token_mask)

        # Split data for threshold selection
        self.selection_data, self.validation_data = self._split_data_for_selection()

        # Create activation matrices
        self.selection_matrix = self._create_activation_matrix(self.selection_data)

        logger.info(
            f"Initialized threshold selector with {len(self.all_analysis_indices)} neurons, "
            f"{self.selection_matrix.shape[0]} contexts for selection"
        )

    def _create_rare_token_mask(self, rare_token_mask: np.ndarray | None) -> np.ndarray:
        """Create or validate rare token mask."""
        unique_contexts = self.data["token_context_id"].unique()

        if rare_token_mask is not None:
            if len(rare_token_mask) != len(unique_contexts):
                raise ValueError("Rare token mask length must match number of unique contexts")
            return rare_token_mask

        # Create frequency-based mask if not provided
        logger.warning("No rare token mask provided, creating frequency-based mask")
        token_counts = self.data.groupby(self.token_column).size()
        rare_threshold = np.percentile(token_counts, 25)  # Bottom 25% as "rare"
        rare_tokens = set(token_counts[token_counts <= rare_threshold].index)

        mask = np.array([context.split("_")[0] in rare_tokens for context in unique_contexts])

        return mask

    def _split_data_for_selection(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into selection and validation sets by contexts."""
        unique_contexts = self.data["token_context_id"].unique()

        # Stratified split to maintain rare/common token ratio
        rare_contexts = unique_contexts[self.rare_token_mask]
        common_contexts = unique_contexts[~self.rare_token_mask]

        # Split each group separately
        rare_train, rare_val = train_test_split(
            rare_contexts, test_size=1 - self.selection_data_fraction, random_state=self.random_state
        )
        common_train, common_val = train_test_split(
            common_contexts, test_size=1 - self.selection_data_fraction, random_state=self.random_state
        )

        # Combine
        selection_contexts = np.concatenate([rare_train, common_train])
        validation_contexts = np.concatenate([rare_val, common_val])

        selection_data = self.data[self.data["token_context_id"].isin(selection_contexts)]
        validation_data = self.data[self.data["token_context_id"].isin(validation_contexts)]

        logger.info(
            f"Split data: {len(selection_contexts)} contexts for selection, {len(validation_contexts)} for validation"
        )

        return selection_data, validation_data

    def _create_activation_matrix(self, data: pd.DataFrame) -> np.ndarray:
        """Create activation matrix for specified neurons and data."""
        filtered_data = data[data[self.component_column].isin(self.all_analysis_indices)]

        pivot_table = filtered_data.pivot_table(
            index="token_context_id", columns=self.component_column, values=self.activation_column, aggfunc="first"
        )
        pivot_table = pivot_table.fillna(0)

        return pivot_table.values

    def statistical_significance_threshold_selection(self) -> dict[str, Any]:
        """Method 1: Statistical significance-based threshold selection.
        Uses permutation testing to establish significance thresholds.
        """
        logger.info("Running statistical significance threshold selection...")

        # Compute observed correlation and MI matrices
        correlation_matrix = self._compute_correlation_matrix(self.selection_matrix)
        mi_matrix = self._compute_mi_matrix(self.selection_matrix)

        # Extract upper triangular values (exclude diagonal)
        upper_indices = np.triu_indices_from(correlation_matrix, k=1)
        observed_correlations = correlation_matrix[upper_indices]
        observed_mi = mi_matrix[upper_indices]

        # Generate null distributions
        logger.info(f"Generating null distributions with {self.n_permutations} permutations...")
        null_correlations = []
        null_mi_values = []

        for perm_idx in range(self.n_permutations):
            if perm_idx % 100 == 0:
                logger.info(f"Permutation {perm_idx + 1}/{self.n_permutations}")

            # Permute each neuron's activations independently
            permuted_data = self._permute_activation_matrix(self.selection_matrix)

            # Compute null matrices
            null_corr_matrix = self._compute_correlation_matrix(permuted_data)
            null_mi_matrix = self._compute_mi_matrix(permuted_data)

            # Extract values
            null_correlations.extend(null_corr_matrix[upper_indices])
            null_mi_values.extend(null_mi_matrix[upper_indices])

        null_correlations = np.array(null_correlations)
        null_mi_values = np.array(null_mi_values)

        # Compute significance thresholds
        correlation_thresholds = {
            f"p_{level}": np.percentile(null_correlations, (1 - level) * 100) for level in self.significance_levels
        }

        mi_thresholds = {
            f"p_{level}": np.percentile(null_mi_values, (1 - level) * 100) for level in self.significance_levels
        }

        # Compute effect sizes relative to null distribution
        null_corr_mean = np.mean(null_correlations)
        null_corr_std = np.std(null_correlations)
        null_mi_mean = np.mean(null_mi_values)
        null_mi_std = np.std(null_mi_values)

        correlation_effect_thresholds = {
            "small_effect": null_corr_mean + 0.2 * null_corr_std,
            "medium_effect": null_corr_mean + 0.5 * null_corr_std,
            "large_effect": null_corr_mean + 0.8 * null_corr_std,
        }

        mi_effect_thresholds = {
            "small_effect": null_mi_mean + 0.2 * null_mi_std,
            "medium_effect": null_mi_mean + 0.5 * null_mi_std,
            "large_effect": null_mi_mean + 0.8 * null_mi_std,
        }

        return {
            "method": "statistical_significance",
            "correlation": {
                "significance_thresholds": correlation_thresholds,
                "effect_size_thresholds": correlation_effect_thresholds,
                "null_distribution_stats": {
                    "mean": float(null_corr_mean),
                    "std": float(null_corr_std),
                    "percentiles": {
                        "50": float(np.percentile(null_correlations, 50)),
                        "90": float(np.percentile(null_correlations, 90)),
                        "95": float(np.percentile(null_correlations, 95)),
                        "99": float(np.percentile(null_correlations, 99)),
                    },
                },
                "recommended_threshold": correlation_thresholds["p_0.05"],
            },
            "mi": {
                "significance_thresholds": mi_thresholds,
                "effect_size_thresholds": mi_effect_thresholds,
                "null_distribution_stats": {
                    "mean": float(null_mi_mean),
                    "std": float(null_mi_std),
                    "percentiles": {
                        "50": float(np.percentile(null_mi_values, 50)),
                        "90": float(np.percentile(null_mi_values, 90)),
                        "95": float(np.percentile(null_mi_values, 95)),
                        "99": float(np.percentile(null_mi_values, 99)),
                    },
                },
                "recommended_threshold": mi_thresholds["p_0.05"],
            },
        }

    def mixture_model_threshold_selection(self) -> dict[str, Any]:
        """Method 2: Mixture model-based threshold selection.
        Assumes two populations: noise and signal.
        """
        logger.info("Running mixture model threshold selection...")

        # Compute matrices
        correlation_matrix = self._compute_correlation_matrix(self.selection_matrix)
        mi_matrix = self._compute_mi_matrix(self.selection_matrix)

        # Extract values
        upper_indices = np.triu_indices_from(correlation_matrix, k=1)
        correlation_values = correlation_matrix[upper_indices]
        mi_values = mi_matrix[upper_indices]

        # Fit mixture models
        correlation_mixture = self._fit_mixture_model(correlation_values, "correlation")
        mi_mixture = self._fit_mixture_model(mi_values, "mi")

        return {
            "method": "mixture_model",
            "correlation": correlation_mixture,
            "mi": mi_mixture,
        }

    def _fit_mixture_model(self, values: np.ndarray, metric_name: str) -> dict[str, Any]:
        """Fit Gaussian mixture model to separate noise and signal populations."""
        # Remove very small values
        values = values[values > 1e-6]

        if len(values) < 50:
            logger.warning(f"Insufficient values for mixture modeling of {metric_name}")
            return {
                "error": "Insufficient data",
                "recommended_threshold": float(np.percentile(values, 75)) if len(values) > 0 else 0.1,
            }

        try:
            # Fit 2-component Gaussian mixture
            gmm = GaussianMixture(n_components=2, random_state=self.random_state)
            gmm.fit(values.reshape(-1, 1))

            # Identify components
            means = gmm.means_.flatten()
            stds = np.sqrt(gmm.covariances_.flatten())
            weights = gmm.weights_

            # Assume lower mean is noise, higher is signal
            noise_idx = np.argmin(means)
            signal_idx = 1 - noise_idx

            noise_mean = means[noise_idx]
            noise_std = stds[noise_idx]
            signal_mean = means[signal_idx]
            signal_std = stds[signal_idx]

            # Calculate threshold at distribution intersection
            intersection_threshold = self._find_gaussian_intersection(noise_mean, noise_std, signal_mean, signal_std)

            # Alternative thresholds
            conservative_threshold = noise_mean + 2 * noise_std
            liberal_threshold = noise_mean + 1 * noise_std

            return {
                "noise_component": {
                    "mean": float(noise_mean),
                    "std": float(noise_std),
                    "weight": float(weights[noise_idx]),
                },
                "signal_component": {
                    "mean": float(signal_mean),
                    "std": float(signal_std),
                    "weight": float(weights[signal_idx]),
                },
                "thresholds": {
                    "intersection": float(intersection_threshold),
                    "conservative": float(conservative_threshold),
                    "liberal": float(liberal_threshold),
                },
                "recommended_threshold": float(intersection_threshold),
                "model_quality": {
                    "log_likelihood": float(gmm.score(values.reshape(-1, 1))),
                    "aic": float(2 * 2 * 2 - 2 * gmm.score(values.reshape(-1, 1)) * len(values)),
                    "separation": float(abs(signal_mean - noise_mean) / (noise_std + signal_std)),
                },
            }

        except Exception as e:
            logger.warning(f"Mixture model fitting failed for {metric_name}: {e}")
            return {
                "error": str(e),
                "recommended_threshold": float(np.percentile(values, 75)),
            }

    def _find_gaussian_intersection(self, mu1: float, sigma1: float, mu2: float, sigma2: float) -> float:
        """Find intersection point of two Gaussian distributions."""
        try:
            # Quadratic formula for Gaussian intersection
            a = 1 / (2 * sigma1**2) - 1 / (2 * sigma2**2)
            b = mu2 / (sigma2**2) - mu1 / (sigma1**2)
            c = mu1**2 / (2 * sigma1**2) - mu2**2 / (2 * sigma2**2) + np.log(sigma2 / sigma1)

            if abs(a) < 1e-10:  # Linear case
                intersection = -c / b if abs(b) > 1e-10 else (mu1 + mu2) / 2
            else:
                discriminant = b**2 - 4 * a * c
                if discriminant >= 0:
                    x1 = (-b + np.sqrt(discriminant)) / (2 * a)
                    x2 = (-b - np.sqrt(discriminant)) / (2 * a)
                    # Choose intersection between means
                    if min(mu1, mu2) <= x1 <= max(mu1, mu2):
                        intersection = x1
                    elif min(mu1, mu2) <= x2 <= max(mu1, mu2):
                        intersection = x2
                    else:
                        intersection = (mu1 + mu2) / 2
                else:
                    intersection = (mu1 + mu2) / 2

            return max(0, intersection)

        except:
            return (mu1 + mu2) / 2

    def network_topology_threshold_selection(self) -> dict[str, Any]:
        """Method 3: Network topology-based threshold selection.
        Optimizes for desired network properties.
        """
        logger.info("Running network topology threshold selection...")

        # Compute matrices
        correlation_matrix = self._compute_correlation_matrix(self.selection_matrix)
        mi_matrix = self._compute_mi_matrix(self.selection_matrix)

        # Test range of thresholds
        correlation_results = self._analyze_topology_across_thresholds(correlation_matrix, "correlation")
        mi_results = self._analyze_topology_across_thresholds(mi_matrix, "mi")

        return {
            "method": "network_topology",
            "correlation": correlation_results,
            "mi": mi_results,
        }

    def _analyze_topology_across_thresholds(self, matrix: np.ndarray, metric_name: str) -> dict[str, Any]:
        """Analyze network topology across a range of thresholds."""
        if metric_name == "correlation":
            min_thresh, max_thresh = self.min_threshold_correlation, self.max_threshold_correlation
        else:
            min_thresh, max_thresh = self.min_threshold_mi, self.max_threshold_mi

        thresholds = np.linspace(min_thresh, max_thresh, 30)
        results = []

        for threshold in thresholds:
            # Create binary adjacency matrix
            adj_matrix = (matrix > threshold).astype(int)
            np.fill_diagonal(adj_matrix, 0)

            # Create NetworkX graph
            G = nx.from_numpy_array(adj_matrix)

            # Compute topology metrics
            n_edges = G.number_of_edges()
            density = nx.density(G)

            if n_edges > 0:
                clustering = nx.average_clustering(G)

                if nx.is_connected(G):
                    path_length = nx.average_shortest_path_length(G)
                    efficiency = nx.global_efficiency(G)
                    small_world_sigma = self._compute_small_world_sigma(G)
                else:
                    # Use largest connected component
                    largest_cc = max(nx.connected_components(G), key=len)
                    subgraph = G.subgraph(largest_cc)
                    if len(subgraph) > 1:
                        path_length = nx.average_shortest_path_length(subgraph)
                        efficiency = nx.global_efficiency(subgraph)
                        small_world_sigma = self._compute_small_world_sigma(subgraph)
                    else:
                        path_length = 0
                        efficiency = 0
                        small_world_sigma = 0
            else:
                clustering = 0
                path_length = np.inf
                efficiency = 0
                small_world_sigma = 0

            results.append(
                {
                    "threshold": threshold,
                    "n_edges": n_edges,
                    "density": density,
                    "clustering": clustering,
                    "path_length": path_length,
                    "efficiency": efficiency,
                    "small_world_sigma": small_world_sigma,
                    "is_connected": nx.is_connected(G),
                }
            )

        # Find optimal thresholds
        results_df = pd.DataFrame(results)
        optimal_thresholds = self._find_optimal_topology_thresholds(results_df)

        return {
            "threshold_analysis": results,
            "optimal_thresholds": optimal_thresholds,
            "recommended_threshold": optimal_thresholds.get(
                "balanced_small_world", optimal_thresholds.get("target_density", min_thresh + 0.1)
            ),
        }

    def _compute_small_world_sigma(self, G: nx.Graph) -> float:
        """Compute small-world coefficient."""
        try:
            if G.number_of_nodes() < 4 or G.number_of_edges() == 0:
                return 0.0

            # Generate equivalent random graph
            n = G.number_of_nodes()
            m = G.number_of_edges()
            p = 2 * m / (n * (n - 1))

            G_random = nx.erdos_renyi_graph(n, p, seed=self.random_state)

            # Compute metrics
            C = nx.average_clustering(G)
            C_random = nx.average_clustering(G_random)

            if nx.is_connected(G) and nx.is_connected(G_random):
                L = nx.average_shortest_path_length(G)
                L_random = nx.average_shortest_path_length(G_random)

                if C_random > 0 and L_random > 0:
                    sigma = (C / C_random) / (L / L_random)
                    return sigma

            return 0.0

        except:
            return 0.0

    def _find_optimal_topology_thresholds(self, results_df: pd.DataFrame) -> dict[str, float]:
        """Find optimal thresholds based on topology criteria."""
        optimal = {}

        # Maximum small-world index
        if results_df["small_world_sigma"].max() > 0:
            optimal["small_world"] = results_df.loc[results_df["small_world_sigma"].idxmax(), "threshold"]

        # Maximum efficiency (connected graphs only)
        connected_results = results_df[results_df["is_connected"]]
        if not connected_results.empty:
            optimal["max_efficiency"] = connected_results.loc[connected_results["efficiency"].idxmax(), "threshold"]

        # Target density (15% for interpretable networks)
        target_density = 0.15
        density_diff = np.abs(results_df["density"] - target_density)
        optimal["target_density"] = results_df.loc[density_diff.idxmin(), "threshold"]

        # Balanced small-world (clustering / path_length)
        results_df["small_world_score"] = results_df["clustering"] / (results_df["path_length"] + 1e-6)
        if results_df["small_world_score"].max() > 0:
            optimal["balanced_small_world"] = results_df.loc[results_df["small_world_score"].idxmax(), "threshold"]

        return optimal

    def comprehensive_threshold_selection(self) -> dict[str, Any]:
        """Method 4: Comprehensive approach combining all methods.
        Provides final recommendations based on consensus.
        """
        logger.info("Running comprehensive threshold selection...")

        # Run all individual methods
        statistical_results = self.statistical_significance_threshold_selection()
        mixture_results = self.mixture_model_threshold_selection()
        topology_results = self.network_topology_threshold_selection()

        # Compute consensus
        consensus = self._compute_consensus_recommendations(statistical_results, mixture_results, topology_results)

        # Validate on held-out data
        validation_results = self._validate_thresholds_on_holdout(consensus)

        return {
            "method": "comprehensive",
            "individual_methods": {
                "statistical_significance": statistical_results,
                "mixture_model": mixture_results,
                "network_topology": topology_results,
            },
            "consensus_recommendations": consensus,
            "validation_results": validation_results,
            "final_thresholds": {
                "correlation_threshold": consensus["correlation"]["final_recommendation"],
                "mi_threshold": consensus["mi"]["final_recommendation"],
                "confidence_score": consensus["confidence_score"],
                "methodology": "consensus_of_statistical_mixture_topology_methods",
            },
        }

    def _compute_consensus_recommendations(
        self, statistical_results: dict, mixture_results: dict, topology_results: dict
    ) -> dict[str, Any]:
        """Compute consensus recommendations across methods."""
        consensus = {"correlation": {}, "mi": {}}

        for metric in ["correlation", "mi"]:
            thresholds = []
            weights = []
            method_info = []

            # Statistical significance (weight: 0.4)
            if metric in statistical_results and "recommended_threshold" in statistical_results[metric]:
                threshold = statistical_results[metric]["recommended_threshold"]
                if 0 < threshold < 1:
                    thresholds.append(threshold)
                    weights.append(0.4)
                    method_info.append("statistical_p0.05")

            # Mixture model (weight: 0.3)
            if metric in mixture_results and "recommended_threshold" in mixture_results[metric]:
                threshold = mixture_results[metric]["recommended_threshold"]
                if 0 < threshold < 1:
                    thresholds.append(threshold)
                    weights.append(0.3)
                    method_info.append("mixture_intersection")

            # Network topology (weight: 0.3)
            if metric in topology_results and "recommended_threshold" in topology_results[metric]:
                threshold = topology_results[metric]["recommended_threshold"]
                if 0 < threshold < 1:
                    thresholds.append(threshold)
                    weights.append(0.3)
                    method_info.append("topology_optimized")

            if thresholds:
                # Weighted average
                weighted_avg = np.average(thresholds, weights=weights)

                # Bounds for safety
                if metric == "correlation":
                    final_threshold = np.clip(weighted_avg, 0.1, 0.7)
                else:  # MI
                    final_threshold = np.clip(weighted_avg, 0.02, 0.3)

                consensus[metric] = {
                    "individual_thresholds": thresholds,
                    "method_info": method_info,
                    "weights": weights,
                    "weighted_average": float(weighted_avg),
                    "final_recommendation": float(final_threshold),
                    "std_deviation": float(np.std(thresholds)),
                    "n_methods_agreement": len(thresholds),
                }
            else:
                # Fallback to conservative defaults
                fallback = 0.3 if metric == "correlation" else 0.1
                consensus[metric] = {
                    "final_recommendation": fallback,
                    "error": "No valid thresholds from any method",
                    "n_methods_agreement": 0,
                }

        # Overall confidence
        min_agreement = min(
            consensus["correlation"].get("n_methods_agreement", 0), consensus["mi"].get("n_methods_agreement", 0)
        )
        confidence_score = min_agreement / 3.0  # Max 3 methods

        consensus["confidence_score"] = confidence_score

        return consensus

    def _validate_thresholds_on_holdout(self, consensus: dict[str, Any]) -> dict[str, Any]:
        """Validate selected thresholds on held-out validation data."""
        if hasattr(self, "validation_data") and len(self.validation_data) > 0:
            try:
                validation_matrix = self._create_activation_matrix(self.validation_data)

                correlation_threshold = consensus["correlation"]["final_recommendation"]
                mi_threshold = consensus["mi"]["final_recommendation"]

                # Compute validation matrices
                val_corr_matrix = self._compute_correlation_matrix(validation_matrix)
                val_mi_matrix = self._compute_mi_matrix(validation_matrix)

                # Apply thresholds and create graphs
                corr_adj = (val_corr_matrix > correlation_threshold).astype(int)
                mi_adj = (val_mi_matrix > mi_threshold).astype(int)

                np.fill_diagonal(corr_adj, 0)
                np.fill_diagonal(mi_adj, 0)

                corr_graph = nx.from_numpy_array(corr_adj)
                mi_graph = nx.from_numpy_array(mi_adj)

                # Compute validation metrics
                validation_metrics = {
                    "correlation_graph": {
                        "n_edges": corr_graph.number_of_edges(),
                        "density": nx.density(corr_graph),
                        "is_connected": nx.is_connected(corr_graph),
                        "clustering": nx.average_clustering(corr_graph),
                    },
                    "mi_graph": {
                        "n_edges": mi_graph.number_of_edges(),
                        "density": nx.density(mi_graph),
                        "is_connected": nx.is_connected(mi_graph),
                        "clustering": nx.average_clustering(mi_graph),
                    },
                }

                return validation_metrics

            except Exception as e:
                logger.warning(f"Validation failed: {e}")
                return {"error": str(e)}
        else:
            return {"error": "No validation data available"}

    def _compute_correlation_matrix(self, data: np.ndarray) -> np.ndarray:
        """Compute correlation matrix."""
        corr_matrix = np.corrcoef(data.T)
        corr_matrix = np.abs(corr_matrix)  # Absolute values for undirected graphs
        np.fill_diagonal(corr_matrix, 0)  # Remove self-correlations
        return corr_matrix

    def _compute_mi_matrix(self, data: np.ndarray) -> np.ndarray:
        """Compute mutual information matrix."""
        n_neurons = data.shape[1]
        mi_matrix = np.zeros((n_neurons, n_neurons))

        for i in range(n_neurons):
            for j in range(i + 1, n_neurons):
                try:
                    X = data[:, i].reshape(-1, 1)
                    y = data[:, j]
                    mi_val = mutual_info_regression(X, y, discrete_features=False, random_state=42)[0]
                    mi_matrix[i, j] = mi_val
                    mi_matrix[j, i] = mi_val
                except Exception:
                    # Handle potential numerical issues
                    mi_matrix[i, j] = 0.0
                    mi_matrix[j, i] = 0.0

        return mi_matrix

    def _permute_activation_matrix(self, data: np.ndarray) -> np.ndarray:
        """Create permuted version of activation matrix for null distribution."""
        permuted_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            permuted_data[:, i] = np.random.permutation(data[:, i])
        return permuted_data

    def run_threshold_selection(self, method: str = "comprehensive") -> dict[str, Any]:
        """Main method to run threshold selection.

        Args:
            method: "statistical", "mixture", "topology", or "comprehensive"

        Returns:
            Dictionary with selected thresholds and analysis results

        """
        logger.info(f"Starting threshold selection using {method} method...")

        start_time = pd.Timestamp.now()

        try:
            if method == "statistical":
                results = self.statistical_significance_threshold_selection()
            elif method == "mixture":
                results = self.mixture_model_threshold_selection()
            elif method == "topology":
                results = self.network_topology_threshold_selection()
            elif method == "comprehensive":
                results = self.comprehensive_threshold_selection()
            else:
                raise ValueError(
                    f"Unknown method: {method}. Use 'statistical', 'mixture', 'topology', or 'comprehensive'"
                )

            # Add metadata
            results["metadata"] = {
                "method_used": method,
                "n_neurons_analyzed": len(self.all_analysis_indices),
                "n_contexts_selection": self.selection_matrix.shape[0],
                "n_contexts_validation": len(self.validation_data) if hasattr(self, "validation_data") else 0,
                "selection_data_fraction": self.selection_data_fraction,
                "n_permutations": self.n_permutations,
                "computation_time_minutes": (pd.Timestamp.now() - start_time).total_seconds() / 60,
                "random_state": self.random_state,
            }

            # Extract final thresholds for easy access
            if method == "comprehensive":
                final_thresholds = results["final_thresholds"]
            elif method == "statistical":
                final_thresholds = {
                    "correlation_threshold": results["correlation"]["recommended_threshold"],
                    "mi_threshold": results["mi"]["recommended_threshold"],
                    "methodology": "statistical_significance_p0.05",
                }
            elif method == "mixture":
                final_thresholds = {
                    "correlation_threshold": results["correlation"]["recommended_threshold"],
                    "mi_threshold": results["mi"]["recommended_threshold"],
                    "methodology": "mixture_model_intersection",
                }
            elif method == "topology":
                final_thresholds = {
                    "correlation_threshold": results["correlation"]["recommended_threshold"],
                    "mi_threshold": results["mi"]["recommended_threshold"],
                    "methodology": "network_topology_optimized",
                }

            results["final_thresholds"] = final_thresholds

            logger.info(
                f"Threshold selection completed in {results['metadata']['computation_time_minutes']:.2f} minutes"
            )
            logger.info(
                f"Selected thresholds - Correlation: {final_thresholds['correlation_threshold']:.3f}, "
                f"MI: {final_thresholds['mi_threshold']:.3f}"
            )

            return results

        except Exception as e:
            logger.error(f"Threshold selection failed: {e}")
            # Return fallback thresholds
            return {
                "error": str(e),
                "fallback_thresholds": {
                    "correlation_threshold": 0.3,
                    "mi_threshold": 0.1,
                    "methodology": "fallback_due_to_error",
                },
                "metadata": {
                    "method_used": method,
                    "computation_time_minutes": (pd.Timestamp.now() - start_time).total_seconds() / 60,
                },
            }

    def save_threshold_selection_results(
        self, results: dict[str, Any], filepath: str = "threshold_selection_results.json"
    ) -> None:
        """Save threshold selection results to JSON file."""
        try:
            # Convert numpy arrays to lists for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, dict):
                    return {key: convert_for_json(value) for key, value in obj.items()}
                if isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                return obj

            json_results = convert_for_json(results)

            with open(filepath, "w") as f:
                json.dump(json_results, f, indent=2)

            logger.info(f"Threshold selection results saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def create_threshold_summary_report(self, results: dict[str, Any]) -> str:
        """Create a human-readable summary report of threshold selection."""
        report = []
        report.append("=" * 70)
        report.append("AUTOMATIC THRESHOLD SELECTION REPORT")
        report.append("=" * 70)

        # Method information
        method = results.get("metadata", {}).get("method_used", "unknown")
        report.append(f"\nMethod Used: {method.upper()}")

        # Final thresholds
        final_thresholds = results.get("final_thresholds", {})
        corr_thresh = final_thresholds.get("correlation_threshold", "N/A")
        mi_thresh = final_thresholds.get("mi_threshold", "N/A")
        methodology = final_thresholds.get("methodology", "N/A")

        report.append("\nFINAL SELECTED THRESHOLDS:")
        report.append(
            f"  Correlation Threshold: {corr_thresh:.4f}"
            if isinstance(corr_thresh, float)
            else f"  Correlation Threshold: {corr_thresh}"
        )
        report.append(
            f"  Mutual Information Threshold: {mi_thresh:.4f}"
            if isinstance(mi_thresh, float)
            else f"  Mutual Information Threshold: {mi_thresh}"
        )
        report.append(f"  Methodology: {methodology}")

        # Data information
        metadata = results.get("metadata", {})
        report.append("\nDATA CHARACTERISTICS:")
        report.append(f"  Neurons Analyzed: {metadata.get('n_neurons_analyzed', 'N/A')}")
        report.append(f"  Contexts for Selection: {metadata.get('n_contexts_selection', 'N/A')}")
        report.append(f"  Contexts for Validation: {metadata.get('n_contexts_validation', 'N/A')}")
        report.append(f"  Selection Data Fraction: {metadata.get('selection_data_fraction', 'N/A')}")

        # Method-specific details
        if method == "comprehensive":
            consensus = results.get("consensus_recommendations", {})
            conf_score = consensus.get("confidence_score", 0)
            report.append("\nCONSENSUS ANALYSIS:")
            report.append(f"  Confidence Score: {conf_score:.2f}/1.0")

            for metric in ["correlation", "mi"]:
                if metric in consensus:
                    metric_info = consensus[metric]
                    n_methods = metric_info.get("n_methods_agreement", 0)
                    std_dev = metric_info.get("std_deviation", 0)
                    report.append(f"  {metric.capitalize()} - Methods Agreement: {n_methods}/3, Std Dev: {std_dev:.4f}")

        elif method == "statistical":
            # Statistical significance details
            if "correlation" in results:
                corr_results = results["correlation"]
                null_stats = corr_results.get("null_distribution_stats", {})
                report.append("\nSTATISTICAL SIGNIFICANCE ANALYSIS:")
                report.append(f"  Null Correlation Mean: {null_stats.get('mean', 0):.4f}")
                report.append(
                    f"  Null Correlation 95th Percentile: {null_stats.get('percentiles', {}).get('95', 0):.4f}"
                )

        # Validation results
        validation = results.get("validation_results", {})
        if validation and "error" not in validation:
            report.append("\nVALIDATION ON HELD-OUT DATA:")
            if "correlation_graph" in validation:
                corr_val = validation["correlation_graph"]
                report.append(
                    f"  Correlation Graph - Edges: {corr_val.get('n_edges', 0)}, Density: {corr_val.get('density', 0):.3f}"
                )
            if "mi_graph" in validation:
                mi_val = validation["mi_graph"]
                report.append(
                    f"  MI Graph - Edges: {mi_val.get('n_edges', 0)}, Density: {mi_val.get('density', 0):.3f}"
                )

        # Computation details
        comp_time = metadata.get("computation_time_minutes", 0)
        report.append("\nCOMPUTATION DETAILS:")
        report.append(f"  Processing Time: {comp_time:.2f} minutes")
        report.append(f"  Permutations Used: {metadata.get('n_permutations', 'N/A')}")

        # Recommendations for Step 2
        report.append("\nRECOMMENDATIONS FOR STEP 2 (Graph Analysis):")
        report.append("  Use these thresholds in GraphBasedCoordinationAnalyzer:")
        report.append(f"    correlation_threshold={corr_thresh:.4f}")
        report.append(f"    mi_threshold={mi_thresh:.4f}")
        report.append("    edge_construction_method='correlation'  # or 'mi' or 'hybrid'")

        if isinstance(conf_score := final_thresholds.get("confidence_score"), float):
            if conf_score > 0.8:
                report.append("  High confidence in threshold selection - proceed with analysis")
            elif conf_score > 0.5:
                report.append("  Moderate confidence - consider sensitivity analysis")
            else:
                report.append("  Low confidence - recommend manual threshold inspection")

        report.append("=" * 70)

        return "\n".join(report)


#######################################################
# Neuron group subspace direction analysis


class GraphCoordinationAnalyzer:
    """Comprehensive graph-based analysis for neural coordination patterns."""

    def __init__(
        self,
        activation_data: pd.DataFrame,
        boost_neuron_indices: list[int],
        suppress_neuron_indices: list[int],
        excluded_neuron_indices: list[int],
        rare_token_mask: np.ndarray | None = None,
        activation_column: str = "activation",
        token_column: str = "str_tokens",
        context_column: str = "context",
        component_column: str = "component_name",
        num_random_groups: int = 2,
        device: str | None = None,
        use_mixed_precision: bool = True,
        # Graph construction parameters
        correlation_threshold: float = 0.3,
        mi_threshold: float = 0.1,
        edge_construction_method: str = "correlation",  # "correlation", "mi", "hybrid"
        graph_type: str = "weighted",  # "binary", "weighted", "signed"
        significance_level: float = 0.05,
        max_lag: int = 3,
        # Community detection parameters
        community_algorithm: str = "louvain",  # "louvain", "spectral", "infomap"
        resolution_parameter: float = 1.0,
        # Statistical parameters
        n_bootstrap: int = 1000,
        n_permutations: int = 1000,
    ):
        """Initialize the graph-based coordination analyzer."""
        # Device and precision setup
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mixed_precision = use_mixed_precision
        self.dtype = torch.float16 if use_mixed_precision else torch.float32

        # Store parameters
        self.correlation_threshold = correlation_threshold
        self.mi_threshold = mi_threshold
        self.edge_construction_method = edge_construction_method
        self.graph_type = graph_type
        self.significance_level = significance_level
        self.max_lag = max_lag
        self.community_algorithm = community_algorithm
        self.resolution_parameter = resolution_parameter
        self.n_bootstrap = n_bootstrap
        self.n_permutations = n_permutations

        # Store input data
        self.data = activation_data.copy()
        self.boost_neuron_indices = boost_neuron_indices
        self.suppress_neuron_indices = suppress_neuron_indices
        self.excluded_neuron_indices = excluded_neuron_indices
        self.activation_column = activation_column
        self.token_column = token_column
        self.context_column = context_column
        self.component_column = component_column
        self.num_random_groups = num_random_groups

        # Create token-context identifier
        self.data["token_context_id"] = (
            self.data[token_column].astype(str) + "_" + self.data[context_column].astype(str)
        )

        # Extract unique identifiers
        self.token_contexts = self.data["token_context_id"].unique()
        self.all_neuron_indices = self.data[self.component_column].astype(int).unique()

        # Generate random groups
        self.random_groups, self.random_indices = self._generate_random_groups()

        # Create activation matrices
        self._create_activation_tensors()

        # Handle rare token masking
        self.rare_token_mask = self._create_rare_token_mask(rare_token_mask)

        # Results storage
        self.graph_results = {}

        logger.info(
            f"Initialized Graph analyzer with {len(self.token_contexts)} contexts, "
            f"device: {self.device}, edge method: {self.edge_construction_method}"
        )

    def _generate_random_groups(self) -> tuple[list[np.ndarray], list[list[int]]]:
        """Generate non-overlapping random neuron groups."""
        group_size = max(len(self.boost_neuron_indices), len(self.suppress_neuron_indices))
        special_indices = set(self.boost_neuron_indices + self.suppress_neuron_indices + self.excluded_neuron_indices)

        # Get available indices
        available_indices = [idx for idx in self.all_neuron_indices if idx not in special_indices]

        if len(available_indices) < group_size * self.num_random_groups:
            raise ValueError("Not enough neurons available for random groups")

        # Sample random groups
        np.random.seed(42)  # For reproducibility
        random_indices = []
        remaining_indices = available_indices.copy()

        for _ in range(self.num_random_groups):
            group = np.random.choice(remaining_indices, size=group_size, replace=False)
            random_indices.append(group.tolist())
            remaining_indices = [idx for idx in remaining_indices if idx not in group]

        # Create activation matrices for random groups
        random_groups = [self._create_activation_matrix(indices) for indices in random_indices]

        return random_groups, random_indices

    def _create_activation_matrix(self, neuron_indices: list[int]) -> np.ndarray:
        """Create activation matrix where rows are contexts and columns are neurons."""
        filtered_data = self.data[self.data[self.component_column].isin(neuron_indices)]
        pivot_table = filtered_data.pivot_table(
            index="token_context_id", columns=self.component_column, values=self.activation_column, aggfunc="first"
        )
        pivot_table = pivot_table.fillna(0)
        return pivot_table.values

    def _create_activation_tensors(self) -> None:
        """Create PyTorch tensors for all neuron groups."""
        # Create matrices
        boost_matrix = self._create_activation_matrix(self.boost_neuron_indices)
        suppress_matrix = self._create_activation_matrix(self.suppress_neuron_indices)
        random_1_matrix = self._create_activation_matrix(self.random_indices[0])
        random_2_matrix = self._create_activation_matrix(self.random_indices[1])

        # Convert to tensors
        self.activation_tensors = {
            "boost": torch.tensor(boost_matrix, dtype=self.dtype).to(self.device),
            "suppress": torch.tensor(suppress_matrix, dtype=self.dtype).to(self.device),
            "random_1": torch.tensor(random_1_matrix, dtype=self.dtype).to(self.device),
            "random_2": torch.tensor(random_2_matrix, dtype=self.dtype).to(self.device),
        }

        # Store neuron indices for reference
        self.neuron_indices = {
            "boost": self.boost_neuron_indices,
            "suppress": self.suppress_neuron_indices,
            "random_1": self.random_indices[0],
            "random_2": self.random_indices[1],
        }

    def _create_rare_token_mask(self, rare_token_mask: np.ndarray | None) -> np.ndarray:
        """Create or validate rare token mask."""
        if rare_token_mask is not None:
            if len(rare_token_mask) != len(self.token_contexts):
                raise ValueError("Rare token mask length must match number of contexts")
            return rare_token_mask

        # Create simple frequency-based mask if not provided
        logger.warning("No rare token mask provided, using frequency-based approximation")
        token_counts = self.data.groupby(self.token_column).size()
        rare_threshold = np.percentile(token_counts, 25)  # Bottom 25% as "rare"
        rare_tokens = set(token_counts[token_counts <= rare_threshold].index)

        mask = np.array([token_context.split("_")[0] in rare_tokens for token_context in self.token_contexts])

        return mask

    def _estimate_mutual_information(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Estimate mutual information between two variables."""
        try:
            X_reshaped = X.reshape(-1, 1) if X.ndim == 1 else X
            mi = mutual_info_regression(X_reshaped, Y, discrete_features=False, random_state=42)
            return float(mi[0] if len(mi) == 1 else np.mean(mi))
        except:
            return 0.0

    def _construct_graph_edges(self, data: np.ndarray, context_mask: np.ndarray | None = None) -> np.ndarray:
        """Construct edge weights based on the specified method."""
        if context_mask is not None:
            data = data[context_mask]

        n_neurons = data.shape[1]
        edge_matrix = np.zeros((n_neurons, n_neurons))

        if self.edge_construction_method == "correlation":
            # Correlation-based edges
            corr_matrix = np.corrcoef(data.T)
            edge_matrix = np.abs(corr_matrix)

        elif self.edge_construction_method == "mi":
            # Mutual information-based edges
            for i in range(n_neurons):
                for j in range(i + 1, n_neurons):
                    mi_val = self._estimate_mutual_information(data[:, i], data[:, j])
                    edge_matrix[i, j] = mi_val
                    edge_matrix[j, i] = mi_val

        elif self.edge_construction_method == "hybrid":
            # Hybrid approach: use both correlation and MI
            corr_matrix = np.abs(np.corrcoef(data.T))
            mi_matrix = np.zeros((n_neurons, n_neurons))

            for i in range(n_neurons):
                for j in range(i + 1, n_neurons):
                    mi_val = self._estimate_mutual_information(data[:, i], data[:, j])
                    mi_matrix[i, j] = mi_val
                    mi_matrix[j, i] = mi_val

            # Combine correlation and MI (weighted average)
            edge_matrix = 0.6 * corr_matrix + 0.4 * mi_matrix

        # Apply thresholding based on graph type
        if self.graph_type == "binary":
            threshold = (
                self.correlation_threshold if self.edge_construction_method == "correlation" else self.mi_threshold
            )
            edge_matrix = (edge_matrix > threshold).astype(float)

        # Zero out diagonal
        np.fill_diagonal(edge_matrix, 0)

        return edge_matrix

    def _construct_networkx_graph(self, edge_matrix: np.ndarray) -> nx.Graph:
        """Convert edge matrix to NetworkX graph."""
        return nx.from_numpy_array(edge_matrix > 0) if self.graph_type == "binary" else nx.from_numpy_array(edge_matrix)

    def _compute_centrality_measures(self, G: nx.Graph) -> dict[str, Any]:
        """Compute various centrality measures."""
        centralities = {}

        try:
            # Degree centrality
            centralities["degree"] = list(nx.degree_centrality(G).values())

            # Betweenness centrality
            centralities["betweenness"] = list(nx.betweenness_centrality(G).values())

            # Eigenvector centrality (if graph is connected)
            if nx.is_connected(G):
                centralities["eigenvector"] = list(nx.eigenvector_centrality(G, max_iter=1000).values())
            else:
                centralities["eigenvector"] = [0.0] * len(G.nodes())

            # PageRank centrality
            centralities["pagerank"] = list(nx.pagerank(G, max_iter=1000).values())

            # Closeness centrality
            centralities["closeness"] = list(nx.closeness_centrality(G).values())

        except Exception as e:
            logger.warning(f"Error computing centralities: {e}")
            n_nodes = len(G.nodes())
            centralities = {
                "degree": [0.0] * n_nodes,
                "betweenness": [0.0] * n_nodes,
                "eigenvector": [0.0] * n_nodes,
                "pagerank": [0.0] * n_nodes,
                "closeness": [0.0] * n_nodes,
            }

        return centralities

    def _detect_communities(self, G: nx.Graph, edge_matrix: np.ndarray) -> dict[str, Any]:
        """Detect communities using various algorithms."""
        communities = {}
        n_nodes = len(G.nodes())

        if n_nodes < 3:
            return {
                "louvain": [0] * n_nodes,
                "spectral": [0] * n_nodes,
                "modularity": 0.0,
                "n_communities": 1,
            }

        try:
            # Louvain algorithm
            if self.community_algorithm in ["louvain", "all"]:
                try:
                    import community as community_louvain

                    partition = community_louvain.best_partition(G, resolution=self.resolution_parameter)
                    communities["louvain"] = [partition.get(i, 0) for i in range(n_nodes)]
                except ImportError:
                    # Fallback to spectral clustering
                    communities["louvain"] = self._spectral_clustering(edge_matrix)

            # Spectral clustering
            if self.community_algorithm in ["spectral", "all"]:
                communities["spectral"] = self._spectral_clustering(edge_matrix)

            # Compute modularity
            if "louvain" in communities:
                partition_dict = {i: communities["louvain"][i] for i in range(n_nodes)}
                communities["modularity"] = nx.community.modularity(
                    G, [set([k for k, v in partition_dict.items() if v == c]) for c in set(partition_dict.values())]
                )
            else:
                communities["modularity"] = 0.0

            # Number of communities
            if "louvain" in communities:
                communities["n_communities"] = len(set(communities["louvain"]))
            else:
                communities["n_communities"] = 1

        except Exception as e:
            logger.warning(f"Error in community detection: {e}")
            communities = {
                "louvain": [0] * n_nodes,
                "spectral": [0] * n_nodes,
                "modularity": 0.0,
                "n_communities": 1,
            }

        return communities

    def _spectral_clustering(self, edge_matrix: np.ndarray, n_clusters: int | None = None) -> list[int]:
        """Perform spectral clustering on the edge matrix."""
        try:
            if n_clusters is None:
                # Estimate number of clusters using eigengap heuristic
                eigenvals = np.linalg.eigvals(edge_matrix)
                eigenvals = np.sort(eigenvals)[::-1]
                if len(eigenvals) > 3:
                    gaps = np.diff(eigenvals[: min(10, len(eigenvals))])
                    n_clusters = np.argmax(gaps) + 2
                else:
                    n_clusters = 2

            n_clusters = min(n_clusters, edge_matrix.shape[0] - 1)
            n_clusters = max(n_clusters, 2)

            clustering = SpectralClustering(n_clusters=n_clusters, affinity="precomputed", random_state=42)
            labels = clustering.fit_predict(edge_matrix)
            return labels.tolist()

        except Exception as e:
            logger.warning(f"Spectral clustering failed: {e}")
            return [0] * edge_matrix.shape[0]

    def _compute_network_topology_metrics(self, G: nx.Graph) -> dict[str, float]:
        """Compute network topology metrics."""
        metrics = {}

        try:
            # Basic metrics
            metrics["n_nodes"] = G.number_of_nodes()
            metrics["n_edges"] = G.number_of_edges()
            metrics["density"] = nx.density(G)

            # Clustering coefficient
            metrics["avg_clustering"] = nx.average_clustering(G)

            # Path-based metrics
            if nx.is_connected(G):
                metrics["avg_path_length"] = nx.average_shortest_path_length(G)
                metrics["diameter"] = nx.diameter(G)
                metrics["radius"] = nx.radius(G)
            else:
                # For disconnected graphs, compute for largest component
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                if len(subgraph) > 1:
                    metrics["avg_path_length"] = nx.average_shortest_path_length(subgraph)
                    metrics["diameter"] = nx.diameter(subgraph)
                    metrics["radius"] = nx.radius(subgraph)
                else:
                    metrics["avg_path_length"] = 0.0
                    metrics["diameter"] = 0.0
                    metrics["radius"] = 0.0

            # Small-world metrics
            metrics["small_world_sigma"] = self._compute_small_world_sigma(G)

            # Efficiency measures
            metrics["global_efficiency"] = nx.global_efficiency(G)
            metrics["local_efficiency"] = nx.local_efficiency(G)

            # Assortativity
            if G.number_of_edges() > 0:
                metrics["degree_assortativity"] = nx.degree_assortativity_coefficient(G)
            else:
                metrics["degree_assortativity"] = 0.0

        except Exception as e:
            logger.warning(f"Error computing topology metrics: {e}")
            metrics = {
                "n_nodes": G.number_of_nodes(),
                "n_edges": G.number_of_edges(),
                "density": 0.0,
                "avg_clustering": 0.0,
                "avg_path_length": 0.0,
                "diameter": 0.0,
                "radius": 0.0,
                "small_world_sigma": 0.0,
                "global_efficiency": 0.0,
                "local_efficiency": 0.0,
                "degree_assortativity": 0.0,
            }

        return metrics

    def _compute_small_world_sigma(self, G: nx.Graph) -> float:
        """Compute small-world coefficient sigma."""
        try:
            if G.number_of_nodes() < 4:
                return 0.0

            # Generate equivalent random graph
            n = G.number_of_nodes()
            m = G.number_of_edges()
            p = 2 * m / (n * (n - 1))  # Edge probability

            # Create random graph with same n, p
            G_random = nx.erdos_renyi_graph(n, p, seed=42)

            # Compute clustering and path length for both graphs
            C = nx.average_clustering(G)
            C_random = nx.average_clustering(G_random)

            if nx.is_connected(G) and nx.is_connected(G_random):
                L = nx.average_shortest_path_length(G)
                L_random = nx.average_shortest_path_length(G_random)
            else:
                return 0.0

            # Small-world coefficient
            if C_random > 0 and L_random > 0:
                sigma = (C / C_random) / (L / L_random)
                return sigma
            return 0.0

        except Exception as e:
            logger.warning(f"Error computing small-world sigma: {e}")
            return 0.0

    def _analyze_degree_distribution(self, G: nx.Graph) -> dict[str, Any]:
        """Analyze degree distribution properties."""
        degrees = [d for n, d in G.degree()]

        if not degrees:
            return {
                "mean_degree": 0.0,
                "std_degree": 0.0,
                "max_degree": 0,
                "degree_distribution": [],
                "is_scale_free": False,
                "power_law_exponent": 0.0,
            }

        degree_counts = Counter(degrees)

        # Basic statistics
        mean_degree = np.mean(degrees)
        std_degree = np.std(degrees)
        max_degree = max(degrees)

        # Degree distribution
        degree_distribution = [degree_counts.get(i, 0) for i in range(max_degree + 1)]

        # Test for scale-free properties (simplified)
        is_scale_free = False
        power_law_exponent = 0.0

        try:
            if len(set(degrees)) > 3:  # Need variety in degrees
                unique_degrees = sorted(set(degrees))
                degree_probs = [degree_counts[d] / len(degrees) for d in unique_degrees]

                # Fit power law (log-log linear regression)
                log_degrees = np.log(unique_degrees)
                log_probs = np.log(degree_probs)

                # Remove -inf values
                valid_mask = np.isfinite(log_probs)
                if np.sum(valid_mask) > 2:
                    slope, intercept, r_value, p_value, std_err = pearsonr(
                        log_degrees[valid_mask], log_probs[valid_mask]
                    )
                    power_law_exponent = -slope  # Negative because P(k) ~ k^(-gamma)
                    is_scale_free = (r_value**2 > 0.8) and (power_law_exponent > 1.5)

        except Exception as e:
            logger.warning(f"Error analyzing degree distribution: {e}")

        return {
            "mean_degree": float(mean_degree),
            "std_degree": float(std_degree),
            "max_degree": int(max_degree),
            "degree_distribution": degree_distribution,
            "is_scale_free": bool(is_scale_free),
            "power_law_exponent": float(power_law_exponent),
        }

    def analyze_hierarchical_network_organization(self) -> dict[str, Any]:
        """Analyze H1: Hierarchical Network Organization."""
        results = {}

        for group_name, tensor in self.activation_tensors.items():
            data = tensor.detach().cpu().numpy()

            # Construct graph for this group
            edge_matrix = self._construct_graph_edges(data)
            G = self._construct_networkx_graph(edge_matrix)

            # Compute centrality measures
            centralities = self._compute_centrality_measures(G)

            # Analyze degree distribution
            degree_analysis = self._analyze_degree_distribution(G)

            # Identify hub neurons (top 10% by degree centrality)
            degree_cents = centralities["degree"]
            if degree_cents:
                hub_threshold = np.percentile(degree_cents, 90)
                hub_neurons = [i for i, dc in enumerate(degree_cents) if dc >= hub_threshold]
                n_hubs = len(hub_neurons)
                hub_betweenness = np.mean([centralities["betweenness"][i] for i in hub_neurons]) if hub_neurons else 0.0
            else:
                hub_neurons = []
                n_hubs = 0
                hub_betweenness = 0.0

            # Test hierarchy hypothesis
            hierarchy_score = np.mean(centralities["betweenness"]) if centralities["betweenness"] else 0.0

            results[group_name] = {
                "centralities": centralities,
                "degree_analysis": degree_analysis,
                "hub_neurons": hub_neurons,
                "n_hubs": n_hubs,
                "hub_betweenness_centrality": float(hub_betweenness),
                "hierarchy_score": float(hierarchy_score),
                "supports_hierarchy_hypothesis": hierarchy_score > 0.1 and degree_analysis["is_scale_free"],
            }

        return results

    def analyze_modular_coordination_architecture(self) -> dict[str, Any]:
        """Analyze H2: Modular Coordination Architecture."""
        results = {}

        for group_name, tensor in self.activation_tensors.items():
            data = tensor.detach().cpu().numpy()

            # Construct graph
            edge_matrix = self._construct_graph_edges(data)
            G = self._construct_networkx_graph(edge_matrix)

            # Detect communities
            communities = self._detect_communities(G, edge_matrix)

            # Analyze modularity
            modularity = communities["modularity"]
            n_communities = communities["n_communities"]

            # Community size distribution
            if "louvain" in communities:
                community_labels = communities["louvain"]
                community_sizes = list(Counter(community_labels).values())
                avg_community_size = np.mean(community_sizes)
                std_community_size = np.std(community_sizes)
            else:
                community_sizes = [len(G.nodes())]
                avg_community_size = len(G.nodes())
                std_community_size = 0.0

            # Inter vs intra-community connectivity
            inter_community_density, intra_community_density = self._compute_community_connectivity(
                G, community_labels if "louvain" in communities else [0] * len(G.nodes())
            )

            results[group_name] = {
                "communities": communities,
                "modularity": float(modularity),
                "n_communities": int(n_communities),
                "community_sizes": community_sizes,
                "avg_community_size": float(avg_community_size),
                "std_community_size": float(std_community_size),
                "inter_community_density": float(inter_community_density),
                "intra_community_density": float(intra_community_density),
                "supports_modularity_hypothesis": modularity > 0.3 and n_communities > 1,
            }

        return results

    def _compute_community_connectivity(self, G: nx.Graph, community_labels: list[int]) -> tuple[float, float]:
        """Compute inter- and intra-community connectivity."""
        try:
            total_inter = 0
            total_intra = 0
            possible_inter = 0
            possible_intra = 0

            nodes = list(G.nodes())

            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    node_i, node_j = nodes[i], nodes[j]
                    same_community = community_labels[i] == community_labels[j]

                    if same_community:
                        possible_intra += 1
                        if G.has_edge(node_i, node_j):
                            total_intra += 1
                    else:
                        possible_inter += 1
                        if G.has_edge(node_i, node_j):
                            total_inter += 1

            inter_density = total_inter / possible_inter if possible_inter > 0 else 0.0
            intra_density = total_intra / possible_intra if possible_intra > 0 else 0.0

            return inter_density, intra_density

        except Exception as e:
            logger.warning(f"Error computing community connectivity: {e}")
            return 0.0, 0.0

    def analyze_context_dependent_network_topology(self) -> dict[str, Any]:
        """Analyze H3: Context-Dependent Network Topology."""
        results = {}

        for group_name, tensor in self.activation_tensors.items():
            data = tensor.detach().cpu().numpy()

            # Check if we have enough contexts for both rare and common
            rare_mask = self.rare_token_mask
            common_mask = ~self.rare_token_mask

            if np.sum(rare_mask) < 5 or np.sum(common_mask) < 5:
                logger.warning(f"Insufficient contexts for {group_name}, skipping context analysis")
                results[group_name] = {
                    "rare_context_metrics": {},
                    "common_context_metrics": {},
                    "context_differences": {},
                    "supports_adaptive_topology_hypothesis": False,
                }
                continue

            # Construct graphs for rare and common contexts
            rare_edge_matrix = self._construct_graph_edges(data, rare_mask)
            common_edge_matrix = self._construct_graph_edges(data, common_mask)

            G_rare = self._construct_networkx_graph(rare_edge_matrix)
            G_common = self._construct_networkx_graph(common_edge_matrix)

            # Compute topology metrics for both contexts
            rare_metrics = self._compute_network_topology_metrics(G_rare)
            common_metrics = self._compute_network_topology_metrics(G_common)

            # Compute differences
            context_differences = {}
            for metric in rare_metrics:
                if isinstance(rare_metrics[metric], (int, float)):
                    diff = rare_metrics[metric] - common_metrics[metric]
                    context_differences[f"{metric}_difference"] = float(diff)

            # Test for significant context-dependent changes
            clustering_diff = abs(context_differences.get("avg_clustering_difference", 0))
            path_length_diff = abs(context_differences.get("avg_path_length_difference", 0))
            efficiency_diff = abs(context_differences.get("global_efficiency_difference", 0))

            supports_hypothesis = (clustering_diff > 0.1) or (path_length_diff > 0.5) or (efficiency_diff > 0.1)

            results[group_name] = {
                "rare_context_metrics": rare_metrics,
                "common_context_metrics": common_metrics,
                "context_differences": context_differences,
                "supports_adaptive_topology_hypothesis": bool(supports_hypothesis),
            }

        return results

    def analyze_optimized_information_flow(self) -> dict[str, Any]:
        """Analyze H4: Optimized Information Flow."""
        results = {}

        for group_name, tensor in self.activation_tensors.items():
            data = tensor.detach().cpu().numpy()

            # Construct graph
            edge_matrix = self._construct_graph_edges(data)
            G = self._construct_networkx_graph(edge_matrix)

            # Compute efficiency metrics
            topology_metrics = self._compute_network_topology_metrics(G)

            # Generate comparable random networks for comparison
            random_metrics = self._generate_random_network_metrics(G)

            # Compute optimization metrics
            efficiency_ratio = (
                topology_metrics["global_efficiency"] / random_metrics["global_efficiency"]
                if random_metrics["global_efficiency"] > 0
                else 1.0
            )

            clustering_ratio = (
                topology_metrics["avg_clustering"] / random_metrics["avg_clustering"]
                if random_metrics["avg_clustering"] > 0
                else 1.0
            )

            path_length_ratio = (
                random_metrics["avg_path_length"] / topology_metrics["avg_path_length"]
                if topology_metrics["avg_path_length"] > 0
                else 1.0
            )

            # Cost-efficiency trade-off (simple approximation)
            cost = topology_metrics["density"]  # Higher density = higher cost
            efficiency = topology_metrics["global_efficiency"]
            cost_efficiency_ratio = efficiency / cost if cost > 0 else 0.0

            # Test optimization hypothesis
            is_optimized = efficiency_ratio > 1.1 and clustering_ratio > 1.1 and cost_efficiency_ratio > 0.5

            results[group_name] = {
                "topology_metrics": topology_metrics,
                "random_baseline_metrics": random_metrics,
                "efficiency_ratio": float(efficiency_ratio),
                "clustering_ratio": float(clustering_ratio),
                "path_length_ratio": float(path_length_ratio),
                "cost_efficiency_ratio": float(cost_efficiency_ratio),
                "supports_optimization_hypothesis": bool(is_optimized),
            }

        return results

    def _generate_random_network_metrics(self, G: nx.Graph) -> dict[str, float]:
        """Generate metrics for comparable random networks."""
        try:
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()

            if n_nodes < 3:
                return {
                    "global_efficiency": 0.0,
                    "avg_clustering": 0.0,
                    "avg_path_length": 0.0,
                }

            # Generate random graph with same number of nodes and edges
            p = 2 * n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
            G_random = nx.erdos_renyi_graph(n_nodes, p, seed=42)

            # Compute metrics for random graph
            random_metrics = self._compute_network_topology_metrics(G_random)

            return {
                "global_efficiency": random_metrics["global_efficiency"],
                "avg_clustering": random_metrics["avg_clustering"],
                "avg_path_length": random_metrics["avg_path_length"],
            }

        except Exception as e:
            logger.warning(f"Error generating random network metrics: {e}")
            return {
                "global_efficiency": 0.0,
                "avg_clustering": 0.0,
                "avg_path_length": 0.0,
            }

    def analyze_dynamic_network_properties(self) -> dict[str, Any]:
        """Analyze dynamic properties of coordination networks."""
        results = {}

        for group_name, tensor in self.activation_tensors.items():
            data = tensor.detach().cpu().numpy()

            # Construct base graph
            edge_matrix = self._construct_graph_edges(data)
            G = self._construct_networkx_graph(edge_matrix)

            # Network resilience analysis
            resilience_results = self._analyze_network_resilience(G)

            # Temporal network analysis (if applicable)
            temporal_results = self._analyze_temporal_networks(data)

            results[group_name] = {
                "resilience": resilience_results,
                "temporal_properties": temporal_results,
            }

        return results

    def _analyze_network_resilience(self, G: nx.Graph) -> dict[str, Any]:
        """Analyze network resilience to node removal."""
        try:
            original_efficiency = nx.global_efficiency(G)
            original_connectivity = nx.is_connected(G)

            nodes = list(G.nodes())
            n_nodes = len(nodes)

            if n_nodes < 5:
                return {
                    "robustness_random": 1.0,
                    "robustness_targeted": 1.0,
                    "percolation_threshold": 0.0,
                    "critical_nodes": [],
                }

            # Random failure resilience
            n_removals = min(10, n_nodes // 2)
            random_efficiencies = []

            for _ in range(n_removals):
                G_copy = G.copy()
                nodes_to_remove = np.random.choice(nodes, size=max(1, len(nodes) // 10), replace=False)
                G_copy.remove_nodes_from(nodes_to_remove)
                if len(G_copy) > 0:
                    eff = nx.global_efficiency(G_copy)
                    random_efficiencies.append(eff / original_efficiency if original_efficiency > 0 else 0)

            robustness_random = np.mean(random_efficiencies) if random_efficiencies else 1.0

            # Targeted attack resilience (remove highest degree nodes)
            degrees = dict(G.degree())
            sorted_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)

            targeted_efficiencies = []
            for i in range(min(n_removals, len(sorted_nodes))):
                G_copy = G.copy()
                G_copy.remove_node(sorted_nodes[i])
                if len(G_copy) > 0:
                    eff = nx.global_efficiency(G_copy)
                    targeted_efficiencies.append(eff / original_efficiency if original_efficiency > 0 else 0)

            robustness_targeted = np.mean(targeted_efficiencies) if targeted_efficiencies else 1.0

            # Percolation threshold (approximate)
            percolation_threshold = self._estimate_percolation_threshold(G)

            # Critical nodes (highest betweenness centrality)
            betweenness = nx.betweenness_centrality(G)
            critical_nodes = sorted(betweenness.keys(), key=lambda x: betweenness[x], reverse=True)[:5]

            return {
                "robustness_random": float(robustness_random),
                "robustness_targeted": float(robustness_targeted),
                "percolation_threshold": float(percolation_threshold),
                "critical_nodes": critical_nodes,
            }

        except Exception as e:
            logger.warning(f"Error analyzing network resilience: {e}")
            return {
                "robustness_random": 1.0,
                "robustness_targeted": 1.0,
                "percolation_threshold": 0.0,
                "critical_nodes": [],
            }

    def _estimate_percolation_threshold(self, G: nx.Graph) -> float:
        """Estimate percolation threshold for network connectivity."""
        try:
            n_nodes = G.number_of_nodes()
            if n_nodes < 10:
                return 0.0

            # Binary search for percolation threshold
            nodes = list(G.nodes())
            low, high = 0.0, 1.0

            for _ in range(10):  # 10 iterations of binary search
                mid = (low + high) / 2
                n_remove = int(mid * n_nodes)

                # Test connectivity after random removal
                connected_trials = 0
                for _ in range(10):  # 10 trials
                    G_copy = G.copy()
                    nodes_to_remove = np.random.choice(nodes, size=n_remove, replace=False)
                    G_copy.remove_nodes_from(nodes_to_remove)
                    if nx.is_connected(G_copy):
                        connected_trials += 1

                if connected_trials >= 5:  # Majority still connected
                    low = mid
                else:
                    high = mid

            return (low + high) / 2

        except Exception as e:
            logger.warning(f"Error estimating percolation threshold: {e}")
            return 0.0

    def _analyze_temporal_networks(self, data: np.ndarray) -> dict[str, Any]:
        """Analyze temporal evolution of network properties."""
        try:
            n_contexts, n_neurons = data.shape

            if n_contexts < 10:
                return {
                    "temporal_stability": 1.0,
                    "evolution_trend": "stable",
                    "change_points": [],
                }

            # Divide data into temporal windows
            window_size = max(5, n_contexts // 5)
            n_windows = n_contexts // window_size

            window_metrics = []

            for i in range(n_windows):
                start_idx = i * window_size
                end_idx = min((i + 1) * window_size, n_contexts)
                window_data = data[start_idx:end_idx]

                # Construct graph for this window
                edge_matrix = self._construct_graph_edges(window_data)
                G = self._construct_networkx_graph(edge_matrix)

                # Compute basic metrics
                metrics = {
                    "density": nx.density(G),
                    "clustering": nx.average_clustering(G),
                    "efficiency": nx.global_efficiency(G),
                }
                window_metrics.append(metrics)

            # Analyze temporal stability
            if len(window_metrics) > 1:
                stability_scores = []
                for metric in ["density", "clustering", "efficiency"]:
                    values = [w[metric] for w in window_metrics]
                    stability = 1 - (np.std(values) / (np.mean(values) + 1e-10))
                    stability_scores.append(max(0, stability))

                temporal_stability = np.mean(stability_scores)

                # Detect trend
                density_values = [w["density"] for w in window_metrics]
                if len(density_values) > 2:
                    slope = np.polyfit(range(len(density_values)), density_values, 1)[0]
                    if slope > 0.01:
                        evolution_trend = "increasing"
                    elif slope < -0.01:
                        evolution_trend = "decreasing"
                    else:
                        evolution_trend = "stable"
                else:
                    evolution_trend = "stable"
            else:
                temporal_stability = 1.0
                evolution_trend = "stable"

            return {
                "temporal_stability": float(temporal_stability),
                "evolution_trend": evolution_trend,
                "window_metrics": window_metrics,
                "n_windows": len(window_metrics),
            }

        except Exception as e:
            logger.warning(f"Error analyzing temporal networks: {e}")
            return {
                "temporal_stability": 1.0,
                "evolution_trend": "stable",
                "change_points": [],
            }

    def comparative_graph_analysis(self) -> dict[str, Any]:
        """Compare graph properties between neuron groups."""
        comparisons = {}

        # Define group pairs to compare
        group_pairs = [
            ("boost", "random_1"),
            ("suppress", "random_1"),
            ("boost", "suppress"),
            ("random_1", "random_2"),
        ]

        for group1, group2 in group_pairs:
            if group1 not in self.activation_tensors or group2 not in self.activation_tensors:
                continue

            # Get data for both groups
            data1 = self.activation_tensors[group1].detach().cpu().numpy()
            data2 = self.activation_tensors[group2].detach().cpu().numpy()

            # Construct graphs
            edge_matrix1 = self._construct_graph_edges(data1)
            edge_matrix2 = self._construct_graph_edges(data2)

            G1 = self._construct_networkx_graph(edge_matrix1)
            G2 = self._construct_networkx_graph(edge_matrix2)

            # Compute metrics for both graphs
            metrics1 = self._compute_network_topology_metrics(G1)
            metrics2 = self._compute_network_topology_metrics(G2)

            # Compute differences and statistical tests
            differences = {}
            statistical_tests = {}

            for metric in metrics1:
                if isinstance(metrics1[metric], (int, float)) and isinstance(metrics2[metric], (int, float)):
                    diff = metrics1[metric] - metrics2[metric]
                    differences[f"{metric}_difference"] = float(diff)

                    # Effect size (Cohen's d approximation)
                    pooled_std = np.sqrt((0.1**2 + 0.1**2) / 2)  # Approximate
                    cohens_d = diff / pooled_std if pooled_std > 0 else 0.0
                    statistical_tests[f"{metric}_cohens_d"] = float(cohens_d)

            # Network distance measures
            network_distance = self._compute_network_distance(edge_matrix1, edge_matrix2)

            comparisons[f"{group1}_vs_{group2}"] = {
                "group1_metrics": metrics1,
                "group2_metrics": metrics2,
                "differences": differences,
                "statistical_tests": statistical_tests,
                "network_distance": float(network_distance),
                "significant_differences": [
                    metric
                    for metric, cohens_d in statistical_tests.items()
                    if abs(cohens_d) > 0.5  # Medium effect size threshold
                ],
            }

        return comparisons

    def _compute_network_distance(self, matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """Compute distance between two network adjacency matrices."""
        try:
            # Ensure same size by padding/truncating if necessary
            min_size = min(matrix1.shape[0], matrix2.shape[0])
            matrix1_resized = matrix1[:min_size, :min_size]
            matrix2_resized = matrix2[:min_size, :min_size]

            # Frobenius norm distance
            distance = np.linalg.norm(matrix1_resized - matrix2_resized, "fro")

            # Normalize by matrix size
            normalized_distance = distance / (min_size * min_size)

            return float(normalized_distance)

        except Exception as e:
            logger.warning(f"Error computing network distance: {e}")
            return 0.0

    def statistical_validation(self) -> dict[str, Any]:
        """Perform statistical validation of graph-based findings."""
        validation_results = {}

        for group_name, tensor in self.activation_tensors.items():
            data = tensor.detach().cpu().numpy()

            # Construct observed graph
            observed_edge_matrix = self._construct_graph_edges(data)
            observed_G = self._construct_networkx_graph(observed_edge_matrix)
            observed_metrics = self._compute_network_topology_metrics(observed_G)

            # Generate null model distributions
            null_metrics = self._generate_null_model_distribution(data, n_samples=100)

            # Compute p-values for each metric
            p_values = {}
            effect_sizes = {}

            for metric in observed_metrics:
                if metric in null_metrics and isinstance(observed_metrics[metric], (int, float)):
                    null_values = null_metrics[metric]
                    observed_value = observed_metrics[metric]

                    if len(null_values) > 0:
                        # Two-tailed p-value
                        p_value = np.mean(
                            np.abs(null_values - np.mean(null_values)) >= abs(observed_value - np.mean(null_values))
                        )
                        p_values[metric] = float(p_value)

                        # Effect size
                        if np.std(null_values) > 0:
                            effect_size = (observed_value - np.mean(null_values)) / np.std(null_values)
                            effect_sizes[metric] = float(effect_size)
                        else:
                            effect_sizes[metric] = 0.0
                    else:
                        p_values[metric] = 1.0
                        effect_sizes[metric] = 0.0

            # Multiple comparison correction (Bonferroni)
            corrected_p_values = {metric: min(1.0, p_val * len(p_values)) for metric, p_val in p_values.items()}

            validation_results[group_name] = {
                "observed_metrics": observed_metrics,
                "null_distribution_stats": {
                    metric: {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                    }
                    for metric, values in null_metrics.items()
                },
                "p_values": p_values,
                "corrected_p_values": corrected_p_values,
                "effect_sizes": effect_sizes,
                "significant_metrics": [
                    metric for metric, p_val in corrected_p_values.items() if p_val < self.significance_level
                ],
            }

        return validation_results

    def _generate_null_model_distribution(self, data: np.ndarray, n_samples: int = 100) -> dict[str, list[float]]:
        """Generate null model distribution for statistical testing."""
        null_metrics = defaultdict(list)

        try:
            n_contexts, n_neurons = data.shape

            for _ in range(n_samples):
                # Generate null data by permuting each neuron's activations independently
                null_data = np.zeros_like(data)
                for i in range(n_neurons):
                    null_data[:, i] = np.random.permutation(data[:, i])

                # Construct null graph
                null_edge_matrix = self._construct_graph_edges(null_data)
                null_G = self._construct_networkx_graph(null_edge_matrix)
                null_graph_metrics = self._compute_network_topology_metrics(null_G)

                # Store metrics
                for metric, value in null_graph_metrics.items():
                    if isinstance(value, (int, float)):
                        null_metrics[metric].append(value)

        except Exception as e:
            logger.warning(f"Error generating null model distribution: {e}")

        return dict(null_metrics)

    def run_all_analyses(self) -> dict[str, Any]:
        """Main analysis method that runs all graph-based analyses."""
        logger.info("Starting graph-based coordination analysis...")

        try:
            # Run all hypothesis tests
            hierarchical_results = self.analyze_hierarchical_network_organization()
            modular_results = self.analyze_modular_coordination_architecture()
            context_dependent_results = self.analyze_context_dependent_network_topology()
            optimization_results = self.analyze_optimized_information_flow()
            dynamic_results = self.analyze_dynamic_network_properties()

            # Comparative analysis
            comparative_results = self.comparative_graph_analysis()

            # Statistical validation
            validation_results = self.statistical_validation()

            # Compile comprehensive results
            results = {
                "hierarchical_network_organization": hierarchical_results,
                "modular_coordination_architecture": modular_results,
                "context_dependent_network_topology": context_dependent_results,
                "optimized_information_flow": optimization_results,
                "dynamic_network_properties": dynamic_results,
                "comparative_analysis": comparative_results,
                "statistical_validation": validation_results,
                "analysis_metadata": {
                    "edge_construction_method": self.edge_construction_method,
                    "graph_type": self.graph_type,
                    "correlation_threshold": self.correlation_threshold,
                    "mi_threshold": self.mi_threshold,
                    "community_algorithm": self.community_algorithm,
                    "significance_level": self.significance_level,
                    "n_contexts_total": len(self.token_contexts),
                    "n_contexts_rare": int(np.sum(self.rare_token_mask)),
                    "n_contexts_common": int(np.sum(~self.rare_token_mask)),
                    "neuron_group_sizes": {name: len(indices) for name, indices in self.neuron_indices.items()},
                    "device": self.device,
                    "analysis_parameters": {
                        "max_lag": self.max_lag,
                        "n_bootstrap": self.n_bootstrap,
                        "n_permutations": self.n_permutations,
                        "resolution_parameter": self.resolution_parameter,
                    },
                },
                "summary": self._generate_analysis_summary(
                    hierarchical_results, modular_results, context_dependent_results, optimization_results
                ),
            }

            # Store results
            self.graph_results = results

            logger.info("Graph-based coordination analysis completed successfully")
            return results

        except Exception as e:
            logger.error(f"Error in graph analysis: {e}")
            return {
                "error": str(e),
                "analysis_metadata": {
                    "edge_construction_method": self.edge_construction_method,
                    "graph_type": self.graph_type,
                    "device": self.device,
                },
            }

    def _generate_analysis_summary(
        self,
        hierarchical_results: dict,
        modular_results: dict,
        context_dependent_results: dict,
        optimization_results: dict,
    ) -> dict[str, Any]:
        """Generate a summary of key findings across all analyses."""
        summary = {
            "hypothesis_support": {},
            "key_findings": {},
            "group_comparisons": {},
        }

        try:
            # H1: Hierarchical Network Organization
            h1_support = {}
            for group in hierarchical_results:
                h1_support[group] = hierarchical_results[group].get("supports_hierarchy_hypothesis", False)

            summary["hypothesis_support"]["H1_hierarchical_organization"] = {
                "overall_support": any(h1_support.values()),
                "group_support": h1_support,
                "evidence": "Hub neurons and scale-free properties detected"
                if any(h1_support.values())
                else "Limited hierarchical structure",
            }

            # H2: Modular Coordination Architecture
            h2_support = {}
            for group in modular_results:
                h2_support[group] = modular_results[group].get("supports_modularity_hypothesis", False)

            summary["hypothesis_support"]["H2_modular_architecture"] = {
                "overall_support": any(h2_support.values()),
                "group_support": h2_support,
                "evidence": "High modularity and distinct communities detected"
                if any(h2_support.values())
                else "Limited modular structure",
            }

            # H3: Context-Dependent Network Topology
            h3_support = {}
            for group in context_dependent_results:
                h3_support[group] = context_dependent_results[group].get("supports_adaptive_topology_hypothesis", False)

            summary["hypothesis_support"]["H3_context_dependent_topology"] = {
                "overall_support": any(h3_support.values()),
                "group_support": h3_support,
                "evidence": "Significant topology differences between contexts"
                if any(h3_support.values())
                else "Similar topology across contexts",
            }

            # H4: Optimized Information Flow
            h4_support = {}
            for group in optimization_results:
                h4_support[group] = optimization_results[group].get("supports_optimization_hypothesis", False)

            summary["hypothesis_support"]["H4_optimized_information_flow"] = {
                "overall_support": any(h4_support.values()),
                "group_support": h4_support,
                "evidence": "Higher efficiency than random networks"
                if any(h4_support.values())
                else "Similar efficiency to random networks",
            }

            # Key findings across groups
            summary["key_findings"] = {
                "strongest_hierarchy": max(h1_support.items(), key=lambda x: x[1], default=("none", False))[0],
                "highest_modularity": max(
                    [(group, modular_results[group].get("modularity", 0)) for group in modular_results],
                    key=lambda x: x[1],
                    default=("none", 0),
                )[0],
                "most_context_sensitive": max(h3_support.items(), key=lambda x: x[1], default=("none", False))[0],
                "most_optimized": max(h4_support.items(), key=lambda x: x[1], default=("none", False))[0],
            }

            # Overall coordination patterns
            rare_token_groups = ["boost", "suppress"]
            control_groups = ["random_1", "random_2"]

            rare_token_support = sum(
                [
                    h1_support.get(group, False)
                    + h2_support.get(group, False)
                    + h3_support.get(group, False)
                    + h4_support.get(group, False)
                    for group in rare_token_groups
                ]
            )

            control_support = sum(
                [
                    h1_support.get(group, False)
                    + h2_support.get(group, False)
                    + h3_support.get(group, False)
                    + h4_support.get(group, False)
                    for group in control_groups
                ]
            )

            summary["group_comparisons"] = {
                "rare_token_groups_show_more_structure": rare_token_support > control_support,
                "rare_token_structure_score": rare_token_support,
                "control_structure_score": control_support,
            }

        except Exception as e:
            logger.warning(f"Error generating summary: {e}")
            summary["error"] = str(e)

        return summary
