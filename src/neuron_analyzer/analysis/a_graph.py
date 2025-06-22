import logging
import warnings
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import mutual_info_regression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)

#######################################################
# Threshold selection


class ThresholdSelector:
    """Step 1: Automatic threshold selection for graph-based neural coordination analysis."""

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
        self.device = device
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
        """Method 1: Statistical significance-based threshold selection."""
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

    def run_all_analyses(self, method: str = "comprehensive") -> dict[str, Any]:
        """Main method to run threshold selection."""
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

    def summarize_results(self, results: dict[str, Any]) -> str:
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
    """Comprehensive graph-based analysis for neural coordination patterns with signed networks."""

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
        preserve_edge_signs: bool = True,  # NEW: Controls whether to keep negative weights
        significance_level: float = 0.05,
        max_lag: int = 3,
        # Community detection parameters
        community_algorithm: str = "louvain",  # "louvain", "spectral", "infomap"
        resolution_parameter: float = 1.0,
        # Statistical parameters
        n_bootstrap: int = 1000,
        n_permutations: int = 1000,
    ):
        """Initialize the graph-based coordination analyzer with signed network support."""
        # Device and precision setup
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mixed_precision = use_mixed_precision
        self.dtype = torch.float16 if use_mixed_precision else torch.float32

        # Store parameters
        self.correlation_threshold = correlation_threshold
        self.mi_threshold = mi_threshold
        self.edge_construction_method = edge_construction_method
        self.graph_type = graph_type
        self.preserve_edge_signs = preserve_edge_signs  # NEW
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

        # Enhanced logging for signed networks
        signed_status = "signed" if self.preserve_edge_signs else "unsigned"
        logger.info(
            f"Initialized Graph analyzer with {len(self.token_contexts)} contexts, "
            f"device: {self.device}, edge method: {self.edge_construction_method}, "
            f"graph type: {self.graph_type}, signed network: {signed_status}"
        )

    def _generate_random_groups(self) -> tuple[list[np.ndarray], list[list[int]]]:
        """Generate non-overlapping random neuron groups with enhanced validation."""
        group_size = max(len(self.boost_neuron_indices), len(self.suppress_neuron_indices))
        special_indices = set(self.boost_neuron_indices + self.suppress_neuron_indices + self.excluded_neuron_indices)

        # Get available indices
        available_indices = [idx for idx in self.all_neuron_indices if idx not in special_indices]

        if len(available_indices) < group_size * self.num_random_groups:
            logger.warning(
                f"Insufficient neurons for {self.num_random_groups} random groups. "
                f"Available: {len(available_indices)}, needed: {group_size * self.num_random_groups}"
            )
            # Adjust group size or number of groups
            if len(available_indices) >= self.num_random_groups:
                group_size = len(available_indices) // self.num_random_groups
                logger.info(f"Adjusted random group size to {group_size}")
            else:
                raise ValueError("Not enough neurons available for random groups")

        # Sample random groups with improved reproducibility
        np.random.seed(42)  # For reproducibility
        random_indices = []
        remaining_indices = available_indices.copy()

        for i in range(self.num_random_groups):
            actual_group_size = min(group_size, len(remaining_indices))
            if actual_group_size == 0:
                break

            group = np.random.choice(remaining_indices, size=actual_group_size, replace=False)
            random_indices.append(group.tolist())
            remaining_indices = [idx for idx in remaining_indices if idx not in group]

        # Create activation matrices for random groups
        random_groups = [self._create_activation_matrix(indices) for indices in random_indices]

        logger.info(
            f"Generated {len(random_indices)} random groups with sizes: {[len(indices) for indices in random_indices]}"
        )

        return random_groups, random_indices

    def _create_activation_matrix(self, neuron_indices: list[int]) -> np.ndarray:
        """Create activation matrix where rows are contexts and columns are neurons."""
        if not neuron_indices:
            logger.warning("Empty neuron indices provided")
            return np.array([]).reshape(len(self.token_contexts), 0)

        filtered_data = self.data[self.data[self.component_column].isin(neuron_indices)]

        if filtered_data.empty:
            logger.warning(f"No data found for neuron indices: {neuron_indices}")
            return np.zeros((len(self.token_contexts), len(neuron_indices)))

        pivot_table = filtered_data.pivot_table(
            index="token_context_id", columns=self.component_column, values=self.activation_column, aggfunc="first"
        )

        # Ensure all contexts are represented
        pivot_table = pivot_table.reindex(index=self.token_contexts, columns=neuron_indices, fill_value=0)

        # Handle any remaining NaN values
        pivot_table = pivot_table.fillna(0)

        matrix = pivot_table.values
        logger.debug(f"Created activation matrix shape: {matrix.shape} for {len(neuron_indices)} neurons")

        return matrix

    def _create_activation_tensors(self) -> None:
        """Create PyTorch tensors for all neuron groups with enhanced error handling."""
        try:
            # Create matrices with validation
            boost_matrix = self._create_activation_matrix(self.boost_neuron_indices)
            suppress_matrix = self._create_activation_matrix(self.suppress_neuron_indices)

            # Handle variable number of random groups
            random_matrices = {}
            for i, indices in enumerate(self.random_indices):
                random_matrices[f"random_{i + 1}"] = self._create_activation_matrix(indices)

            # Convert to tensors with proper device placement
            self.activation_tensors = {
                "boost": torch.tensor(boost_matrix, dtype=self.dtype).to(self.device),
                "suppress": torch.tensor(suppress_matrix, dtype=self.dtype).to(self.device),
            }

            # Add random group tensors
            for key, matrix in random_matrices.items():
                self.activation_tensors[key] = torch.tensor(matrix, dtype=self.dtype).to(self.device)

            # Store neuron indices for reference with enhanced metadata
            self.neuron_indices = {
                "boost": self.boost_neuron_indices,
                "suppress": self.suppress_neuron_indices,
            }

            # Add random group indices
            for i, indices in enumerate(self.random_indices):
                self.neuron_indices[f"random_{i + 1}"] = indices

            # Log tensor information
            for group_name, tensor in self.activation_tensors.items():
                logger.debug(
                    f"Group '{group_name}': tensor shape {tensor.shape}, device {tensor.device}, dtype {tensor.dtype}"
                )

        except Exception as e:
            logger.error(f"Error creating activation tensors: {e}")
            raise

    def _create_rare_token_mask(self, rare_token_mask: np.ndarray | None) -> np.ndarray:
        """Create or validate rare token mask with enhanced validation."""
        if rare_token_mask is not None:
            if len(rare_token_mask) != len(self.token_contexts):
                raise ValueError(
                    f"Rare token mask length ({len(rare_token_mask)}) must match "
                    f"number of contexts ({len(self.token_contexts)})"
                )

            # Validate mask values
            if not np.all(np.isin(rare_token_mask, [0, 1, True, False])):
                logger.warning("Rare token mask contains non-boolean values, converting to boolean")
                rare_token_mask = rare_token_mask.astype(bool)

            logger.info(f"Using provided rare token mask: {np.sum(rare_token_mask)} rare contexts")
            return rare_token_mask.astype(bool)

        # Create enhanced frequency-based mask if not provided
        logger.warning("No rare token mask provided, using frequency-based approximation")

        try:
            token_counts = self.data.groupby(self.token_column).size()

            # Use more sophisticated rare token detection
            if len(token_counts) > 4:
                # Use interquartile range for better threshold
                q1 = np.percentile(token_counts, 25)
                q3 = np.percentile(token_counts, 75)
                iqr = q3 - q1
                rare_threshold = max(1, q1 - 1.5 * iqr)  # Outlier detection
            else:
                rare_threshold = np.percentile(token_counts, 25)  # Bottom 25% as "rare"

            rare_tokens = set(token_counts[token_counts <= rare_threshold].index)

            # Create mask based on token extraction from context ID
            mask = np.array([token_context.split("_")[0] in rare_tokens for token_context in self.token_contexts])

            logger.info(
                f"Created frequency-based rare token mask: {np.sum(mask)} rare contexts "
                f"out of {len(mask)} total (threshold: {rare_threshold})"
            )

            return mask

        except Exception as e:
            logger.error(f"Error creating rare token mask: {e}")
            # Fallback to simple split
            mask = np.zeros(len(self.token_contexts), dtype=bool)
            mask[: len(self.token_contexts) // 4] = True  # First 25% as rare
            return mask


#######################################################
# Threshold selection


class GraphCoordinationAnalyzer:
    """Comprehensive graph-based analysis for neural coordination patterns with signed networks."""

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
        preserve_edge_signs: bool = True,  # NEW: Controls whether to keep negative weights
        significance_level: float = 0.05,
        max_lag: int = 3,
        # Community detection parameters
        community_algorithm: str = "louvain",  # "louvain", "spectral", "infomap"
        resolution_parameter: float = 1.0,
        # Statistical parameters
        n_bootstrap: int = 1000,
        n_permutations: int = 1000,
    ):
        """Initialize the graph-based coordination analyzer with signed network support."""
        # Device and precision setup
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mixed_precision = use_mixed_precision
        self.dtype = torch.float16 if use_mixed_precision else torch.float32

        # Store parameters
        self.correlation_threshold = correlation_threshold
        self.mi_threshold = mi_threshold
        self.edge_construction_method = edge_construction_method
        self.graph_type = graph_type
        self.preserve_edge_signs = preserve_edge_signs  # NEW
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

        # Enhanced logging for signed networks
        signed_status = "signed" if self.preserve_edge_signs else "unsigned"
        logger.info(
            f"Initialized Graph analyzer with {len(self.token_contexts)} contexts, "
            f"device: {self.device}, edge method: {self.edge_construction_method}, "
            f"graph type: {self.graph_type}, signed network: {signed_status}"
        )

    def _generate_random_groups(self) -> tuple[list[np.ndarray], list[list[int]]]:
        """Generate non-overlapping random neuron groups with enhanced validation."""
        group_size = max(len(self.boost_neuron_indices), len(self.suppress_neuron_indices))
        special_indices = set(self.boost_neuron_indices + self.suppress_neuron_indices + self.excluded_neuron_indices)

        # Get available indices
        available_indices = [idx for idx in self.all_neuron_indices if idx not in special_indices]

        if len(available_indices) < group_size * self.num_random_groups:
            logger.warning(
                f"Insufficient neurons for {self.num_random_groups} random groups. "
                f"Available: {len(available_indices)}, needed: {group_size * self.num_random_groups}"
            )
            # Adjust group size or number of groups
            if len(available_indices) >= self.num_random_groups:
                group_size = len(available_indices) // self.num_random_groups
                logger.info(f"Adjusted random group size to {group_size}")
            else:
                raise ValueError("Not enough neurons available for random groups")

        # Sample random groups with improved reproducibility
        np.random.seed(42)  # For reproducibility
        random_indices = []
        remaining_indices = available_indices.copy()

        for i in range(self.num_random_groups):
            actual_group_size = min(group_size, len(remaining_indices))
            if actual_group_size == 0:
                break

            group = np.random.choice(remaining_indices, size=actual_group_size, replace=False)
            random_indices.append(group.tolist())
            remaining_indices = [idx for idx in remaining_indices if idx not in group]

        # Create activation matrices for random groups
        random_groups = [self._create_activation_matrix(indices) for indices in random_indices]

        logger.info(
            f"Generated {len(random_indices)} random groups with sizes: {[len(indices) for indices in random_indices]}"
        )

        return random_groups, random_indices

    def _create_activation_matrix(self, neuron_indices: list[int]) -> np.ndarray:
        """Create activation matrix where rows are contexts and columns are neurons."""
        if not neuron_indices:
            logger.warning("Empty neuron indices provided")
            return np.array([]).reshape(len(self.token_contexts), 0)

        filtered_data = self.data[self.data[self.component_column].isin(neuron_indices)]

        if filtered_data.empty:
            logger.warning(f"No data found for neuron indices: {neuron_indices}")
            return np.zeros((len(self.token_contexts), len(neuron_indices)))

        pivot_table = filtered_data.pivot_table(
            index="token_context_id", columns=self.component_column, values=self.activation_column, aggfunc="first"
        )

        # Ensure all contexts are represented
        pivot_table = pivot_table.reindex(index=self.token_contexts, columns=neuron_indices, fill_value=0)

        # Handle any remaining NaN values
        pivot_table = pivot_table.fillna(0)

        matrix = pivot_table.values
        logger.debug(f"Created activation matrix shape: {matrix.shape} for {len(neuron_indices)} neurons")

        return matrix

    def _create_activation_tensors(self) -> None:
        """Create PyTorch tensors for all neuron groups with enhanced error handling."""
        try:
            # Create matrices with validation
            boost_matrix = self._create_activation_matrix(self.boost_neuron_indices)
            suppress_matrix = self._create_activation_matrix(self.suppress_neuron_indices)

            # Handle variable number of random groups
            random_matrices = {}
            for i, indices in enumerate(self.random_indices):
                random_matrices[f"random_{i + 1}"] = self._create_activation_matrix(indices)

            # Convert to tensors with proper device placement
            self.activation_tensors = {
                "boost": torch.tensor(boost_matrix, dtype=self.dtype).to(self.device),
                "suppress": torch.tensor(suppress_matrix, dtype=self.dtype).to(self.device),
            }

            # Add random group tensors
            for key, matrix in random_matrices.items():
                self.activation_tensors[key] = torch.tensor(matrix, dtype=self.dtype).to(self.device)

            # Store neuron indices for reference with enhanced metadata
            self.neuron_indices = {
                "boost": self.boost_neuron_indices,
                "suppress": self.suppress_neuron_indices,
            }

            # Add random group indices
            for i, indices in enumerate(self.random_indices):
                self.neuron_indices[f"random_{i + 1}"] = indices

            # Log tensor information
            for group_name, tensor in self.activation_tensors.items():
                logger.debug(
                    f"Group '{group_name}': tensor shape {tensor.shape}, device {tensor.device}, dtype {tensor.dtype}"
                )

        except Exception as e:
            logger.error(f"Error creating activation tensors: {e}")
            raise

    def _create_rare_token_mask(self, rare_token_mask: np.ndarray | None) -> np.ndarray:
        """Create or validate rare token mask with enhanced validation."""
        if rare_token_mask is not None:
            if len(rare_token_mask) != len(self.token_contexts):
                raise ValueError(
                    f"Rare token mask length ({len(rare_token_mask)}) must match "
                    f"number of contexts ({len(self.token_contexts)})"
                )

            # Validate mask values
            if not np.all(np.isin(rare_token_mask, [0, 1, True, False])):
                logger.warning("Rare token mask contains non-boolean values, converting to boolean")
                rare_token_mask = rare_token_mask.astype(bool)

            logger.info(f"Using provided rare token mask: {np.sum(rare_token_mask)} rare contexts")
            return rare_token_mask.astype(bool)

        # Create enhanced frequency-based mask if not provided
        logger.warning("No rare token mask provided, using frequency-based approximation")

        try:
            token_counts = self.data.groupby(self.token_column).size()

            # Use more sophisticated rare token detection
            if len(token_counts) > 4:
                # Use interquartile range for better threshold
                q1 = np.percentile(token_counts, 25)
                q3 = np.percentile(token_counts, 75)
                iqr = q3 - q1
                rare_threshold = max(1, q1 - 1.5 * iqr)  # Outlier detection
            else:
                rare_threshold = np.percentile(token_counts, 25)  # Bottom 25% as "rare"

            rare_tokens = set(token_counts[token_counts <= rare_threshold].index)

            # Create mask based on token extraction from context ID
            mask = np.array([token_context.split("_")[0] in rare_tokens for token_context in self.token_contexts])

            logger.info(
                f"Created frequency-based rare token mask: {np.sum(mask)} rare contexts "
                f"out of {len(mask)} total (threshold: {rare_threshold})"
            )

            return mask

        except Exception as e:
            logger.error(f"Error creating rare token mask: {e}")
            # Fallback to simple split
            mask = np.zeros(len(self.token_contexts), dtype=bool)
            mask[: len(self.token_contexts) // 4] = True  # First 25% as rare
            return mask

    def _estimate_mutual_information(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Estimate mutual information between two variables with enhanced robustness."""
        try:
            # Handle edge cases
            if len(X) == 0 or len(Y) == 0:
                return 0.0

            # Remove any infinite or NaN values
            valid_mask = np.isfinite(X) & np.isfinite(Y)
            if np.sum(valid_mask) < 3:  # Need at least 3 points
                return 0.0

            X_clean = X[valid_mask]
            Y_clean = Y[valid_mask]

            # Check for constant variables
            if np.std(X_clean) == 0 or np.std(Y_clean) == 0:
                return 0.0

            X_reshaped = X_clean.reshape(-1, 1) if X_clean.ndim == 1 else X_clean
            mi = mutual_info_regression(X_reshaped, Y_clean, discrete_features=False, random_state=42)
            return float(mi[0] if len(mi) == 1 else np.mean(mi))
        except Exception as e:
            logger.debug(f"MI estimation failed: {e}")
            return 0.0

    def _construct_graph_edges(self, data: np.ndarray, context_mask: np.ndarray | None = None) -> np.ndarray:
        """Construct edge weights with signed network support and enhanced methods."""
        if context_mask is not None:
            data = data[context_mask]

        n_neurons = data.shape[1]
        if n_neurons < 2:
            return np.zeros((n_neurons, n_neurons))

        edge_matrix = np.zeros((n_neurons, n_neurons))

        try:
            if self.edge_construction_method == "correlation":
                # Enhanced correlation-based edges with sign preservation
                corr_matrix = np.corrcoef(data.T)

                # Handle NaN correlations (constant variables)
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)

                if self.preserve_edge_signs and self.graph_type in ["signed", "weighted"]:
                    # Preserve correlation signs
                    edge_matrix = corr_matrix.copy()
                    # Apply threshold to absolute values but keep signs
                    threshold_mask = np.abs(corr_matrix) > self.correlation_threshold
                    edge_matrix[~threshold_mask] = 0.0
                else:
                    # Use absolute values (original behavior)
                    edge_matrix = np.abs(corr_matrix)

            elif self.edge_construction_method == "mi":
                # Mutual information-based edges (always positive)
                for i in range(n_neurons):
                    for j in range(i + 1, n_neurons):
                        mi_val = self._estimate_mutual_information(data[:, i], data[:, j])
                        edge_matrix[i, j] = mi_val
                        edge_matrix[j, i] = mi_val

            elif self.edge_construction_method == "hybrid":
                # Enhanced hybrid approach with sign awareness
                corr_matrix = np.corrcoef(data.T)
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)

                mi_matrix = np.zeros((n_neurons, n_neurons))
                for i in range(n_neurons):
                    for j in range(i + 1, n_neurons):
                        mi_val = self._estimate_mutual_information(data[:, i], data[:, j])
                        mi_matrix[i, j] = mi_val
                        mi_matrix[j, i] = mi_val

                if self.preserve_edge_signs and self.graph_type in ["signed", "weighted"]:
                    # Combine correlation signs with MI magnitudes
                    corr_signs = np.sign(corr_matrix)
                    mi_magnitudes = mi_matrix
                    edge_matrix = corr_signs * mi_magnitudes

                    # Apply hybrid threshold
                    corr_threshold_mask = np.abs(corr_matrix) > self.correlation_threshold
                    mi_threshold_mask = mi_matrix > self.mi_threshold
                    combined_mask = corr_threshold_mask & mi_threshold_mask
                    edge_matrix[~combined_mask] = 0.0
                else:
                    # Original weighted average approach
                    edge_matrix = 0.6 * np.abs(corr_matrix) + 0.4 * mi_matrix

            # Apply thresholding based on graph type
            if self.graph_type == "binary":
                if self.edge_construction_method == "correlation":
                    threshold = self.correlation_threshold
                    if self.preserve_edge_signs:
                        # Binary signed: -1, 0, +1
                        edge_matrix = np.sign(edge_matrix) * (np.abs(edge_matrix) > threshold).astype(float)
                    else:
                        edge_matrix = (np.abs(edge_matrix) > threshold).astype(float)
                else:
                    threshold = self.mi_threshold
                    edge_matrix = (edge_matrix > threshold).astype(float)

            # Zero out diagonal
            np.fill_diagonal(edge_matrix, 0)

            # Log edge statistics
            n_edges = np.sum(edge_matrix != 0)
            if self.preserve_edge_signs:
                n_positive = np.sum(edge_matrix > 0)
                n_negative = np.sum(edge_matrix < 0)
                logger.debug(
                    f"Constructed signed graph: {n_edges} edges ({n_positive} positive, {n_negative} negative)"
                )
            else:
                logger.debug(f"Constructed unsigned graph: {n_edges} edges")

        except Exception as e:
            logger.error(f"Error in edge construction: {e}")
            edge_matrix = np.zeros((n_neurons, n_neurons))

        return edge_matrix

    def _construct_networkx_graph(self, edge_matrix: np.ndarray) -> nx.Graph:
        """Convert edge matrix to NetworkX graph with signed network support."""
        try:
            if self.preserve_edge_signs and self.graph_type in ["signed", "weighted"]:
                # Create signed graph (can have negative weights)
                if self.graph_type == "binary":
                    # For binary signed graphs, use non-zero edges regardless of sign
                    G = nx.from_numpy_array(edge_matrix != 0)
                    # Add sign information as edge attributes
                    for i, j, data in G.edges(data=True):
                        data["weight"] = edge_matrix[i, j]
                        data["sign"] = np.sign(edge_matrix[i, j])
                else:
                    # For weighted signed graphs, use actual weights
                    G = nx.from_numpy_array(edge_matrix)
                    # Add sign information as edge attributes
                    for i, j, data in G.edges(data=True):
                        data["sign"] = np.sign(data["weight"])
            # Original unsigned graph construction
            elif self.graph_type == "binary":
                G = nx.from_numpy_array(edge_matrix > 0)
            else:
                G = nx.from_numpy_array(edge_matrix)

            # Add graph metadata
            G.graph["signed"] = self.preserve_edge_signs
            G.graph["edge_method"] = self.edge_construction_method
            G.graph["graph_type"] = self.graph_type

            return G

        except Exception as e:
            logger.error(f"Error constructing NetworkX graph: {e}")
            # Return empty graph as fallback
            return nx.Graph()

    def _compute_centrality_measures(self, G: nx.Graph) -> dict[str, Any]:
        """Compute centrality measures with signed network support."""
        centralities = {}
        n_nodes = len(G.nodes())

        if n_nodes == 0:
            return {
                "degree": [],
                "betweenness": [],
                "eigenvector": [],
                "pagerank": [],
                "closeness": [],
                "positive_degree": [],
                "negative_degree": [],
                "signed_centrality_balance": [],
            }

        try:
            # Basic centrality measures
            centralities["degree"] = list(nx.degree_centrality(G).values())
            centralities["betweenness"] = list(nx.betweenness_centrality(G).values())
            centralities["closeness"] = list(nx.closeness_centrality(G).values())

            # Eigenvector centrality with robust handling
            try:
                if nx.is_connected(G) and not self._has_negative_weights(G):
                    centralities["eigenvector"] = list(nx.eigenvector_centrality(G, max_iter=1000).values())
                else:
                    # Use absolute weights for eigenvector centrality with negative weights
                    G_abs = self._create_absolute_weight_graph(G)
                    if nx.is_connected(G_abs):
                        centralities["eigenvector"] = list(nx.eigenvector_centrality(G_abs, max_iter=1000).values())
                    else:
                        centralities["eigenvector"] = [0.0] * n_nodes
            except (nx.PowerIterationFailedConvergence, np.linalg.LinAlgError):
                centralities["eigenvector"] = [0.0] * n_nodes

            # PageRank with signed network handling
            try:
                if self._has_negative_weights(G):
                    # Use absolute weights for PageRank with negative weights
                    G_abs = self._create_absolute_weight_graph(G)
                    centralities["pagerank"] = list(nx.pagerank(G_abs, max_iter=1000).values())
                else:
                    centralities["pagerank"] = list(nx.pagerank(G, max_iter=1000).values())
            except (nx.PowerIterationFailedConvergence, ZeroDivisionError):
                centralities["pagerank"] = [1.0 / n_nodes] * n_nodes

            # Signed network specific centralities
            if self.preserve_edge_signs and G.graph.get("signed", False):
                pos_deg, neg_deg, balance = self._compute_signed_degree_centralities(G)
                centralities["positive_degree"] = pos_deg
                centralities["negative_degree"] = neg_deg
                centralities["signed_centrality_balance"] = balance
            else:
                centralities["positive_degree"] = centralities["degree"].copy()
                centralities["negative_degree"] = [0.0] * n_nodes
                centralities["signed_centrality_balance"] = [1.0] * n_nodes

        except Exception as e:
            logger.warning(f"Error computing centralities: {e}")
            centralities = {
                "degree": [0.0] * n_nodes,
                "betweenness": [0.0] * n_nodes,
                "eigenvector": [0.0] * n_nodes,
                "pagerank": [0.0] * n_nodes,
                "closeness": [0.0] * n_nodes,
                "positive_degree": [0.0] * n_nodes,
                "negative_degree": [0.0] * n_nodes,
                "signed_centrality_balance": [1.0] * n_nodes,
            }

        return centralities

    def _has_negative_weights(self, G: nx.Graph) -> bool:
        """Check if graph has negative edge weights."""
        try:
            for _, _, data in G.edges(data=True):
                if data.get("weight", 1.0) < 0:
                    return True
            return False
        except:
            return False

    def _create_absolute_weight_graph(self, G: nx.Graph) -> nx.Graph:
        """Create a copy of the graph with absolute edge weights."""
        G_abs = G.copy()
        for i, j, data in G_abs.edges(data=True):
            data["weight"] = abs(data.get("weight", 1.0))
        return G_abs

    def _compute_signed_degree_centralities(self, G: nx.Graph) -> tuple[list[float], list[float], list[float]]:
        """Compute positive and negative degree centralities for signed networks."""
        try:
            n_nodes = len(G.nodes())
            positive_degrees = [0.0] * n_nodes
            negative_degrees = [0.0] * n_nodes

            # Calculate positive and negative degree sums
            for node in G.nodes():
                pos_sum = 0.0
                neg_sum = 0.0

                for neighbor in G.neighbors(node):
                    weight = G[node][neighbor].get("weight", 1.0)
                    if weight > 0:
                        pos_sum += weight
                    elif weight < 0:
                        neg_sum += abs(weight)

                positive_degrees[node] = pos_sum
                negative_degrees[node] = neg_sum

            # Normalize by maximum possible degree
            max_pos = max(positive_degrees) if positive_degrees else 1.0
            max_neg = max(negative_degrees) if negative_degrees else 1.0

            if max_pos > 0:
                positive_degrees = [d / max_pos for d in positive_degrees]
            if max_neg > 0:
                negative_degrees = [d / max_neg for d in negative_degrees]

            # Compute balance ratio (positive / total)
            balance = []
            for i in range(n_nodes):
                total = positive_degrees[i] + negative_degrees[i]
                if total > 0:
                    balance.append(positive_degrees[i] / total)
                else:
                    balance.append(0.5)  # Neutral balance for isolated nodes

            return positive_degrees, negative_degrees, balance

        except Exception as e:
            logger.warning(f"Error computing signed degree centralities: {e}")
            n_nodes = len(G.nodes())
            return [0.0] * n_nodes, [0.0] * n_nodes, [0.5] * n_nodes

    def _detect_communities(self, G: nx.Graph, edge_matrix: np.ndarray) -> dict[str, Any]:
        """Detect communities with signed network support and enhanced algorithms."""
        communities = {}
        n_nodes = len(G.nodes())

        if n_nodes < 3:
            return {
                "louvain": [0] * n_nodes,
                "spectral": [0] * n_nodes,
                "modularity": 0.0,
                "signed_modularity": 0.0,
                "n_communities": 1,
                "community_balance_scores": [0.5] * n_nodes,
                "positive_modularity": 0.0,
                "negative_modularity": 0.0,
            }

        try:
            # Enhanced community detection for signed networks
            if self.preserve_edge_signs and G.graph.get("signed", False):
                communities.update(self._detect_signed_communities(G, edge_matrix))
            else:
                # Standard unsigned community detection
                communities.update(self._detect_unsigned_communities(G, edge_matrix))

            # Compute modularity measures
            if "louvain" in communities:
                communities["modularity"] = self._compute_standard_modularity(G, communities["louvain"])

                if self.preserve_edge_signs:
                    communities["signed_modularity"] = self._compute_signed_modularity(G, communities["louvain"])
                    pos_mod, neg_mod = self._compute_positive_negative_modularity(G, communities["louvain"])
                    communities["positive_modularity"] = pos_mod
                    communities["negative_modularity"] = neg_mod
                    communities["community_balance_scores"] = self._compute_community_balance_scores(
                        G, communities["louvain"]
                    )
                else:
                    communities["signed_modularity"] = communities["modularity"]
                    communities["positive_modularity"] = communities["modularity"]
                    communities["negative_modularity"] = 0.0
                    communities["community_balance_scores"] = [0.5] * n_nodes
            else:
                communities["modularity"] = 0.0
                communities["signed_modularity"] = 0.0
                communities["positive_modularity"] = 0.0
                communities["negative_modularity"] = 0.0
                communities["community_balance_scores"] = [0.5] * n_nodes

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
                "signed_modularity": 0.0,
                "n_communities": 1,
                "community_balance_scores": [0.5] * n_nodes,
                "positive_modularity": 0.0,
                "negative_modularity": 0.0,
            }

        return communities

    def _detect_signed_communities(self, G: nx.Graph, edge_matrix: np.ndarray) -> dict[str, Any]:
        """Detect communities in signed networks."""
        communities = {}

        try:
            # Separate positive and negative subgraphs
            G_positive = self._extract_positive_subgraph(G)
            G_negative = self._extract_negative_subgraph(G)

            # Community detection on positive subgraph
            if G_positive.number_of_edges() > 0:
                pos_communities = self._detect_unsigned_communities(G_positive, None)
                communities["positive_subgraph_communities"] = pos_communities.get("louvain", [0] * len(G.nodes()))
            else:
                communities["positive_subgraph_communities"] = [0] * len(G.nodes())

            # Try signed-aware Louvain if available
            try:
                import community as community_louvain

                # Use absolute weights for standard Louvain, then post-process
                G_abs = self._create_absolute_weight_graph(G)
                partition = community_louvain.best_partition(G_abs, resolution=self.resolution_parameter)
                communities["louvain"] = [partition.get(i, 0) for i in range(len(G.nodes()))]
            except ImportError:
                # Fallback to spectral clustering
                communities["louvain"] = self._spectral_clustering_signed(edge_matrix)

            # Spectral clustering for signed networks
            communities["spectral"] = self._spectral_clustering_signed(edge_matrix)

        except Exception as e:
            logger.warning(f"Error in signed community detection: {e}")
            n_nodes = len(G.nodes())
            communities = {
                "louvain": [0] * n_nodes,
                "spectral": [0] * n_nodes,
                "positive_subgraph_communities": [0] * n_nodes,
            }

        return communities

    def _detect_unsigned_communities(self, G: nx.Graph, edge_matrix: np.ndarray) -> dict[str, Any]:
        """Detect communities in unsigned networks (original method)."""
        communities = {}

        try:
            # Louvain algorithm
            if self.community_algorithm in ["louvain", "all"]:
                try:
                    import community as community_louvain

                    partition = community_louvain.best_partition(G, resolution=self.resolution_parameter)
                    communities["louvain"] = [partition.get(i, 0) for i in range(len(G.nodes()))]
                except ImportError:
                    # Fallback to spectral clustering
                    communities["louvain"] = self._spectral_clustering(edge_matrix)

            # Spectral clustering
            if self.community_algorithm in ["spectral", "all"]:
                communities["spectral"] = self._spectral_clustering(edge_matrix)

        except Exception as e:
            logger.warning(f"Error in unsigned community detection: {e}")
            n_nodes = len(G.nodes())
            communities = {
                "louvain": [0] * n_nodes,
                "spectral": [0] * n_nodes,
            }

        return communities

    def _extract_positive_subgraph(self, G: nx.Graph) -> nx.Graph:
        """Extract subgraph with only positive edges."""
        G_pos = nx.Graph()
        G_pos.add_nodes_from(G.nodes())

        for i, j, data in G.edges(data=True):
            weight = data.get("weight", 1.0)
            if weight > 0:
                G_pos.add_edge(i, j, **data)

        return G_pos

    def _extract_negative_subgraph(self, G: nx.Graph) -> nx.Graph:
        """Extract subgraph with only negative edges (converted to positive)."""
        G_neg = nx.Graph()
        G_neg.add_nodes_from(G.nodes())

        for i, j, data in G.edges(data=True):
            weight = data.get("weight", 1.0)
            if weight < 0:
                # Convert negative weight to positive for analysis
                new_data = data.copy()
                new_data["weight"] = abs(weight)
                G_neg.add_edge(i, j, **new_data)

        return G_neg

    def _spectral_clustering_signed(self, edge_matrix: np.ndarray, n_clusters: int | None = None) -> list[int]:
        """Perform spectral clustering on signed networks."""
        try:
            if n_clusters is None:
                # Enhanced eigengap heuristic for signed networks
                # Use the signed Laplacian matrix
                signed_laplacian = self._compute_signed_laplacian(edge_matrix)
                eigenvals = np.linalg.eigvals(signed_laplacian)
                eigenvals = np.sort(eigenvals)

                if len(eigenvals) > 3:
                    # Look for the largest gap in eigenvalues
                    gaps = np.diff(eigenvals[: min(10, len(eigenvals))])
                    n_clusters = np.argmax(gaps) + 2
                else:
                    n_clusters = 2

            n_clusters = min(n_clusters, edge_matrix.shape[0] - 1)
            n_clusters = max(n_clusters, 2)

            # Use signed spectral clustering
            affinity_matrix = self._compute_signed_affinity_matrix(edge_matrix)
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity="precomputed",
                random_state=42,
                eigen_solver="arpack",  # More stable for signed matrices
            )
            labels = clustering.fit_predict(affinity_matrix)
            return labels.tolist()

        except Exception as e:
            logger.warning(f"Signed spectral clustering failed: {e}")
            # Fallback to unsigned spectral clustering
            return self._spectral_clustering(np.abs(edge_matrix))

    def _compute_signed_laplacian(self, edge_matrix: np.ndarray) -> np.ndarray:
        """Compute signed Laplacian matrix."""
        try:
            # Signed degree matrix
            degrees = np.sum(np.abs(edge_matrix), axis=1)
            D = np.diag(degrees)

            # Signed Laplacian: L = D - A (where A can have negative values)
            L = D - edge_matrix

            return L
        except Exception as e:
            logger.warning(f"Error computing signed Laplacian: {e}")
            return np.eye(edge_matrix.shape[0])

    def _compute_signed_affinity_matrix(self, edge_matrix: np.ndarray) -> np.ndarray:
        """Compute affinity matrix for signed spectral clustering."""
        try:
            # For signed networks, we need to handle negative weights carefully
            # One approach: use absolute values but encode sign information
            abs_matrix = np.abs(edge_matrix)

            # Alternative: use the approach from signed spectral clustering literature
            # Convert to similarity matrix using exponential of negative squared distance
            # But preserve sign information

            # Simple approach: use absolute values for clustering
            # More sophisticated approaches could use balance theory
            return abs_matrix

        except Exception as e:
            logger.warning(f"Error computing signed affinity matrix: {e}")
            return np.abs(edge_matrix)

    def _compute_standard_modularity(self, G: nx.Graph, partition: list[int]) -> float:
        """Compute standard modularity."""
        try:
            if len(set(partition)) <= 1:
                return 0.0

            partition_dict = {i: partition[i] for i in range(len(partition))}
            communities_sets = [
                set([k for k, v in partition_dict.items() if v == c]) for c in set(partition_dict.values())
            ]

            return nx.community.modularity(G, communities_sets)
        except Exception as e:
            logger.warning(f"Error computing standard modularity: {e}")
            return 0.0

    def _compute_signed_modularity(self, G: nx.Graph, partition: list[int]) -> float:
        """Compute modularity for signed networks."""
        try:
            if len(set(partition)) <= 1:
                return 0.0

            # Signed modularity computation
            # Q = (1/2m) * Σ[A_ij - (k_i * k_j)/(2m)] * δ(c_i, c_j)
            # where A_ij can be negative

            total_weight = 0.0
            positive_edges = 0
            negative_edges = 0

            # Calculate total weight and edge counts
            for i, j, data in G.edges(data=True):
                weight = data.get("weight", 1.0)
                total_weight += abs(weight)
                if weight > 0:
                    positive_edges += 1
                else:
                    negative_edges += 1

            if total_weight == 0:
                return 0.0

            modularity = 0.0
            nodes = list(G.nodes())

            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    node_i, node_j = nodes[i], nodes[j]

                    # Check if nodes are in same community
                    if partition[i] == partition[j]:
                        # Get edge weight (0 if no edge)
                        if G.has_edge(node_i, node_j):
                            A_ij = G[node_i][node_j].get("weight", 1.0)
                        else:
                            A_ij = 0.0

                        # Calculate expected weight
                        ki = sum(abs(G[node_i][neighbor].get("weight", 1.0)) for neighbor in G.neighbors(node_i))
                        kj = sum(abs(G[node_j][neighbor].get("weight", 1.0)) for neighbor in G.neighbors(node_j))

                        expected = (ki * kj) / (2 * total_weight) if total_weight > 0 else 0.0

                        modularity += A_ij - expected

            return modularity / (2 * total_weight) if total_weight > 0 else 0.0

        except Exception as e:
            logger.warning(f"Error computing signed modularity: {e}")
            return 0.0

    def _compute_positive_negative_modularity(self, G: nx.Graph, partition: list[int]) -> tuple[float, float]:
        """Compute modularity for positive and negative subgraphs separately."""
        try:
            G_pos = self._extract_positive_subgraph(G)
            G_neg = self._extract_negative_subgraph(G)

            pos_modularity = self._compute_standard_modularity(G_pos, partition)
            neg_modularity = self._compute_standard_modularity(G_neg, partition)

            return pos_modularity, neg_modularity

        except Exception as e:
            logger.warning(f"Error computing positive/negative modularity: {e}")
            return 0.0, 0.0

    def _compute_community_balance_scores(self, G: nx.Graph, partition: list[int]) -> list[float]:
        """Compute balance scores for each community."""
        try:
            n_nodes = len(G.nodes())
            balance_scores = [0.5] * n_nodes  # Default neutral balance

            # Group nodes by community
            communities = {}
            for i, comm in enumerate(partition):
                if comm not in communities:
                    communities[comm] = []
                communities[comm].append(i)

            # Calculate balance for each community
            for comm_id, nodes in communities.items():
                if len(nodes) < 2:
                    continue

                positive_weight = 0.0
                negative_weight = 0.0

                # Calculate intra-community positive and negative weights
                for i in nodes:
                    for j in nodes:
                        if i != j and G.has_edge(i, j):
                            weight = G[i][j].get("weight", 1.0)
                            if weight > 0:
                                positive_weight += weight
                            else:
                                negative_weight += abs(weight)

                # Calculate balance score
                total_weight = positive_weight + negative_weight
                if total_weight > 0:
                    balance = positive_weight / total_weight
                else:
                    balance = 0.5

                # Assign balance score to all nodes in community
                for node in nodes:
                    balance_scores[node] = balance

            return balance_scores

        except Exception as e:
            logger.warning(f"Error computing community balance scores: {e}")
            return [0.5] * len(G.nodes())

    def _spectral_clustering(self, edge_matrix: np.ndarray, n_clusters: int | None = None) -> list[int]:
        """Perform spectral clustering on the edge matrix with signed network support."""
        try:
            if n_clusters is None:
                # Enhanced eigengap heuristic considering signed networks
                if self.preserve_edge_signs:
                    # Use signed Laplacian for eigenvalue analysis
                    laplacian = self._compute_signed_laplacian(edge_matrix)
                    eigenvals = np.linalg.eigvals(laplacian)
                else:
                    eigenvals = np.linalg.eigvals(edge_matrix)

                eigenvals = np.sort(eigenvals)[::-1]
                if len(eigenvals) > 3:
                    gaps = np.diff(eigenvals[: min(10, len(eigenvals))])
                    n_clusters = np.argmax(gaps) + 2
                else:
                    n_clusters = 2

            n_clusters = min(n_clusters, edge_matrix.shape[0] - 1)
            n_clusters = max(n_clusters, 2)

            # Choose appropriate affinity matrix
            if self.preserve_edge_signs:
                affinity_matrix = self._compute_signed_affinity_matrix(edge_matrix)
            else:
                affinity_matrix = edge_matrix

            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity="precomputed",
                random_state=42,
                eigen_solver="arpack" if self.preserve_edge_signs else "auto",
            )
            labels = clustering.fit_predict(affinity_matrix)
            return labels.tolist()

        except Exception as e:
            logger.warning(f"Spectral clustering failed: {e}")
            return [0] * edge_matrix.shape[0]

    def _compute_network_topology_metrics(self, G: nx.Graph) -> dict[str, float]:
        """Compute network topology metrics with signed network enhancements."""
        metrics = {}

        try:
            # Basic metrics
            metrics["n_nodes"] = G.number_of_nodes()
            metrics["n_edges"] = G.number_of_edges()
            metrics["density"] = nx.density(G)

            # Signed network specific metrics
            if self.preserve_edge_signs and G.graph.get("signed", False):
                pos_edges, neg_edges, edge_balance = self._analyze_edge_signs(G)
                metrics["positive_edges"] = pos_edges
                metrics["negative_edges"] = neg_edges
                metrics["edge_balance_ratio"] = edge_balance
                metrics["edge_frustration"] = self._compute_edge_frustration(G)
            else:
                metrics["positive_edges"] = metrics["n_edges"]
                metrics["negative_edges"] = 0
                metrics["edge_balance_ratio"] = 1.0
                metrics["edge_frustration"] = 0.0

            # Clustering coefficient (use absolute weights for signed networks)
            if self.preserve_edge_signs and self._has_negative_weights(G):
                G_abs = self._create_absolute_weight_graph(G)
                metrics["avg_clustering"] = nx.average_clustering(G_abs)
            else:
                metrics["avg_clustering"] = nx.average_clustering(G)

            # Path-based metrics (use absolute weights for signed networks)
            working_graph = self._create_absolute_weight_graph(G) if self._has_negative_weights(G) else G

            if nx.is_connected(working_graph):
                metrics["avg_path_length"] = nx.average_shortest_path_length(working_graph)
                metrics["diameter"] = nx.diameter(working_graph)
                metrics["radius"] = nx.radius(working_graph)
            else:
                # For disconnected graphs, compute for largest component
                largest_cc = max(nx.connected_components(working_graph), key=len)
                subgraph = working_graph.subgraph(largest_cc)
                if len(subgraph) > 1:
                    metrics["avg_path_length"] = nx.average_shortest_path_length(subgraph)
                    metrics["diameter"] = nx.diameter(subgraph)
                    metrics["radius"] = nx.radius(subgraph)
                else:
                    metrics["avg_path_length"] = 0.0
                    metrics["diameter"] = 0.0
                    metrics["radius"] = 0.0

            # Small-world metrics
            metrics["small_world_sigma"] = self._compute_small_world_sigma(working_graph)

            # Efficiency measures
            metrics["global_efficiency"] = nx.global_efficiency(working_graph)
            metrics["local_efficiency"] = nx.local_efficiency(working_graph)

            # Signed network efficiency metrics
            if self.preserve_edge_signs and G.graph.get("signed", False):
                pos_eff, neg_eff = self._compute_signed_subgraph_efficiency(G)
                metrics["positive_subgraph_efficiency"] = pos_eff
                metrics["negative_subgraph_efficiency"] = neg_eff

            # Assortativity
            if working_graph.number_of_edges() > 0:
                metrics["degree_assortativity"] = nx.degree_assortativity_coefficient(working_graph)
            else:
                metrics["degree_assortativity"] = 0.0

            # Balance theory metrics for signed networks
            if self.preserve_edge_signs and G.graph.get("signed", False):
                metrics["structural_balance"] = self._compute_structural_balance(G)
                metrics["triangle_balance_ratio"] = self._compute_triangle_balance(G)

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
                "positive_edges": 0,
                "negative_edges": 0,
                "edge_balance_ratio": 1.0,
                "edge_frustration": 0.0,
            }

        return metrics

    def _analyze_edge_signs(self, G: nx.Graph) -> tuple[int, int, float]:
        """Analyze the distribution of positive and negative edges."""
        try:
            positive_edges = 0
            negative_edges = 0

            for _, _, data in G.edges(data=True):
                weight = data.get("weight", 1.0)
                if weight > 0:
                    positive_edges += 1
                elif weight < 0:
                    negative_edges += 1

            total_edges = positive_edges + negative_edges
            balance_ratio = positive_edges / total_edges if total_edges > 0 else 0.5

            return positive_edges, negative_edges, balance_ratio

        except Exception as e:
            logger.warning(f"Error analyzing edge signs: {e}")
            return 0, 0, 0.5

    def _compute_edge_frustration(self, G: nx.Graph) -> float:
        """Compute edge frustration metric for signed networks."""
        try:
            if G.number_of_edges() == 0:
                return 0.0

            frustrated_edges = 0
            total_triangles = 0

            # Check all triangles for balance
            for triangle in nx.enumerate_all_cliques(G):
                if len(triangle) == 3:
                    total_triangles += 1
                    i, j, k = triangle

                    # Get edge weights
                    w_ij = G[i][j].get("weight", 1.0) if G.has_edge(i, j) else 0
                    w_jk = G[j][k].get("weight", 1.0) if G.has_edge(j, k) else 0
                    w_ik = G[i][k].get("weight", 1.0) if G.has_edge(i, k) else 0

                    # Triangle is balanced if product of signs is positive
                    sign_product = np.sign(w_ij) * np.sign(w_jk) * np.sign(w_ik)
                    if sign_product < 0:
                        frustrated_edges += 1

            return frustrated_edges / total_triangles if total_triangles > 0 else 0.0

        except Exception as e:
            logger.warning(f"Error computing edge frustration: {e}")
            return 0.0

    def _compute_signed_subgraph_efficiency(self, G: nx.Graph) -> tuple[float, float]:
        """Compute efficiency for positive and negative subgraphs."""
        try:
            G_pos = self._extract_positive_subgraph(G)
            G_neg = self._extract_negative_subgraph(G)

            pos_efficiency = nx.global_efficiency(G_pos)
            neg_efficiency = nx.global_efficiency(G_neg)

            return pos_efficiency, neg_efficiency

        except Exception as e:
            logger.warning(f"Error computing signed subgraph efficiency: {e}")
            return 0.0, 0.0

    def _compute_structural_balance(self, G: nx.Graph) -> float:
        """Compute structural balance metric based on balance theory."""
        try:
            if G.number_of_nodes() < 3:
                return 1.0

            balanced_triangles = 0
            total_triangles = 0

            for triangle in nx.enumerate_all_cliques(G):
                if len(triangle) == 3:
                    total_triangles += 1
                    i, j, k = triangle

                    # Get edge signs
                    sign_ij = np.sign(G[i][j].get("weight", 1.0)) if G.has_edge(i, j) else 0
                    sign_jk = np.sign(G[j][k].get("weight", 1.0)) if G.has_edge(j, k) else 0
                    sign_ik = np.sign(G[i][k].get("weight", 1.0)) if G.has_edge(i, k) else 0

                    # Triangle is balanced if it has 0 or 2 negative edges
                    negative_edges = sum([1 for sign in [sign_ij, sign_jk, sign_ik] if sign < 0])
                    if negative_edges % 2 == 0:
                        balanced_triangles += 1

            return balanced_triangles / total_triangles if total_triangles > 0 else 1.0

        except Exception as e:
            logger.warning(f"Error computing structural balance: {e}")
            return 1.0

    def _compute_triangle_balance(self, G: nx.Graph) -> float:
        """Compute the ratio of balanced to total triangles."""
        try:
            return self._compute_structural_balance(G)
        except Exception as e:
            logger.warning(f"Error computing triangle balance: {e}")
            return 1.0

    def analyze_hierarchical_network_organization(self) -> dict[str, Any]:
        """Analyze H1: Hierarchical Network Organization with signed network enhancements."""
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

            # Enhanced hub analysis for signed networks
            if self.preserve_edge_signs and G.graph.get("signed", False):
                hub_analysis = self._analyze_signed_hubs(G, centralities)
            else:
                hub_analysis = self._analyze_unsigned_hubs(centralities)

            # Test hierarchy hypothesis with signed network considerations
            hierarchy_score = self._compute_hierarchy_score(G, centralities, hub_analysis)

            results[group_name] = {
                "centralities": centralities,
                "degree_analysis": degree_analysis,
                "hub_analysis": hub_analysis,
                "hierarchy_score": float(hierarchy_score),
                "supports_hierarchy_hypothesis": self._evaluate_hierarchy_hypothesis(
                    hierarchy_score, degree_analysis, hub_analysis
                ),
            }

        return results

    def _analyze_signed_hubs(self, G: nx.Graph, centralities: dict) -> dict[str, Any]:
        """Analyze hub neurons in signed networks."""
        try:
            # Identify excitatory and inhibitory hubs
            pos_degree_cents = centralities.get("positive_degree", [])
            neg_degree_cents = centralities.get("negative_degree", [])

            if not pos_degree_cents or not neg_degree_cents:
                return self._analyze_unsigned_hubs(centralities)

            # Excitatory hubs (high positive degree)
            pos_hub_threshold = np.percentile(pos_degree_cents, 90) if pos_degree_cents else 0
            excitatory_hubs = [i for i, dc in enumerate(pos_degree_cents) if dc >= pos_hub_threshold]

            # Inhibitory hubs (high negative degree)
            neg_hub_threshold = np.percentile(neg_degree_cents, 90) if neg_degree_cents else 0
            inhibitory_hubs = [i for i, dc in enumerate(neg_degree_cents) if dc >= neg_hub_threshold]

            # Balance analysis
            balance_scores = centralities.get("signed_centrality_balance", [])

            # Hub statistics
            n_excitatory_hubs = len(excitatory_hubs)
            n_inhibitory_hubs = len(inhibitory_hubs)

            excitatory_hub_betweenness = (
                np.mean([centralities["betweenness"][i] for i in excitatory_hubs]) if excitatory_hubs else 0.0
            )
            inhibitory_hub_betweenness = (
                np.mean([centralities["betweenness"][i] for i in inhibitory_hubs]) if inhibitory_hubs else 0.0
            )

            # Excitation-inhibition balance
            avg_balance = np.mean(balance_scores) if balance_scores else 0.5
            balance_variance = np.var(balance_scores) if balance_scores else 0.0

            return {
                "excitatory_hubs": excitatory_hubs,
                "inhibitory_hubs": inhibitory_hubs,
                "n_excitatory_hubs": n_excitatory_hubs,
                "n_inhibitory_hubs": n_inhibitory_hubs,
                "excitatory_hub_betweenness": float(excitatory_hub_betweenness),
                "inhibitory_hub_betweenness": float(inhibitory_hub_betweenness),
                "excitation_inhibition_balance": float(avg_balance),
                "balance_variance": float(balance_variance),
                "hub_balance_ratio": float(n_excitatory_hubs / (n_excitatory_hubs + n_inhibitory_hubs + 1e-10)),
            }

        except Exception as e:
            logger.warning(f"Error analyzing signed hubs: {e}")
            return self._analyze_unsigned_hubs(centralities)

    def _analyze_unsigned_hubs(self, centralities: dict) -> dict[str, Any]:
        """Analyze hub neurons in unsigned networks (original method)."""
        try:
            degree_cents = centralities.get("degree", [])

            if not degree_cents:
                return {
                    "hub_neurons": [],
                    "n_hubs": 0,
                    "hub_betweenness_centrality": 0.0,
                    "excitation_inhibition_balance": 0.5,
                }

            hub_threshold = np.percentile(degree_cents, 90)
            hub_neurons = [i for i, dc in enumerate(degree_cents) if dc >= hub_threshold]
            n_hubs = len(hub_neurons)

            hub_betweenness = np.mean([centralities["betweenness"][i] for i in hub_neurons]) if hub_neurons else 0.0

            return {
                "hub_neurons": hub_neurons,
                "n_hubs": n_hubs,
                "hub_betweenness_centrality": float(hub_betweenness),
                "excitation_inhibition_balance": 0.5,  # Neutral for unsigned networks
            }

        except Exception as e:
            logger.warning(f"Error analyzing unsigned hubs: {e}")
            return {
                "hub_neurons": [],
                "n_hubs": 0,
                "hub_betweenness_centrality": 0.0,
                "excitation_inhibition_balance": 0.5,
            }

    def _compute_hierarchy_score(self, G: nx.Graph, centralities: dict, hub_analysis: dict) -> float:
        """Compute hierarchy score with signed network considerations."""
        try:
            # Base hierarchy score from betweenness centrality
            base_score = np.mean(centralities.get("betweenness", [0])) if centralities.get("betweenness") else 0.0

            # Enhance with signed network metrics
            if self.preserve_edge_signs and G.graph.get("signed", False):
                # Consider excitation-inhibition balance
                ei_balance = hub_analysis.get("excitation_inhibition_balance", 0.5)
                balance_factor = min(ei_balance, 1 - ei_balance) * 2  # Favor balanced networks

                # Consider hub diversity (both excitatory and inhibitory hubs)
                n_exc_hubs = hub_analysis.get("n_excitatory_hubs", 0)
                n_inh_hubs = hub_analysis.get("n_inhibitory_hubs", 0)
                hub_diversity = min(n_exc_hubs, n_inh_hubs) / max(n_exc_hubs + n_inh_hubs, 1)

                # Combined hierarchy score
                hierarchy_score = base_score * (1 + balance_factor + hub_diversity)
            else:
                hierarchy_score = base_score

            return hierarchy_score

        except Exception as e:
            logger.warning(f"Error computing hierarchy score: {e}")
            return 0.0

    def _evaluate_hierarchy_hypothesis(self, hierarchy_score: float, degree_analysis: dict, hub_analysis: dict) -> bool:
        """Evaluate hierarchy hypothesis with enhanced criteria."""
        try:
            # Base criteria
            has_scale_free = degree_analysis.get("is_scale_free", False)
            sufficient_hierarchy = hierarchy_score > 0.1

            # Enhanced criteria for signed networks
            if self.preserve_edge_signs:
                # Check for balanced hub structure
                ei_balance = hub_analysis.get("excitation_inhibition_balance", 0.5)
                balanced_hubs = 0.3 <= ei_balance <= 0.7  # Not too extreme

                # Check for hub diversity
                n_exc = hub_analysis.get("n_excitatory_hubs", 0)
                n_inh = hub_analysis.get("n_inhibitory_hubs", 0)
                has_diverse_hubs = n_exc > 0 and n_inh > 0

                return sufficient_hierarchy and (has_scale_free or balanced_hubs or has_diverse_hubs)
            return sufficient_hierarchy and has_scale_free

        except Exception as e:
            logger.warning(f"Error evaluating hierarchy hypothesis: {e}")
            return False

    def analyze_modular_coordination_architecture(self) -> dict[str, Any]:
        """Analyze H2: Modular Coordination Architecture with signed network enhancements."""
        results = {}

        for group_name, tensor in self.activation_tensors.items():
            data = tensor.detach().cpu().numpy()

            # Construct graph
            edge_matrix = self._construct_graph_edges(data)
            G = self._construct_networkx_graph(edge_matrix)

            # Detect communities
            communities = self._detect_communities(G, edge_matrix)

            # Enhanced modularity analysis for signed networks
            modularity_analysis = self._analyze_modularity_metrics(G, communities)

            # Community connectivity analysis with sign awareness
            connectivity_analysis = self._analyze_community_connectivity_signed(G, communities)

            results[group_name] = {
                "communities": communities,
                "modularity_analysis": modularity_analysis,
                "connectivity_analysis": connectivity_analysis,
                "supports_modularity_hypothesis": self._evaluate_modularity_hypothesis(
                    modularity_analysis, connectivity_analysis
                ),
            }

        return results

    def _analyze_modularity_metrics(self, G: nx.Graph, communities: dict) -> dict[str, Any]:
        """Analyze modularity metrics with signed network support."""
        try:
            analysis = {}

            # Basic modularity metrics
            analysis["standard_modularity"] = communities.get("modularity", 0.0)
            analysis["n_communities"] = communities.get("n_communities", 1)

            # Signed network modularity metrics
            if self.preserve_edge_signs and G.graph.get("signed", False):
                analysis["signed_modularity"] = communities.get("signed_modularity", 0.0)
                analysis["positive_modularity"] = communities.get("positive_modularity", 0.0)
                analysis["negative_modularity"] = communities.get("negative_modularity", 0.0)
                analysis["community_balance_scores"] = communities.get("community_balance_scores", [])

                # Modularity balance ratio
                pos_mod = analysis["positive_modularity"]
                neg_mod = analysis["negative_modularity"]
                total_mod = abs(pos_mod) + abs(neg_mod)
                analysis["modularity_balance_ratio"] = pos_mod / total_mod if total_mod > 0 else 0.5
            else:
                analysis["signed_modularity"] = analysis["standard_modularity"]
                analysis["positive_modularity"] = analysis["standard_modularity"]
                analysis["negative_modularity"] = 0.0
                analysis["modularity_balance_ratio"] = 1.0

            # Community size statistics
            if "louvain" in communities:
                community_labels = communities["louvain"]
                community_sizes = list(Counter(community_labels).values())
                analysis["community_sizes"] = community_sizes
                analysis["avg_community_size"] = float(np.mean(community_sizes))
                analysis["std_community_size"] = float(np.std(community_sizes))
                analysis["community_size_entropy"] = self._compute_community_size_entropy(community_sizes)
            else:
                analysis["community_sizes"] = [len(G.nodes())]
                analysis["avg_community_size"] = float(len(G.nodes()))
                analysis["std_community_size"] = 0.0
                analysis["community_size_entropy"] = 0.0

            return analysis

        except Exception as e:
            logger.warning(f"Error analyzing modularity metrics: {e}")
            return {"standard_modularity": 0.0, "n_communities": 1}

    def _compute_community_size_entropy(self, community_sizes: list[int]) -> float:
        """Compute entropy of community size distribution."""
        try:
            if not community_sizes:
                return 0.0

            total = sum(community_sizes)
            probabilities = [size / total for size in community_sizes]
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            return float(entropy)

        except Exception as e:
            logger.warning(f"Error computing community size entropy: {e}")
            return 0.0

    def _analyze_community_connectivity_signed(self, G: nx.Graph, communities: dict) -> dict[str, Any]:
        """Analyze community connectivity with signed network awareness."""
        try:
            if "louvain" not in communities:
                return {"inter_community_density": 0.0, "intra_community_density": 0.0}

            community_labels = communities["louvain"]

            # Basic connectivity
            inter_density, intra_density = self._compute_community_connectivity(G, community_labels)

            analysis = {
                "inter_community_density": float(inter_density),
                "intra_community_density": float(intra_density),
                "density_ratio": float(intra_density / inter_density if inter_density > 0 else np.inf),
            }

            # Signed network specific connectivity analysis
            if self.preserve_edge_signs and G.graph.get("signed", False):
                signed_connectivity = self._compute_signed_community_connectivity(G, community_labels)
                analysis.update(signed_connectivity)

            return analysis

        except Exception as e:
            logger.warning(f"Error analyzing community connectivity: {e}")
            return {"inter_community_density": 0.0, "intra_community_density": 0.0}

    def _compute_signed_community_connectivity(self, G: nx.Graph, community_labels: list[int]) -> dict[str, float]:
        """Compute signed community connectivity metrics."""
        try:
            # Initialize counters
            intra_positive = 0
            intra_negative = 0
            inter_positive = 0
            inter_negative = 0

            intra_possible = 0
            inter_possible = 0

            nodes = list(G.nodes())

            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    node_i, node_j = nodes[i], nodes[j]
                    same_community = community_labels[i] == community_labels[j]

                    if same_community:
                        intra_possible += 1
                        if G.has_edge(node_i, node_j):
                            weight = G[node_i][node_j].get("weight", 1.0)
                            if weight > 0:
                                intra_positive += 1
                            else:
                                intra_negative += 1
                    else:
                        inter_possible += 1
                        if G.has_edge(node_i, node_j):
                            weight = G[node_i][node_j].get("weight", 1.0)
                            if weight > 0:
                                inter_positive += 1
                            else:
                                inter_negative += 1

            # Compute densities
            intra_pos_density = intra_positive / intra_possible if intra_possible > 0 else 0.0
            intra_neg_density = intra_negative / intra_possible if intra_possible > 0 else 0.0
            inter_pos_density = inter_positive / inter_possible if inter_possible > 0 else 0.0
            inter_neg_density = inter_negative / inter_possible if inter_possible > 0 else 0.0

            # Balance metrics
            intra_balance = (
                intra_positive / (intra_positive + intra_negative) if (intra_positive + intra_negative) > 0 else 0.5
            )
            inter_balance = (
                inter_positive / (inter_positive + inter_negative) if (inter_positive + inter_negative) > 0 else 0.5
            )

            return {
                "intra_positive_density": float(intra_pos_density),
                "intra_negative_density": float(intra_neg_density),
                "inter_positive_density": float(inter_pos_density),
                "inter_negative_density": float(inter_neg_density),
                "intra_community_balance": float(intra_balance),
                "inter_community_balance": float(inter_balance),
                "community_sign_segregation": float(abs(intra_balance - inter_balance)),
            }

        except Exception as e:
            logger.warning(f"Error computing signed community connectivity: {e}")
            return {
                "intra_positive_density": 0.0,
                "intra_negative_density": 0.0,
                "inter_positive_density": 0.0,
                "inter_negative_density": 0.0,
                "intra_community_balance": 0.5,
                "inter_community_balance": 0.5,
                "community_sign_segregation": 0.0,
            }

    def _evaluate_modularity_hypothesis(self, modularity_analysis: dict, connectivity_analysis: dict) -> bool:
        """Evaluate modularity hypothesis with enhanced criteria."""
        try:
            # Base criteria
            standard_mod = modularity_analysis.get("standard_modularity", 0.0)
            n_communities = modularity_analysis.get("n_communities", 1)
            density_ratio = connectivity_analysis.get("density_ratio", 1.0)

            base_support = standard_mod > 0.3 and n_communities > 1 and density_ratio > 2.0

            # Enhanced criteria for signed networks
            if self.preserve_edge_signs:
                # Check signed modularity
                signed_mod = modularity_analysis.get("signed_modularity", 0.0)

                # Check community sign segregation (positive within, negative between)
                sign_segregation = connectivity_analysis.get("community_sign_segregation", 0.0)
                intra_balance = connectivity_analysis.get("intra_community_balance", 0.5)
                inter_balance = connectivity_analysis.get("inter_community_balance", 0.5)

                # Good modularity in signed networks: positive within communities, negative between
                good_sign_pattern = intra_balance > 0.6 and inter_balance < 0.4
                sufficient_segregation = sign_segregation > 0.2

                return base_support or (signed_mod > 0.2 and (good_sign_pattern or sufficient_segregation))
            return base_support

        except Exception as e:
            logger.warning(f"Error evaluating modularity hypothesis: {e}")
            return False

    def analyze_context_dependent_network_topology(self) -> dict[str, Any]:
        """Analyze H3: Context-Dependent Network Topology with signed network enhancements."""
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
                    "sign_change_analysis": {},
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
            context_differences = self._compute_context_differences(rare_metrics, common_metrics)

            # Enhanced sign change analysis for signed networks
            if self.preserve_edge_signs:
                sign_change_analysis = self._analyze_context_sign_changes(rare_edge_matrix, common_edge_matrix)
            else:
                sign_change_analysis = {}

            # Test for significant context-dependent changes
            supports_hypothesis = self._evaluate_adaptive_topology_hypothesis(context_differences, sign_change_analysis)

            results[group_name] = {
                "rare_context_metrics": rare_metrics,
                "common_context_metrics": common_metrics,
                "context_differences": context_differences,
                "sign_change_analysis": sign_change_analysis,
                "supports_adaptive_topology_hypothesis": bool(supports_hypothesis),
            }

        return results

    def _compute_context_differences(self, rare_metrics: dict, common_metrics: dict) -> dict[str, float]:
        """Compute differences between rare and common context metrics."""
        differences = {}

        try:
            for metric in rare_metrics:
                if isinstance(rare_metrics[metric], (int, float)) and isinstance(common_metrics[metric], (int, float)):
                    diff = rare_metrics[metric] - common_metrics[metric]
                    differences[f"{metric}_difference"] = float(diff)

                    # Compute relative change
                    if common_metrics[metric] != 0:
                        rel_change = diff / abs(common_metrics[metric])
                        differences[f"{metric}_relative_change"] = float(rel_change)
                    else:
                        differences[f"{metric}_relative_change"] = float("inf") if diff != 0 else 0.0

            return differences

        except Exception as e:
            logger.warning(f"Error computing context differences: {e}")
            return {}

    def _analyze_context_sign_changes(self, rare_matrix: np.ndarray, common_matrix: np.ndarray) -> dict[str, Any]:
        """Analyze sign changes between contexts in signed networks."""
        try:
            if rare_matrix.shape != common_matrix.shape:
                logger.warning("Matrix shape mismatch in sign change analysis")
                return {}

            # Edge turnover analysis
            rare_edges = rare_matrix != 0
            common_edges = common_matrix != 0

            # Edges that appear/disappear
            appearing_edges = common_edges & ~rare_edges
            disappearing_edges = rare_edges & ~common_edges
            stable_edges = rare_edges & common_edges

            edge_turnover = {
                "appearing_edges": int(np.sum(appearing_edges)),
                "disappearing_edges": int(np.sum(disappearing_edges)),
                "stable_edges": int(np.sum(stable_edges)),
                "total_possible_edges": int(rare_matrix.size - rare_matrix.shape[0]),  # Exclude diagonal
            }

            edge_turnover["turnover_rate"] = (
                edge_turnover["appearing_edges"] + edge_turnover["disappearing_edges"]
            ) / edge_turnover["total_possible_edges"]

            # Sign change analysis for stable edges
            sign_changes = {}
            if np.sum(stable_edges) > 0:
                rare_signs = np.sign(rare_matrix[stable_edges])
                common_signs = np.sign(common_matrix[stable_edges])

                sign_flips = rare_signs != common_signs
                sign_changes["sign_flip_count"] = int(np.sum(sign_flips))
                sign_changes["sign_flip_rate"] = float(np.mean(sign_flips))

                # Magnitude changes for stable edges
                magnitude_changes = np.abs(rare_matrix[stable_edges] - common_matrix[stable_edges])
                sign_changes["avg_magnitude_change"] = float(np.mean(magnitude_changes))
                sign_changes["max_magnitude_change"] = float(np.max(magnitude_changes))
            else:
                sign_changes = {
                    "sign_flip_count": 0,
                    "sign_flip_rate": 0.0,
                    "avg_magnitude_change": 0.0,
                    "max_magnitude_change": 0.0,
                }

            # Balance changes
            rare_balance = self._compute_matrix_balance(rare_matrix)
            common_balance = self._compute_matrix_balance(common_matrix)

            balance_analysis = {
                "rare_context_balance": float(rare_balance),
                "common_context_balance": float(common_balance),
                "balance_change": float(common_balance - rare_balance),
            }

            return {
                "edge_turnover": edge_turnover,
                "sign_changes": sign_changes,
                "balance_analysis": balance_analysis,
            }

        except Exception as e:
            logger.warning(f"Error analyzing context sign changes: {e}")
            return {}

    def _compute_matrix_balance(self, matrix: np.ndarray) -> float:
        """Compute balance ratio of a matrix (positive edges / total edges)."""
        try:
            non_zero = matrix != 0
            if np.sum(non_zero) == 0:
                return 0.5

            positive_edges = np.sum(matrix[non_zero] > 0)
            total_edges = np.sum(non_zero)

            return positive_edges / total_edges

        except Exception as e:
            logger.warning(f"Error computing matrix balance: {e}")
            return 0.5

    def _evaluate_adaptive_topology_hypothesis(self, context_differences: dict, sign_change_analysis: dict) -> bool:
        """Evaluate adaptive topology hypothesis with enhanced criteria."""
        try:
            # Base criteria from topology differences
            clustering_diff = abs(context_differences.get("avg_clustering_difference", 0))
            path_length_diff = abs(context_differences.get("avg_path_length_difference", 0))
            efficiency_diff = abs(context_differences.get("global_efficiency_difference", 0))

            base_support = (clustering_diff > 0.1) or (path_length_diff > 0.5) or (efficiency_diff > 0.1)

            # Enhanced criteria for signed networks
            if self.preserve_edge_signs and sign_change_analysis:
                # Edge turnover criteria
                turnover_rate = sign_change_analysis.get("edge_turnover", {}).get("turnover_rate", 0.0)

                # Sign change criteria
                sign_flip_rate = sign_change_analysis.get("sign_changes", {}).get("sign_flip_rate", 0.0)

                # Balance change criteria
                balance_change = abs(sign_change_analysis.get("balance_analysis", {}).get("balance_change", 0.0))

                # Support if significant changes in any dimension
                signed_support = (turnover_rate > 0.1) or (sign_flip_rate > 0.1) or (balance_change > 0.1)

                return base_support or signed_support
            return base_support

        except Exception as e:
            logger.warning(f"Error evaluating adaptive topology hypothesis: {e}")
            return False

    def analyze_optimized_information_flow(self) -> dict[str, Any]:
        """Analyze H4: Optimized Information Flow with signed network enhancements."""
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

            # Enhanced optimization analysis for signed networks
            if self.preserve_edge_signs and G.graph.get("signed", False):
                optimization_analysis = self._analyze_signed_optimization(G, topology_metrics, random_metrics)
            else:
                optimization_analysis = self._analyze_unsigned_optimization(topology_metrics, random_metrics)

            # Test optimization hypothesis
            supports_hypothesis = self._evaluate_optimization_hypothesis(optimization_analysis)

            results[group_name] = {
                "topology_metrics": topology_metrics,
                "random_baseline_metrics": random_metrics,
                "optimization_analysis": optimization_analysis,
                "supports_optimization_hypothesis": bool(supports_hypothesis),
            }

        return results

    def _analyze_signed_optimization(self, G: nx.Graph, topology_metrics: dict, random_metrics: dict) -> dict[str, Any]:
        """Analyze optimization for signed networks."""
        try:
            # Base optimization metrics
            base_analysis = self._analyze_unsigned_optimization(topology_metrics, random_metrics)

            # Excitation-inhibition balance optimization
            edge_balance = topology_metrics.get("edge_balance_ratio", 0.5)
            optimal_balance_score = 1.0 - abs(edge_balance - 0.5) * 2  # Closer to 0.5 is better

            # Separate positive/negative subgraph efficiency
            pos_efficiency = topology_metrics.get("positive_subgraph_efficiency", 0.0)
            neg_efficiency = topology_metrics.get("negative_subgraph_efficiency", 0.0)

            # Balance between positive and negative efficiency
            if pos_efficiency + neg_efficiency > 0:
                efficiency_balance = min(pos_efficiency, neg_efficiency) / max(pos_efficiency, neg_efficiency)
            else:
                efficiency_balance = 0.0

            # Structural balance as optimization criterion
            structural_balance = topology_metrics.get("structural_balance", 1.0)

            # Frustration as anti-optimization criterion
            frustration = topology_metrics.get("edge_frustration", 0.0)
            frustration_penalty = 1.0 - frustration

            # Combined balance optimization score
            balance_optimization_score = (
                0.3 * optimal_balance_score
                + 0.3 * efficiency_balance
                + 0.2 * structural_balance
                + 0.2 * frustration_penalty
            )

            # Enhanced cost-efficiency with sign awareness
            density = topology_metrics.get("density", 0.0)
            global_efficiency = topology_metrics.get("global_efficiency", 0.0)

            # Cost includes both connection cost and frustration cost
            total_cost = density + 0.5 * frustration
            cost_efficiency_ratio = global_efficiency / total_cost if total_cost > 0 else 0.0

            signed_analysis = base_analysis.copy()
            signed_analysis.update(
                {
                    "excitation_inhibition_balance": float(edge_balance),
                    "optimal_balance_score": float(optimal_balance_score),
                    "positive_subgraph_efficiency": float(pos_efficiency),
                    "negative_subgraph_efficiency": float(neg_efficiency),
                    "efficiency_balance": float(efficiency_balance),
                    "structural_balance": float(structural_balance),
                    "frustration_penalty": float(frustration_penalty),
                    "balance_optimization_score": float(balance_optimization_score),
                    "enhanced_cost_efficiency_ratio": float(cost_efficiency_ratio),
                }
            )

            return signed_analysis

        except Exception as e:
            logger.warning(f"Error analyzing signed optimization: {e}")
            return self._analyze_unsigned_optimization(topology_metrics, random_metrics)

    def _analyze_unsigned_optimization(self, topology_metrics: dict, random_metrics: dict) -> dict[str, Any]:
        """Analyze optimization for unsigned networks (original method)."""
        try:
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

            return {
                "efficiency_ratio": float(efficiency_ratio),
                "clustering_ratio": float(clustering_ratio),
                "path_length_ratio": float(path_length_ratio),
                "cost_efficiency_ratio": float(cost_efficiency_ratio),
            }

        except Exception as e:
            logger.warning(f"Error analyzing unsigned optimization: {e}")
            return {
                "efficiency_ratio": 1.0,
                "clustering_ratio": 1.0,
                "path_length_ratio": 1.0,
                "cost_efficiency_ratio": 0.0,
            }

    def _evaluate_optimization_hypothesis(self, optimization_analysis: dict) -> bool:
        """Evaluate optimization hypothesis with enhanced criteria."""
        try:
            # Base criteria
            efficiency_ratio = optimization_analysis.get("efficiency_ratio", 1.0)
            clustering_ratio = optimization_analysis.get("clustering_ratio", 1.0)
            cost_efficiency_ratio = optimization_analysis.get("cost_efficiency_ratio", 0.0)

            base_optimized = efficiency_ratio > 1.1 and clustering_ratio > 1.1 and cost_efficiency_ratio > 0.5

            # Enhanced criteria for signed networks
            if "balance_optimization_score" in optimization_analysis:
                balance_score = optimization_analysis.get("balance_optimization_score", 0.0)
                enhanced_cost_eff = optimization_analysis.get("enhanced_cost_efficiency_ratio", 0.0)
                structural_balance = optimization_analysis.get("structural_balance", 1.0)

                # Good optimization in signed networks includes balance and low frustration
                signed_optimized = balance_score > 0.6 and enhanced_cost_eff > 0.3 and structural_balance > 0.7

                return base_optimized or signed_optimized
            return base_optimized

        except Exception as e:
            logger.warning(f"Error evaluating optimization hypothesis: {e}")
            return False

    def analyze_advanced_signed_network_properties(self) -> dict[str, Any]:
        """Analyze advanced signed network properties including balance theory and frustration."""
        if not self.preserve_edge_signs:
            logger.info("Skipping advanced signed network analysis (preserve_edge_signs=False)")
            return {}

        results = {}

        for group_name, tensor in self.activation_tensors.items():
            data = tensor.detach().cpu().numpy()

            # Construct signed graph
            edge_matrix = self._construct_graph_edges(data)
            G = self._construct_networkx_graph(edge_matrix)

            if not G.graph.get("signed", False):
                logger.warning(f"Graph for {group_name} is not signed, skipping advanced analysis")
                continue

            # Balance theory analysis
            balance_analysis = self._comprehensive_balance_analysis(G)

            # Frustration metrics
            frustration_analysis = self._comprehensive_frustration_analysis(G, edge_matrix)

            # Stability analysis
            stability_analysis = self._analyze_network_stability(G)

            # Dynamic sign analysis (if temporal data available)
            dynamic_analysis = self._analyze_dynamic_signs(data)

            results[group_name] = {
                "balance_theory": balance_analysis,
                "frustration_metrics": frustration_analysis,
                "stability_analysis": stability_analysis,
                "dynamic_sign_analysis": dynamic_analysis,
            }

        return results

    def _comprehensive_balance_analysis(self, G: nx.Graph) -> dict[str, Any]:
        """Comprehensive balance theory analysis."""
        try:
            analysis = {}

            # Basic balance metrics
            analysis["structural_balance"] = self._compute_structural_balance(G)
            analysis["triangle_balance_ratio"] = self._compute_triangle_balance(G)

            # Weak balance analysis
            analysis["weak_balance"] = self._compute_weak_balance(G)

            # Local balance for each node
            node_balance_scores = self._compute_node_balance_scores(G)
            analysis["node_balance_scores"] = node_balance_scores
            analysis["avg_node_balance"] = float(np.mean(node_balance_scores)) if node_balance_scores else 0.0
            analysis["balance_variance"] = float(np.var(node_balance_scores)) if node_balance_scores else 0.0

            # Balance satisfaction ratio
            analysis["balance_satisfaction"] = self._compute_balance_satisfaction(G)

            # Polarization analysis
            analysis["network_polarization"] = self._compute_network_polarization(G)

            return analysis

        except Exception as e:
            logger.warning(f"Error in comprehensive balance analysis: {e}")
            return {"structural_balance": 1.0, "triangle_balance_ratio": 1.0}

    def _compute_weak_balance(self, G: nx.Graph) -> float:
        """Compute weak balance (partition into two hostile groups)."""
        try:
            # Weak balance: network can be partitioned into two groups with positive within, negative between
            nodes = list(G.nodes())
            n_nodes = len(nodes)

            if n_nodes < 2:
                return 1.0

            best_balance = 0.0

            # Try different partitions (simplified heuristic)
            for _ in range(min(100, 2 ** (min(n_nodes, 10)))):  # Limit search for large graphs
                # Random partition
                partition = np.random.choice([0, 1], size=n_nodes)
                balance = self._evaluate_partition_balance(G, partition)
                best_balance = max(best_balance, balance)

            return best_balance

        except Exception as e:
            logger.warning(f"Error computing weak balance: {e}")
            return 1.0

    def _evaluate_partition_balance(self, G: nx.Graph, partition: np.ndarray) -> float:
        """Evaluate how well a partition satisfies weak balance."""
        try:
            correct_edges = 0
            total_edges = 0

            nodes = list(G.nodes())

            for i, j, data in G.edges(data=True):
                total_edges += 1
                weight = data.get("weight", 1.0)

                # Find node indices
                i_idx = nodes.index(i)
                j_idx = nodes.index(j)

                same_group = partition[i_idx] == partition[j_idx]
                positive_edge = weight > 0

                # Correct if: positive within group or negative between groups
                if (same_group and positive_edge) or (not same_group and not positive_edge):
                    correct_edges += 1

            return correct_edges / total_edges if total_edges > 0 else 1.0

        except Exception as e:
            logger.warning(f"Error evaluating partition balance: {e}")
            return 0.0

    def _compute_node_balance_scores(self, G: nx.Graph) -> list[float]:
        """Compute balance score for each node."""
        try:
            node_scores = []

            for node in G.nodes():
                balanced_triangles = 0
                total_triangles = 0

                neighbors = list(G.neighbors(node))

                # Check all triangles involving this node
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        neighbor1, neighbor2 = neighbors[i], neighbors[j]

                        if G.has_edge(neighbor1, neighbor2):
                            total_triangles += 1

                            # Get edge weights
                            w1 = G[node][neighbor1].get("weight", 1.0)
                            w2 = G[node][neighbor2].get("weight", 1.0)
                            w3 = G[neighbor1][neighbor2].get("weight", 1.0)

                            # Check if triangle is balanced
                            signs = [np.sign(w1), np.sign(w2), np.sign(w3)]
                            negative_count = sum(1 for s in signs if s < 0)

                            if negative_count % 2 == 0:
                                balanced_triangles += 1

                node_score = balanced_triangles / total_triangles if total_triangles > 0 else 1.0
                node_scores.append(node_score)

            return node_scores

        except Exception as e:
            logger.warning(f"Error computing node balance scores: {e}")
            return []

    def _compute_balance_satisfaction(self, G: nx.Graph) -> float:
        """Compute overall balance satisfaction of the network."""
        try:
            # Measure how well the network satisfies balance theory
            total_triads = 0
            satisfied_triads = 0

            nodes = list(G.nodes())

            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    for k in range(j + 1, len(nodes)):
                        total_triads += 1

                        # Get edge weights (0 if no edge)
                        w_ij = G[nodes[i]][nodes[j]].get("weight", 0.0) if G.has_edge(nodes[i], nodes[j]) else 0.0
                        w_jk = G[nodes[j]][nodes[k]].get("weight", 0.0) if G.has_edge(nodes[j], nodes[k]) else 0.0
                        w_ik = G[nodes[i]][nodes[k]].get("weight", 0.0) if G.has_edge(nodes[i], nodes[k]) else 0.0

                        # Count negative edges
                        negative_edges = sum(1 for w in [w_ij, w_jk, w_ik] if w < 0)

                        # Triad is satisfied if it has even number of negative edges
                        if negative_edges % 2 == 0:
                            satisfied_triads += 1

            return satisfied_triads / total_triads if total_triads > 0 else 1.0

        except Exception as e:
            logger.warning(f"Error computing balance satisfaction: {e}")
            return 1.0

    def _compute_network_polarization(self, G: nx.Graph) -> float:
        """Compute network polarization measure."""
        try:
            # Measure how polarized the network is (tendency to form opposing groups)
            positive_edges = 0
            negative_edges = 0

            for _, _, data in G.edges(data=True):
                weight = data.get("weight", 1.0)
                if weight > 0:
                    positive_edges += 1
                else:
                    negative_edges += 1

            total_edges = positive_edges + negative_edges
            if total_edges == 0:
                return 0.0

            # Polarization is high when there are many negative edges
            polarization = negative_edges / total_edges

            return float(polarization)

        except Exception as e:
            logger.warning(f"Error computing network polarization: {e}")
            return 0.0

    def _comprehensive_frustration_analysis(self, G: nx.Graph, edge_matrix: np.ndarray) -> dict[str, Any]:
        """Comprehensive frustration analysis."""
        try:
            analysis = {}

            # Basic frustration metrics
            analysis["edge_frustration"] = self._compute_edge_frustration(G)
            analysis["global_frustration"] = self._compute_global_frustration(edge_matrix)

            # Local frustration for each node
            node_frustrations = self._compute_node_frustrations(G)
            analysis["node_frustrations"] = node_frustrations
            analysis["avg_node_frustration"] = float(np.mean(node_frustrations)) if node_frustrations else 0.0
            analysis["frustration_variance"] = float(np.var(node_frustrations)) if node_frustrations else 0.0

            # Frustration distribution
            analysis["frustration_distribution"] = self._analyze_frustration_distribution(G)

            # Conflict intensity
            analysis["conflict_intensity"] = self._compute_conflict_intensity(G)

            return analysis

        except Exception as e:
            logger.warning(f"Error in comprehensive frustration analysis: {e}")
            return {"edge_frustration": 0.0, "global_frustration": 0.0}

    def _compute_global_frustration(self, edge_matrix: np.ndarray) -> float:
        """Compute global frustration index."""
        try:
            n_nodes = edge_matrix.shape[0]
            total_frustration = 0.0
            total_triangles = 0

            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    for k in range(j + 1, n_nodes):
                        if edge_matrix[i, j] != 0 and edge_matrix[j, k] != 0 and edge_matrix[i, k] != 0:
                            total_triangles += 1

                            # Product of signs
                            sign_product = (
                                np.sign(edge_matrix[i, j]) * np.sign(edge_matrix[j, k]) * np.sign(edge_matrix[i, k])
                            )

                            if sign_product < 0:
                                total_frustration += 1

            return total_frustration / total_triangles if total_triangles > 0 else 0.0

        except Exception as e:
            logger.warning(f"Error computing global frustration: {e}")
            return 0.0

    def _compute_node_frustrations(self, G: nx.Graph) -> list[float]:
        """Compute frustration for each node."""
        try:
            node_frustrations = []

            for node in G.nodes():
                frustrated_triangles = 0
                total_triangles = 0

                neighbors = list(G.neighbors(node))

                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        neighbor1, neighbor2 = neighbors[i], neighbors[j]

                        if G.has_edge(neighbor1, neighbor2):
                            total_triangles += 1

                            # Get edge signs
                            sign1 = np.sign(G[node][neighbor1].get("weight", 1.0))
                            sign2 = np.sign(G[node][neighbor2].get("weight", 1.0))
                            sign3 = np.sign(G[neighbor1][neighbor2].get("weight", 1.0))

                            # Triangle is frustrated if product is negative
                            if sign1 * sign2 * sign3 < 0:
                                frustrated_triangles += 1

                frustration = frustrated_triangles / total_triangles if total_triangles > 0 else 0.0
                node_frustrations.append(frustration)

            return node_frustrations

        except Exception as e:
            logger.warning(f"Error computing node frustrations: {e}")
            return []

    def _analyze_frustration_distribution(self, G: nx.Graph) -> dict[str, Any]:
        """Analyze the distribution of frustration across the network."""
        try:
            node_frustrations = self._compute_node_frustrations(G)

            if not node_frustrations:
                return {"distribution": [], "entropy": 0.0}

            # Create histogram of frustration levels
            bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
            hist, _ = np.histogram(node_frustrations, bins=bins)

            # Compute entropy of distribution
            total = sum(hist)
            if total > 0:
                probs = [h / total for h in hist if h > 0]
                entropy = -sum(p * np.log2(p) for p in probs)
            else:
                entropy = 0.0

            return {
                "distribution": hist.tolist(),
                "entropy": float(entropy),
                "max_frustration": float(max(node_frustrations)),
                "min_frustration": float(min(node_frustrations)),
            }

        except Exception as e:
            logger.warning(f"Error analyzing frustration distribution: {e}")
            return {"distribution": [], "entropy": 0.0}

    def _compute_conflict_intensity(self, G: nx.Graph) -> float:
        """Compute overall conflict intensity in the network."""
        try:
            total_conflict = 0.0
            total_weight = 0.0

            for _, _, data in G.edges(data=True):
                weight = data.get("weight", 1.0)
                if weight < 0:
                    total_conflict += abs(weight)
                total_weight += abs(weight)

            return total_conflict / total_weight if total_weight > 0 else 0.0

        except Exception as e:
            logger.warning(f"Error computing conflict intensity: {e}")
            return 0.0

    def _analyze_network_stability(self, G: nx.Graph) -> dict[str, Any]:
        """Analyze network stability and feedback loops."""
        try:
            analysis = {}

            # Positive and negative feedback loops
            pos_loops, neg_loops = self._detect_feedback_loops(G)
            analysis["positive_feedback_loops"] = pos_loops
            analysis["negative_feedback_loops"] = neg_loops
            analysis["feedback_loop_ratio"] = (
                pos_loops / (pos_loops + neg_loops) if (pos_loops + neg_loops) > 0 else 0.5
            )

            # Stability indicators
            analysis["stability_index"] = self._compute_stability_index(G)
            analysis["convergence_likelihood"] = self._estimate_convergence_likelihood(G)

            # Dominant eigenvalue analysis for stability
            try:
                adjacency_matrix = nx.adjacency_matrix(G, weight="weight").todense()
                eigenvals = np.linalg.eigvals(adjacency_matrix)
                max_eigenval = max(eigenvals, key=abs)
                analysis["dominant_eigenvalue"] = float(abs(max_eigenval))
                analysis["spectral_radius"] = float(abs(max_eigenval))
                analysis["stable_by_eigenvalue"] = abs(max_eigenval) < 1.0
            except:
                analysis["dominant_eigenvalue"] = 0.0
                analysis["spectral_radius"] = 0.0
                analysis["stable_by_eigenvalue"] = True

            return analysis

        except Exception as e:
            logger.warning(f"Error analyzing network stability: {e}")
            return {"positive_feedback_loops": 0, "negative_feedback_loops": 0}

    def _detect_feedback_loops(self, G: nx.Graph) -> tuple[int, int]:
        """Detect positive and negative feedback loops."""
        try:
            positive_loops = 0
            negative_loops = 0

            # Look for simple cycles (length 2-4)
            for cycle_length in range(2, min(5, len(G.nodes()) + 1)):
                try:
                    cycles = list(nx.simple_cycles(G.to_directed(), length_limit=cycle_length))

                    for cycle in cycles:
                        if len(cycle) == cycle_length:
                            # Compute cycle sign product
                            sign_product = 1.0
                            for i in range(len(cycle)):
                                node1 = cycle[i]
                                node2 = cycle[(i + 1) % len(cycle)]
                                if G.has_edge(node1, node2):
                                    weight = G[node1][node2].get("weight", 1.0)
                                    sign_product *= np.sign(weight)

                            if sign_product > 0:
                                positive_loops += 1
                            elif sign_product < 0:
                                negative_loops += 1
                except:
                    # Skip if cycle detection fails
                    continue

            return positive_loops, negative_loops

        except Exception as e:
            logger.warning(f"Error detecting feedback loops: {e}")
            return 0, 0

    def _compute_stability_index(self, G: nx.Graph) -> float:
        """Compute overall stability index."""
        try:
            # Combine multiple stability indicators

            # 1. Balance contributes to stability
            balance = self._compute_structural_balance(G)

            # 2. Low frustration contributes to stability
            frustration = self._compute_edge_frustration(G)
            frustration_score = 1.0 - frustration

            # 3. Moderate density contributes to stability
            density = nx.density(G)
            density_score = 1.0 - abs(density - 0.5) * 2  # Optimal around 0.5

            # 4. Positive feedback loops reduce stability
            pos_loops, neg_loops = self._detect_feedback_loops(G)
            total_loops = pos_loops + neg_loops
            feedback_score = 1.0 - min(total_loops / (len(G.nodes()) + 1), 1.0)

            # Weighted combination
            stability = 0.3 * balance + 0.3 * frustration_score + 0.2 * density_score + 0.2 * feedback_score

            return float(stability)

        except Exception as e:
            logger.warning(f"Error computing stability index: {e}")
            return 0.5

    def _estimate_convergence_likelihood(self, G: nx.Graph) -> float:
        """Estimate likelihood of network convergence."""
        try:
            # Based on signed network theory and dynamics

            # High balance increases convergence likelihood
            balance = self._compute_structural_balance(G)

            # Low conflict intensity increases convergence
            conflict = self._compute_conflict_intensity(G)
            conflict_score = 1.0 - conflict

            # Moderate connectivity helps convergence
            avg_degree = sum(dict(G.degree()).values()) / len(G.nodes()) if len(G.nodes()) > 0 else 0
            connectivity_score = min(avg_degree / len(G.nodes()), 1.0) if len(G.nodes()) > 0 else 0

            # Combine factors
            convergence = 0.4 * balance + 0.3 * conflict_score + 0.3 * connectivity_score

            return float(convergence)

        except Exception as e:
            logger.warning(f"Error estimating convergence likelihood: {e}")
            return 0.5

    def _analyze_dynamic_signs(self, data: np.ndarray) -> dict[str, Any]:
        """Analyze dynamic sign changes over time/contexts."""
        try:
            n_contexts, n_neurons = data.shape

            if n_contexts < 10:
                return {"temporal_sign_stability": 1.0, "sign_change_patterns": []}

            # Divide into temporal windows
            window_size = max(5, n_contexts // 5)
            n_windows = n_contexts // window_size

            if n_windows < 2:
                return {"temporal_sign_stability": 1.0, "sign_change_patterns": []}

            # Compute correlation matrices for each window
            window_correlations = []
            for i in range(n_windows):
                start_idx = i * window_size
                end_idx = min((i + 1) * window_size, n_contexts)
                window_data = data[start_idx:end_idx]

                if window_data.shape[0] > 1:
                    corr_matrix = np.corrcoef(window_data.T)
                    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
                    window_correlations.append(corr_matrix)

            if len(window_correlations) < 2:
                return {"temporal_sign_stability": 1.0, "sign_change_patterns": []}

            # Analyze sign stability across windows
            sign_changes = []
            for i in range(len(window_correlations) - 1):
                curr_signs = np.sign(window_correlations[i])
                next_signs = np.sign(window_correlations[i + 1])

                # Count sign changes
                changes = np.sum(curr_signs != next_signs)
                total_possible = curr_signs.size - curr_signs.shape[0]  # Exclude diagonal
                change_rate = changes / total_possible if total_possible > 0 else 0.0
                sign_changes.append(change_rate)

            # Overall stability
            avg_stability = 1.0 - np.mean(sign_changes) if sign_changes else 1.0

            # Pattern analysis
            change_patterns = self._analyze_sign_change_patterns(window_correlations)

            return {
                "temporal_sign_stability": float(avg_stability),
                "sign_change_rates": [float(x) for x in sign_changes],
                "sign_change_patterns": change_patterns,
                "n_temporal_windows": len(window_correlations),
            }

        except Exception as e:
            logger.warning(f"Error analyzing dynamic signs: {e}")
            return {"temporal_sign_stability": 1.0, "sign_change_patterns": []}

    def _analyze_sign_change_patterns(self, window_correlations: list[np.ndarray]) -> dict[str, Any]:
        """Analyze patterns in sign changes across temporal windows."""
        try:
            if len(window_correlations) < 2:
                return {}

            patterns = {}

            # Track specific edge sign trajectories
            n_nodes = window_correlations[0].shape[0]

            # Count different types of changes
            persistent_positive = 0
            persistent_negative = 0
            oscillating = 0
            trending_positive = 0
            trending_negative = 0

            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    # Extract sign trajectory for this edge
                    signs = [np.sign(corr[i, j]) for corr in window_correlations]

                    # Classify pattern
                    if all(s > 0 for s in signs):
                        persistent_positive += 1
                    elif all(s < 0 for s in signs):
                        persistent_negative += 1
                    elif len(set(signs)) > 2:  # More than just positive/negative
                        oscillating += 1
                    elif signs[0] <= 0 < signs[-1]:
                        trending_positive += 1
                    elif signs[0] >= 0 > signs[-1]:
                        trending_negative += 1

            total_edges = (n_nodes * (n_nodes - 1)) // 2

            patterns["persistent_positive_ratio"] = persistent_positive / total_edges if total_edges > 0 else 0.0
            patterns["persistent_negative_ratio"] = persistent_negative / total_edges if total_edges > 0 else 0.0
            patterns["oscillating_ratio"] = oscillating / total_edges if total_edges > 0 else 0.0
            patterns["trending_positive_ratio"] = trending_positive / total_edges if total_edges > 0 else 0.0
            patterns["trending_negative_ratio"] = trending_negative / total_edges if total_edges > 0 else 0.0

            return patterns

        except Exception as e:
            logger.warning(f"Error analyzing sign change patterns: {e}")
            return {}

    def _generate_signed_null_model_distribution(
        self, data: np.ndarray, n_samples: int = 100
    ) -> dict[str, list[float]]:
        """Generate null model distribution that preserves sign distributions."""
        null_metrics = defaultdict(list)

        try:
            n_contexts, n_neurons = data.shape

            for _ in range(n_samples):
                if self.preserve_edge_signs:
                    # Generate null data preserving correlation structure but randomizing signs
                    null_data = np.zeros_like(data)

                    # First, permute each neuron's activations independently
                    for i in range(n_neurons):
                        null_data[:, i] = np.random.permutation(data[:, i])

                    # Then randomly flip signs of correlations to preserve sign distribution
                    null_edge_matrix = self._construct_graph_edges(null_data)

                    # Randomly flip signs while preserving the proportion of positive/negative edges
                    original_edge_matrix = self._construct_graph_edges(data)
                    original_pos_ratio = np.mean(original_edge_matrix > 0)

                    # Create sign-preserving null model
                    abs_null_edges = np.abs(null_edge_matrix)
                    edge_mask = abs_null_edges > 0
                    n_edges = np.sum(edge_mask)

                    if n_edges > 0:
                        # Randomly assign signs to preserve original distribution
                        n_positive = int(original_pos_ratio * n_edges)
                        signs = np.concatenate([np.ones(n_positive), -np.ones(n_edges - n_positive)])
                        np.random.shuffle(signs)

                        sign_matrix = np.zeros_like(null_edge_matrix)
                        sign_matrix[edge_mask] = signs
                        null_edge_matrix = abs_null_edges * sign_matrix
                else:
                    # Standard permutation for unsigned networks
                    null_data = np.zeros_like(data)
                    for i in range(n_neurons):
                        null_data[:, i] = np.random.permutation(data[:, i])
                    null_edge_matrix = self._construct_graph_edges(null_data)

                # Construct null graph and compute metrics
                null_G = self._construct_networkx_graph(null_edge_matrix)
                null_graph_metrics = self._compute_network_topology_metrics(null_G)

                # Store metrics
                for metric, value in null_graph_metrics.items():
                    if isinstance(value, (int, float)):
                        null_metrics[metric].append(value)

        except Exception as e:
            logger.warning(f"Error generating signed null model distribution: {e}")

        return dict(null_metrics)

    def statistical_validation(self) -> dict[str, Any]:
        """Perform statistical validation with enhanced signed network support."""
        validation_results = {}

        for group_name, tensor in self.activation_tensors.items():
            data = tensor.detach().cpu().numpy()

            # Construct observed graph
            observed_edge_matrix = self._construct_graph_edges(data)
            observed_G = self._construct_networkx_graph(observed_edge_matrix)
            observed_metrics = self._compute_network_topology_metrics(observed_G)

            # Generate appropriate null model distributions
            if self.preserve_edge_signs:
                null_metrics = self._generate_signed_null_model_distribution(data, n_samples=100)
            else:
                null_metrics = self._generate_null_model_distribution(data, n_samples=100)

            # Compute p-values and effect sizes
            statistical_tests = self._compute_statistical_tests(observed_metrics, null_metrics)

            # Enhanced signed network specific tests
            if self.preserve_edge_signs and observed_G.graph.get("signed", False):
                signed_tests = self._compute_signed_statistical_tests(observed_G, data)
                statistical_tests.update(signed_tests)

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
                "statistical_tests": statistical_tests,
                "significant_metrics": [
                    metric
                    for metric, result in statistical_tests.items()
                    if result.get("corrected_p_value", 1.0) < self.significance_level
                ],
            }

        return validation_results

    def _compute_statistical_tests(self, observed_metrics: dict, null_metrics: dict) -> dict[str, dict]:
        """Compute statistical tests comparing observed vs null metrics."""
        tests = {}

        try:
            for metric in observed_metrics:
                if metric in null_metrics and isinstance(observed_metrics[metric], (int, float)):
                    null_values = null_metrics[metric]
                    observed_value = observed_metrics[metric]

                    if len(null_values) > 0:
                        # Two-tailed p-value
                        p_value = np.mean(
                            np.abs(null_values - np.mean(null_values)) >= abs(observed_value - np.mean(null_values))
                        )

                        # Effect size (Cohen's d)
                        if np.std(null_values) > 0:
                            effect_size = (observed_value - np.mean(null_values)) / np.std(null_values)
                        else:
                            effect_size = 0.0

                        # Bonferroni correction will be applied later
                        tests[metric] = {
                            "p_value": float(p_value),
                            "effect_size": float(effect_size),
                            "observed_value": float(observed_value),
                            "null_mean": float(np.mean(null_values)),
                            "null_std": float(np.std(null_values)),
                        }
                    else:
                        tests[metric] = {
                            "p_value": 1.0,
                            "effect_size": 0.0,
                            "observed_value": float(observed_value),
                            "null_mean": 0.0,
                            "null_std": 0.0,
                        }

            # Apply multiple comparison correction
            if tests:
                p_values = [test["p_value"] for test in tests.values()]
                corrected_p_values = [min(1.0, p * len(p_values)) for p in p_values]

                for i, metric in enumerate(tests.keys()):
                    tests[metric]["corrected_p_value"] = corrected_p_values[i]

            return tests

        except Exception as e:
            logger.warning(f"Error computing statistical tests: {e}")
            return {}

    def _compute_signed_statistical_tests(self, G: nx.Graph, data: np.ndarray) -> dict[str, dict]:
        """Compute statistical tests specific to signed networks."""
        signed_tests = {}

        try:
            # Test balance against random signed networks
            observed_balance = self._compute_structural_balance(G)

            # Generate random signed networks with same edge density and sign ratio
            random_balances = []
            pos_edges, neg_edges, _ = self._analyze_edge_signs(G)
            total_edges = pos_edges + neg_edges

            if total_edges > 0:
                pos_ratio = pos_edges / total_edges

                for _ in range(100):
                    # Create random signed graph with same properties
                    random_G = self._generate_random_signed_graph(len(G.nodes()), total_edges, pos_ratio)
                    random_balance = self._compute_structural_balance(random_G)
                    random_balances.append(random_balance)

                if random_balances:
                    balance_p_value = np.mean(
                        np.abs(random_balances - np.mean(random_balances))
                        >= abs(observed_balance - np.mean(random_balances))
                    )

                    balance_effect_size = (
                        (observed_balance - np.mean(random_balances)) / np.std(random_balances)
                        if np.std(random_balances) > 0
                        else 0.0
                    )

                    signed_tests["balance_significance"] = {
                        "p_value": float(balance_p_value),
                        "effect_size": float(balance_effect_size),
                        "observed_balance": float(observed_balance),
                        "random_balance_mean": float(np.mean(random_balances)),
                    }

            # Test frustration significance
            observed_frustration = self._compute_edge_frustration(G)
            signed_tests["frustration_significance"] = {
                "observed_frustration": float(observed_frustration),
                "frustration_level": "low"
                if observed_frustration < 0.1
                else "medium"
                if observed_frustration < 0.3
                else "high",
            }

            # Test excitation-inhibition balance
            if "edge_balance_ratio" in G.graph or hasattr(G, "graph"):
                try:
                    edge_balance = self._analyze_edge_signs(G)[2]  # Balance ratio

                    # Test against expected 50-50 balance
                    balance_deviation = abs(edge_balance - 0.5)

                    signed_tests["ei_balance_test"] = {
                        "observed_balance": float(edge_balance),
                        "deviation_from_neutral": float(balance_deviation),
                        "significantly_imbalanced": balance_deviation > 0.2,
                    }
                except:
                    pass

            return signed_tests

        except Exception as e:
            logger.warning(f"Error computing signed statistical tests: {e}")
            return {}

    def _generate_random_signed_graph(self, n_nodes: int, n_edges: int, pos_ratio: float) -> nx.Graph:
        """Generate random signed graph with specified properties."""
        try:
            G = nx.Graph()
            G.add_nodes_from(range(n_nodes))

            # Generate random edges
            possible_edges = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
            n_edges = min(n_edges, len(possible_edges))

            selected_edges = np.random.choice(len(possible_edges), size=n_edges, replace=False)

            # Assign signs
            n_positive = int(pos_ratio * n_edges)
            signs = [1.0] * n_positive + [-1.0] * (n_edges - n_positive)
            np.random.shuffle(signs)

            for i, edge_idx in enumerate(selected_edges):
                u, v = possible_edges[edge_idx]
                G.add_edge(u, v, weight=signs[i])

            G.graph["signed"] = True
            return G

        except Exception as e:
            logger.warning(f"Error generating random signed graph: {e}")
            return nx.Graph()

    def run_all_analyses(self) -> dict[str, Any]:
        """Main analysis method that runs all graph-based analyses with signed network support."""
        logger.info("Starting comprehensive graph-based coordination analysis...")

        try:
            # Run all hypothesis tests
            hierarchical_results = self.analyze_hierarchical_network_organization()
            modular_results = self.analyze_modular_coordination_architecture()
            context_dependent_results = self.analyze_context_dependent_network_topology()
            optimization_results = self.analyze_optimized_information_flow()
            dynamic_results = self.analyze_dynamic_network_properties()

            # Advanced signed network analysis
            if self.preserve_edge_signs:
                signed_results = self.analyze_advanced_signed_network_properties()
            else:
                signed_results = {}

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
                "advanced_signed_network_properties": signed_results,
                "comparative_analysis": comparative_results,
                "statistical_validation": validation_results,
                "analysis_metadata": {
                    "edge_construction_method": self.edge_construction_method,
                    "graph_type": self.graph_type,
                    "preserve_edge_signs": self.preserve_edge_signs,
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
                "summary": self._generate_enhanced_analysis_summary(
                    hierarchical_results,
                    modular_results,
                    context_dependent_results,
                    optimization_results,
                    signed_results,
                ),
            }

            # Store results
            self.graph_results = results

            logger.info("Comprehensive graph-based coordination analysis completed successfully")
            return results

        except Exception as e:
            logger.error(f"Error in graph analysis: {e}")
            return {
                "error": str(e),
                "analysis_metadata": {
                    "edge_construction_method": self.edge_construction_method,
                    "graph_type": self.graph_type,
                    "preserve_edge_signs": self.preserve_edge_signs,
                    "device": self.device,
                },
            }

    def _generate_enhanced_analysis_summary(
        self,
        hierarchical_results: dict,
        modular_results: dict,
        context_dependent_results: dict,
        optimization_results: dict,
        signed_results: dict,
    ) -> dict[str, Any]:
        """Generate enhanced summary including signed network insights."""
        summary = {
            "hypothesis_support": {},
            "key_findings": {},
            "group_comparisons": {},
            "signed_network_insights": {},
        }

        try:
            # Standard hypothesis support analysis
            summary["hypothesis_support"] = self._analyze_hypothesis_support(
                hierarchical_results, modular_results, context_dependent_results, optimization_results
            )

            # Enhanced key findings
            summary["key_findings"] = self._extract_key_findings(
                hierarchical_results, modular_results, context_dependent_results, optimization_results
            )

            # Group comparisons
            summary["group_comparisons"] = self._compare_groups(
                hierarchical_results, modular_results, context_dependent_results, optimization_results
            )

            # Signed network specific insights
            if self.preserve_edge_signs and signed_results:
                summary["signed_network_insights"] = self._extract_signed_insights(signed_results)

            return summary

        except Exception as e:
            logger.warning(f"Error generating enhanced summary: {e}")
            summary["error"] = str(e)
            return summary

    def _analyze_hypothesis_support(
        self, h1_results: dict, h2_results: dict, h3_results: dict, h4_results: dict
    ) -> dict:
        """Analyze support for each hypothesis."""
        try:
            # H1: Hierarchical Organization
            h1_support = {
                group: results.get("supports_hierarchy_hypothesis", False) for group, results in h1_results.items()
            }

            # H2: Modular Architecture
            h2_support = {
                group: results.get("supports_modularity_hypothesis", False) for group, results in h2_results.items()
            }

            # H3: Context-Dependent Topology
            h3_support = {
                group: results.get("supports_adaptive_topology_hypothesis", False)
                for group, results in h3_results.items()
            }

            # H4: Optimized Information Flow
            h4_support = {
                group: results.get("supports_optimization_hypothesis", False) for group, results in h4_results.items()
            }

            return {
                "H1_hierarchical_organization": {
                    "overall_support": any(h1_support.values()),
                    "group_support": h1_support,
                    "evidence": "Enhanced hierarchy with excitatory/inhibitory hubs"
                    if any(h1_support.values())
                    else "Limited hierarchical structure",
                },
                "H2_modular_architecture": {
                    "overall_support": any(h2_support.values()),
                    "group_support": h2_support,
                    "evidence": "Signed modularity with balanced communities"
                    if any(h2_support.values())
                    else "Limited modular structure",
                },
                "H3_context_dependent_topology": {
                    "overall_support": any(h3_support.values()),
                    "group_support": h3_support,
                    "evidence": "Adaptive topology with sign changes"
                    if any(h3_support.values())
                    else "Similar topology across contexts",
                },
                "H4_optimized_information_flow": {
                    "overall_support": any(h4_support.values()),
                    "group_support": h4_support,
                    "evidence": "Optimized excitation-inhibition balance"
                    if any(h4_support.values())
                    else "Suboptimal information flow",
                },
            }
        except Exception as e:
            logger.warning(f"Error analyzing hypothesis support: {e}")
            return {}

    def _extract_key_findings(self, h1_results: dict, h2_results: dict, h3_results: dict, h4_results: dict) -> dict:
        """Extract key findings across all analyses."""
        try:
            findings = {}

            # Best examples of each property
            if h1_results:
                best_hierarchy = max(
                    h1_results.items(),
                    key=lambda x: x[1].get("hierarchy_score", 0),
                    default=("none", {"hierarchy_score": 0}),
                )
                findings["strongest_hierarchy"] = {
                    "group": best_hierarchy[0],
                    "score": best_hierarchy[1].get("hierarchy_score", 0),
                }

            if h2_results:
                best_modularity = max(
                    h2_results.items(),
                    key=lambda x: x[1].get("modularity_analysis", {}).get("signed_modularity", 0),
                    default=("none", {"modularity_analysis": {"signed_modularity": 0}}),
                )
                findings["highest_modularity"] = {
                    "group": best_modularity[0],
                    "score": best_modularity[1].get("modularity_analysis", {}).get("signed_modularity", 0),
                }

            return findings

        except Exception as e:
            logger.warning(f"Error extracting key findings: {e}")
            return {}
