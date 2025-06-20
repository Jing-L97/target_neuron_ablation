import logging
import typing as t
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import permutation_test, ttest_ind
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors

from neuron_analyzer.load_util import cleanup

logger = logging.getLogger(__name__)


T = t.TypeVar("T")
ComputationType = t.Literal["within", "between"]
GroupType = t.Literal["boost", "suppress", "random_1", "random_2"]


class MutualInformationAnalyzer:
    """Analyzes mutual information coordination patterns between neuron groups."""

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
        mi_estimator: str = "ksg",  # "ksg", "adaptive", "sklearn"
        max_lag: int = 3,
        mi_batch_size: int = 1000,
        significance_level: float = 0.05,
    ):
        """Initialize the mutual information analyzer."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mixed_precision = use_mixed_precision
        self.dtype = torch.float16 if use_mixed_precision else torch.float32

        # Store configuration
        self.mi_estimator = mi_estimator
        self.max_lag = max_lag
        self.mi_batch_size = mi_batch_size
        self.significance_level = significance_level

        # Store input data and parameters
        self.data = activation_data.copy()
        self.boost_neuron_indices = boost_neuron_indices
        self.suppress_neuron_indices = suppress_neuron_indices
        self.excluded_neuron_indices = excluded_neuron_indices
        self.activation_column = activation_column
        self.token_column = token_column
        self.context_column = context_column
        self.component_column = component_column
        self.num_random_groups = num_random_groups

        # Create token-context identifier (same as original class)
        self.data["token_context_id"] = (
            self.data[token_column].astype(str) + "_" + self.data[context_column].astype(str)
        )

        # Extract unique identifiers
        self.token_contexts = self.data["token_context_id"].unique()
        self.all_neuron_indices = self.data[self.component_column].astype(int).unique()

        # Generate random groups (reuse original logic)
        self.random_groups, self.random_indices = self._generate_random_groups()

        # Create activation matrices
        self._create_activation_tensors()

        # Handle rare token masking
        self.rare_token_mask = self._create_rare_token_mask(rare_token_mask)

        # Results storage
        self.mi_results = {}

        logger.info(
            f"Initialized MI analyzer with {len(self.token_contexts)} contexts, "
            f"device: {self.device}, estimator: {self.mi_estimator}"
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
        if self.mi_estimator == "sklearn":
            # Use sklearn's mutual information estimator
            X_reshaped = X.reshape(-1, 1) if X.ndim == 1 else X
            mi = mutual_info_regression(X_reshaped, Y, discrete_features=False, random_state=42)
            return float(mi[0] if len(mi) == 1 else np.mean(mi))

        if self.mi_estimator == "ksg":
            # Kraskov-Stögbauer-Grassberger estimator (simplified)
            return self._ksg_estimator(X, Y)

        if self.mi_estimator == "adaptive":
            # Adaptive binning estimator
            return self._adaptive_mi_estimator(X, Y)

        raise ValueError(f"Unknown MI estimator: {self.mi_estimator}")

    def _ksg_estimator(self, X: np.ndarray, Y: np.ndarray, k: int = 3) -> float:
        """Simplified KSG mutual information estimator."""
        try:
            from scipy.special import digamma

            X = X.reshape(-1, 1) if X.ndim == 1 else X
            Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y

            n = len(X)
            if n < k + 1:
                return 0.0

            # Combine X and Y
            XY = np.concatenate([X, Y], axis=1)

            # Find k-th nearest neighbors in joint space
            nbrs_xy = NearestNeighbors(n_neighbors=k + 1, metric="chebyshev").fit(XY)
            distances_xy, _ = nbrs_xy.kneighbors(XY)
            eps = distances_xy[:, k]  # Distance to k-th neighbor

            # Count neighbors in marginal spaces
            nx = np.array([np.sum(np.max(np.abs(X - X[i]), axis=1) < eps[i]) - 1 for i in range(n)])
            ny = np.array([np.sum(np.max(np.abs(Y - Y[i]), axis=1) < eps[i]) - 1 for i in range(n)])

            # KSG formula
            mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n)
            return max(0.0, float(mi))

        except Exception as e:
            logger.warning(f"KSG estimator failed: {e}, falling back to adaptive")
            return self._adaptive_mi_estimator(X, Y)

    def _adaptive_mi_estimator(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Adaptive binning mutual information estimator."""
        try:
            # Determine optimal number of bins using Freedman-Diaconis rule
            n = len(X)
            if n < 10:
                return 0.0

            def optimal_bins(data):
                q75, q25 = np.percentile(data, [75, 25])
                iqr = q75 - q25
                if iqr == 0:
                    return min(10, int(np.sqrt(n)))
                h = 2 * iqr * (n ** (-1 / 3))
                return max(5, min(50, int((np.max(data) - np.min(data)) / h)))

            bins_x = optimal_bins(X)
            bins_y = optimal_bins(Y)

            # Create 2D histogram
            hist_xy, x_edges, y_edges = np.histogram2d(X, Y, bins=(bins_x, bins_y))
            hist_x = np.histogram(X, bins=x_edges)[0]
            hist_y = np.histogram(Y, bins=y_edges)[0]

            # Convert to probabilities
            p_xy = hist_xy / n
            p_x = hist_x / n
            p_y = hist_y / n

            # Calculate MI
            mi = 0.0
            for i in range(len(p_x)):
                for j in range(len(p_y)):
                    if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                        mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))

            return max(0.0, float(mi))

        except Exception as e:
            logger.warning(f"Adaptive MI estimator failed: {e}")
            return 0.0

    def _compute_pairwise_mi(self, tensor: torch.Tensor, context_mask: np.ndarray | None = None) -> np.ndarray:
        """Compute pairwise mutual information matrix for a tensor."""
        # Convert to numpy and apply context mask
        data = tensor.detach().cpu().numpy()
        if context_mask is not None:
            data = data[context_mask]

        n_neurons = data.shape[1]
        mi_matrix = np.zeros((n_neurons, n_neurons))

        # Compute pairwise MI
        for i in range(n_neurons):
            for j in range(i, n_neurons):
                if i == j:
                    mi_matrix[i, j] = 0.0  # Self-MI not meaningful for our analysis
                else:
                    mi_val = self._estimate_mutual_information(data[:, i], data[:, j])
                    mi_matrix[i, j] = mi_val
                    mi_matrix[j, i] = mi_val  # Symmetric

        return mi_matrix

    def _compute_lagged_mi(self, tensor: torch.Tensor, lag: int, context_mask: np.ndarray | None = None) -> np.ndarray:
        """Compute mutual information between neurons at different time lags."""
        data = tensor.detach().cpu().numpy()
        if context_mask is not None:
            data = data[context_mask]

        n_contexts, n_neurons = data.shape
        if n_contexts <= lag:
            return np.zeros((n_neurons, n_neurons))

        # Create lagged data
        data_t = data[:-lag] if lag > 0 else data
        data_t_lag = data[lag:] if lag > 0 else data

        mi_matrix = np.zeros((n_neurons, n_neurons))

        for i in range(n_neurons):
            for j in range(n_neurons):
                mi_val = self._estimate_mutual_information(data_t[:, i], data_t_lag[:, j])
                mi_matrix[i, j] = mi_val

        return mi_matrix

    def _compute_interaction_information(self, data: np.ndarray) -> float:
        """Compute interaction information for three variables."""
        if data.shape[1] != 3:
            raise ValueError("Interaction information requires exactly 3 variables")

        x, y, z = data[:, 0], data[:, 1], data[:, 2]

        # I(X;Y;Z) = I(X;Y|Z) - I(X;Y)
        mi_xy = self._estimate_mutual_information(x, y)

        # For conditional MI, we use residuals after linear regression
        # This is an approximation for simplicity

        reg_x = LinearRegression().fit(z.reshape(-1, 1), x)
        reg_y = LinearRegression().fit(z.reshape(-1, 1), y)

        x_residual = x - reg_x.predict(z.reshape(-1, 1))
        y_residual = y - reg_y.predict(z.reshape(-1, 1))

        mi_xy_given_z = self._estimate_mutual_information(x_residual, y_residual)

        return mi_xy_given_z - mi_xy

    def analyze_context_dependent_mi(self) -> dict[str, Any]:
        """Analyze H1: Context-dependent coordination."""
        results = {}

        for group_name, tensor in self.activation_tensors.items():
            # Compute MI for rare and common contexts separately
            rare_mask = self.rare_token_mask
            common_mask = ~self.rare_token_mask

            if np.sum(rare_mask) < 10 or np.sum(common_mask) < 10:
                logger.warning(f"Insufficient contexts for {group_name}, skipping context analysis")
                results[group_name] = {
                    "mi_rare": np.array([[0.0]]),
                    "mi_common": np.array([[0.0]]),
                    "context_effect_size": 0.0,
                    "p_value": 1.0,
                }
                continue

            mi_rare = self._compute_pairwise_mi(tensor, rare_mask)
            mi_common = self._compute_pairwise_mi(tensor, common_mask)

            # Statistical test for context effect
            rare_values = mi_rare[np.triu_indices_from(mi_rare, k=1)]
            common_values = mi_common[np.triu_indices_from(mi_common, k=1)]

            if len(rare_values) > 0 and len(common_values) > 0:
                # Permutation test for difference
                def statistic(x, y):
                    return np.mean(x) - np.mean(y)

                try:
                    perm_result = permutation_test(
                        (rare_values, common_values), statistic, n_resamples=1000, random_state=42
                    )
                    p_value = perm_result.pvalue
                except:
                    # Fallback to t-test
                    _, p_value = ttest_ind(rare_values, common_values)

                effect_size = np.mean(rare_values) - np.mean(common_values)
            else:
                p_value = 1.0
                effect_size = 0.0

            results[group_name] = {
                "mi_rare": mi_rare,
                "mi_common": mi_common,
                "mean_mi_rare": float(np.mean(rare_values)) if len(rare_values) > 0 else 0.0,
                "mean_mi_common": float(np.mean(common_values)) if len(common_values) > 0 else 0.0,
                "context_effect_size": float(effect_size),
                "p_value": float(p_value),
                "is_significant": p_value < self.significance_level,
            }

        return results

    def analyze_nonlinear_coordination(self) -> dict[str, Any]:
        """Analyze H2: Nonlinear coordination patterns."""
        results = {}

        for group_name, tensor in self.activation_tensors.items():
            # Compute MI matrix
            mi_matrix = self._compute_pairwise_mi(tensor)

            # Compute correlation matrix for comparison
            data = tensor.detach().cpu().numpy()
            corr_matrix = np.corrcoef(data.T)
            corr_matrix = np.abs(corr_matrix)  # Use absolute correlation

            # Extract upper triangular values
            upper_indices = np.triu_indices_from(mi_matrix, k=1)
            mi_values = mi_matrix[upper_indices]
            corr_values = corr_matrix[upper_indices]

            # Compute nonlinearity indices
            # Avoid division by zero
            nonzero_mask = corr_values > 1e-6
            nonlinearity_ratios = np.zeros_like(mi_values)
            nonlinearity_ratios[nonzero_mask] = mi_values[nonzero_mask] / corr_values[nonzero_mask]

            # Expected MI from correlation (rough approximation)
            # For Gaussian variables: MI ≈ -0.5 * log(1 - corr^2)
            expected_mi = -0.5 * np.log(1 - corr_values**2 + 1e-10)
            nonlinearity_excess = mi_values - expected_mi

            results[group_name] = {
                "mi_matrix": mi_matrix,
                "correlation_matrix": corr_matrix,
                "mean_mi": float(np.mean(mi_values)),
                "mean_correlation": float(np.mean(corr_values)),
                "mean_nonlinearity_ratio": float(np.mean(nonlinearity_ratios)),
                "mean_nonlinearity_excess": float(np.mean(nonlinearity_excess)),
                "nonlinearity_strength": float(np.mean(nonlinearity_excess > 0)),
            }

        return results

    def analyze_temporal_coordination(self) -> dict[str, Any]:
        """Analyze H3: Multi-scale temporal coordination."""
        results = {}

        for group_name, tensor in self.activation_tensors.items():
            lag_results = {}

            for lag in range(self.max_lag + 1):
                mi_matrix = self._compute_lagged_mi(tensor, lag)
                upper_values = mi_matrix[np.triu_indices_from(mi_matrix, k=1)]

                lag_results[f"lag_{lag}"] = {
                    "mi_matrix": mi_matrix,
                    "mean_mi": float(np.mean(upper_values)),
                    "max_mi": float(np.max(upper_values)),
                    "std_mi": float(np.std(upper_values)),
                }

            # Identify temporal patterns
            lag_means = [lag_results[f"lag_{lag}"]["mean_mi"] for lag in range(self.max_lag + 1)]
            peak_lag = int(np.argmax(lag_means))
            temporal_decay = lag_means[0] - lag_means[-1] if len(lag_means) > 1 else 0.0

            results[group_name] = {
                "lag_results": lag_results,
                "peak_lag": peak_lag,
                "temporal_decay": float(temporal_decay),
                "has_temporal_structure": peak_lag > 0 or temporal_decay > 0.01,
            }

        return results

    def analyze_collective_coordination(self) -> dict[str, Any]:
        """Analyze H4: Collective coordination."""
        results = {}

        for group_name, tensor in self.activation_tensors.items():
            data = tensor.detach().cpu().numpy()
            n_neurons = data.shape[1]

            # Sample triplets for interaction information
            max_triplets = min(50, (n_neurons * (n_neurons - 1) * (n_neurons - 2)) // 6)

            if n_neurons < 3:
                results[group_name] = {
                    "interaction_information": [],
                    "mean_interaction_info": 0.0,
                    "collective_coordination_strength": 0.0,
                }
                continue

            triplet_indices = list(combinations(range(n_neurons), 3))
            if len(triplet_indices) > max_triplets:
                triplet_indices = np.random.choice(len(triplet_indices), max_triplets, replace=False)
                triplet_indices = [list(combinations(range(n_neurons), 3))[i] for i in triplet_indices]

            interaction_infos = []
            for i, j, k in triplet_indices:
                try:
                    ii_value = self._compute_interaction_information(data[:, [i, j, k]])
                    interaction_infos.append(ii_value)
                except:
                    interaction_infos.append(0.0)

            # Compute collective measures
            mean_interaction = np.mean(interaction_infos) if interaction_infos else 0.0
            collective_strength = np.mean(np.array(interaction_infos) > 0) if interaction_infos else 0.0

            results[group_name] = {
                "interaction_information": interaction_infos,
                "mean_interaction_info": float(mean_interaction),
                "collective_coordination_strength": float(collective_strength),
                "n_triplets_analyzed": len(interaction_infos),
            }

        return results

    def statistical_comparison_tests(self) -> dict[str, Any]:
        """Compare MI patterns between neuron groups."""
        tests = {}

        # Compare boost vs random
        groups_to_compare = [
            ("boost", "random_1"),
            ("suppress", "random_1"),
            ("boost", "suppress"),
            ("random_1", "random_2"),
        ]

        for group1, group2 in groups_to_compare:
            # Get MI values for comparison
            tensor1 = self.activation_tensors[group1]
            tensor2 = self.activation_tensors[group2]

            mi1 = self._compute_pairwise_mi(tensor1)
            mi2 = self._compute_pairwise_mi(tensor2)

            values1 = mi1[np.triu_indices_from(mi1, k=1)]
            values2 = mi2[np.triu_indices_from(mi2, k=1)]

            # Statistical test
            if len(values1) > 0 and len(values2) > 0:
                try:
                    _, p_value = ttest_ind(values1, values2)
                    effect_size = np.mean(values1) - np.mean(values2)
                    cohens_d = effect_size / np.sqrt((np.var(values1) + np.var(values2)) / 2)
                except:
                    p_value = 1.0
                    effect_size = 0.0
                    cohens_d = 0.0
            else:
                p_value = 1.0
                effect_size = 0.0
                cohens_d = 0.0

            tests[f"{group1}_vs_{group2}"] = {
                "mean_mi_group1": float(np.mean(values1)) if len(values1) > 0 else 0.0,
                "mean_mi_group2": float(np.mean(values2)) if len(values2) > 0 else 0.0,
                "effect_size": float(effect_size),
                "cohens_d": float(cohens_d),
                "p_value": float(p_value),
                "is_significant": p_value < self.significance_level,
            }

        return tests

    def run_all_analyses(self) -> dict[str, Any]:
        """Main analysis method that runs all MI analyses."""
        logger.info("Starting mutual information analysis...")

        # Run all hypothesis tests
        context_results = self.analyze_context_dependent_mi()
        nonlinear_results = self.analyze_nonlinear_coordination()
        temporal_results = self.analyze_temporal_coordination()
        collective_results = self.analyze_collective_coordination()
        statistical_tests = self.statistical_comparison_tests()

        # Compile results
        results = {
            "context_dependent_coordination": context_results,
            "nonlinear_coordination_patterns": nonlinear_results,
            "temporal_coordination": temporal_results,
            "collective_coordination": collective_results,
            "statistical_comparisons": statistical_tests,
            "analysis_metadata": {
                "mi_estimator": self.mi_estimator,
                "max_lag": self.max_lag,
                "significance_level": self.significance_level,
                "n_contexts_total": len(self.token_contexts),
                "n_contexts_rare": int(np.sum(self.rare_token_mask)),
                "n_contexts_common": int(np.sum(~self.rare_token_mask)),
                "neuron_group_sizes": {name: len(indices) for name, indices in self.neuron_indices.items()},
            },
        }

        # Store results
        self.mi_results = results

        # Clean up GPU memory
        cleanup()

        logger.info("Mutual information analysis completed")
        return results
