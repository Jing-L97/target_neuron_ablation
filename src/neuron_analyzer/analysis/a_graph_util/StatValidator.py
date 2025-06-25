import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
from sklearn.utils import resample

logger = logging.getLogger(__name__)


@dataclass
class StatisticalTest:
    """Container for statistical test results."""

    test_name: str
    p_value: float
    effect_size: float
    confidence_interval: tuple[float, float]
    is_significant: bool
    test_statistic: float
    description: str


@dataclass
class ValidationResult:
    """Container for comprehensive validation results."""

    metric_name: str
    observed_value: float
    null_distribution_stats: dict[str, float]
    statistical_tests: list[StatisticalTest]
    is_significant: bool
    effect_size_interpretation: str
    summary: str


class BaseNullModelGenerator(ABC):
    """Abstract base class for null model generation."""

    def __init__(self, n_samples: int = 1000, random_seed: int = 42):
        self.n_samples = n_samples
        self.random_seed = random_seed
        np.random.seed(random_seed)

    @abstractmethod
    def generate_null_samples(self, data: np.ndarray, **kwargs) -> list[dict[str, float]]:
        """Generate null model samples."""


class PermutationNullModel(BaseNullModelGenerator):
    """Generate null models by permuting node labels."""

    def generate_null_samples(self, data: np.ndarray, **kwargs) -> list[dict[str, float]]:
        """Generate null samples by permuting rows (contexts) independently for each neuron."""
        null_samples = []
        n_contexts, n_neurons = data.shape

        for _ in range(self.n_samples):
            # Create permuted data
            null_data = np.zeros_like(data)
            for neuron in range(n_neurons):
                null_data[:, neuron] = np.random.permutation(data[:, neuron])

            # Compute metrics for this null sample
            metrics = self._compute_null_metrics(null_data)
            null_samples.append(metrics)

        return null_samples

    def _compute_null_metrics(self, data: np.ndarray) -> dict[str, float]:
        """Compute basic network metrics for null model."""
        try:
            # Compute correlation matrix
            if data.shape[0] < 2:
                return {"density": 0.0, "avg_clustering": 0.0, "global_efficiency": 0.0}

            corr_matrix = np.corrcoef(data.T)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

            # Create simple thresholded graph
            threshold = 0.3
            adj_matrix = (np.abs(corr_matrix) > threshold).astype(float)
            np.fill_diagonal(adj_matrix, 0)

            # Create NetworkX graph
            G = nx.from_numpy_array(adj_matrix)

            # Compute basic metrics
            metrics = {
                "density": nx.density(G),
                "avg_clustering": nx.average_clustering(G),
                "global_efficiency": nx.global_efficiency(G),
                "n_edges": G.number_of_edges(),
                "avg_path_length": 0.0,  # Will compute if connected
            }

            # Path length for connected graphs
            if nx.is_connected(G) and G.number_of_nodes() > 1:
                metrics["avg_path_length"] = nx.average_shortest_path_length(G)

            return metrics

        except Exception as e:
            logger.debug(f"Error computing null metrics: {e}")
            return {"density": 0.0, "avg_clustering": 0.0, "global_efficiency": 0.0}


class ConfigurationNullModel(BaseNullModelGenerator):
    """Generate null models preserving degree sequence."""

    def __init__(self, n_samples: int = 1000, max_tries: int = 100, random_seed: int = 42):
        super().__init__(n_samples, random_seed)
        self.max_tries = max_tries

    def generate_null_samples(self, data: np.ndarray, graph: nx.Graph, **kwargs) -> list[dict[str, float]]:
        """Generate null samples by rewiring edges while preserving degree sequence."""
        null_samples = []

        for _ in range(self.n_samples):
            try:
                # Create degree-preserving random graph
                null_graph = self._create_configuration_graph(graph)

                # Compute metrics
                metrics = self._compute_graph_metrics(null_graph)
                null_samples.append(metrics)

            except Exception as e:
                logger.debug(f"Failed to create configuration model: {e}")
                # Fallback to Erdős-Rényi with same density
                null_graph = nx.erdos_renyi_graph(
                    graph.number_of_nodes(), nx.density(graph), seed=np.random.randint(0, 10000)
                )
                metrics = self._compute_graph_metrics(null_graph)
                null_samples.append(metrics)

        return null_samples

    def _create_configuration_graph(self, graph: nx.Graph) -> nx.Graph:
        """Create configuration model preserving degree sequence."""
        if graph.number_of_edges() == 0:
            return nx.Graph(graph.nodes())

        # Get degree sequence
        degree_sequence = [d for n, d in graph.degree()]

        # Create configuration model
        try:
            null_graph = nx.configuration_model(degree_sequence, seed=self.random_seed)
            # Remove self-loops and multi-edges
            null_graph = nx.Graph(null_graph)
            null_graph.remove_edges_from(nx.selfloop_edges(null_graph))
            return null_graph
        except:
            # Fallback to degree-preserving rewiring
            null_graph = graph.copy()
            try:
                null_graph = nx.double_edge_swap(null_graph, nswap=max(1, graph.number_of_edges()))
            except:
                pass
            return null_graph

    def _compute_graph_metrics(self, graph: nx.Graph) -> dict[str, float]:
        """Compute comprehensive graph metrics."""
        metrics = {
            "density": nx.density(graph),
            "n_edges": graph.number_of_edges(),
            "avg_clustering": nx.average_clustering(graph),
            "global_efficiency": nx.global_efficiency(graph),
            "avg_path_length": 0.0,
            "modularity": 0.0,
            "assortativity": 0.0,
        }

        # Path length for connected graphs
        if nx.is_connected(graph) and graph.number_of_nodes() > 1:
            try:
                metrics["avg_path_length"] = nx.average_shortest_path_length(graph)
            except:
                metrics["avg_path_length"] = 0.0

        # Modularity (using simple greedy community detection)
        try:
            communities = nx.community.greedy_modularity_communities(graph)
            if len(communities) > 1:
                metrics["modularity"] = nx.community.modularity(graph, communities)
        except:
            metrics["modularity"] = 0.0

        # Degree assortativity
        try:
            if graph.number_of_edges() > 0:
                metrics["assortativity"] = nx.degree_assortativity_coefficient(graph)
        except:
            metrics["assortativity"] = 0.0

        return metrics


class SignedNullModel(BaseNullModelGenerator):
    """Generate null models for signed networks preserving sign distribution."""

    def generate_null_samples(self, data: np.ndarray, graph: nx.Graph, **kwargs) -> list[dict[str, float]]:
        """Generate null samples for signed networks preserving edge sign distribution.

        Args:
            data: Activation data
            graph: Original signed graph
            **kwargs: Additional parameters

        Returns:
            List of metrics from sign-randomized graphs

        """
        null_samples = []

        # Analyze original sign distribution
        sign_stats = self._analyze_sign_distribution(graph)

        for _ in range(self.n_samples):
            try:
                # Create null graph with randomized signs
                null_graph = self._create_sign_randomized_graph(graph, sign_stats)

                # Compute signed network metrics
                metrics = self._compute_signed_metrics(null_graph)
                null_samples.append(metrics)

            except Exception as e:
                logger.debug(f"Failed to create signed null model: {e}")
                # Fallback to basic metrics
                metrics = {"structural_balance": 0.5, "edge_frustration": 0.5}
                null_samples.append(metrics)

        return null_samples

    def _analyze_sign_distribution(self, graph: nx.Graph) -> dict[str, Any]:
        """Analyze the distribution of edge signs."""
        pos_edges = 0
        neg_edges = 0
        total_weight = 0.0

        for u, v, data in graph.edges(data=True):
            weight = data.get("weight", 1.0)
            total_weight += abs(weight)

            if weight > 0:
                pos_edges += 1
            elif weight < 0:
                neg_edges += 1

        total_edges = pos_edges + neg_edges

        return {
            "positive_ratio": pos_edges / total_edges if total_edges > 0 else 0.5,
            "negative_ratio": neg_edges / total_edges if total_edges > 0 else 0.5,
            "total_edges": total_edges,
            "avg_weight_magnitude": total_weight / total_edges if total_edges > 0 else 1.0,
        }

    def _create_sign_randomized_graph(self, graph: nx.Graph, sign_stats: dict[str, Any]) -> nx.Graph:
        """Create graph with randomized edge signs but preserved structure."""
        null_graph = nx.Graph()
        null_graph.add_nodes_from(graph.nodes())

        # Copy edges but randomize signs
        edges = list(graph.edges(data=True))
        n_edges = len(edges)

        if n_edges == 0:
            return null_graph

        # Generate random signs preserving distribution
        n_positive = int(sign_stats["positive_ratio"] * n_edges)
        signs = [1.0] * n_positive + [-1.0] * (n_edges - n_positive)
        np.random.shuffle(signs)

        # Assign randomized signs to edges
        for i, (u, v, data) in enumerate(edges):
            weight_magnitude = abs(data.get("weight", 1.0))
            new_weight = signs[i] * weight_magnitude

            null_graph.add_edge(u, v, weight=new_weight, sign=signs[i])

        null_graph.graph["signed"] = True
        return null_graph

    def _compute_signed_metrics(self, graph: nx.Graph) -> dict[str, float]:
        """Compute signed network specific metrics."""
        metrics = {
            "structural_balance": self._compute_structural_balance(graph),
            "edge_frustration": self._compute_edge_frustration(graph),
            "positive_edges": 0,
            "negative_edges": 0,
            "edge_balance_ratio": 0.5,
        }

        # Count edge signs
        pos_edges = sum(1 for _, _, d in graph.edges(data=True) if d.get("weight", 1.0) > 0)
        neg_edges = sum(1 for _, _, d in graph.edges(data=True) if d.get("weight", 1.0) < 0)
        total_edges = pos_edges + neg_edges

        metrics.update(
            {
                "positive_edges": pos_edges,
                "negative_edges": neg_edges,
                "edge_balance_ratio": pos_edges / total_edges if total_edges > 0 else 0.5,
            }
        )

        return metrics

    def _compute_structural_balance(self, graph: nx.Graph) -> float:
        """Compute structural balance for signed network."""
        try:
            balanced_triangles = 0
            total_triangles = 0

            for triangle in nx.enumerate_all_cliques(graph):
                if len(triangle) == 3:
                    total_triangles += 1
                    i, j, k = triangle

                    # Get edge signs
                    sign_ij = np.sign(graph[i][j].get("weight", 1.0)) if graph.has_edge(i, j) else 0
                    sign_jk = np.sign(graph[j][k].get("weight", 1.0)) if graph.has_edge(j, k) else 0
                    sign_ik = np.sign(graph[i][k].get("weight", 1.0)) if graph.has_edge(i, k) else 0

                    # Triangle is balanced if even number of negative edges
                    negative_edges = sum(1 for sign in [sign_ij, sign_jk, sign_ik] if sign < 0)
                    if negative_edges % 2 == 0:
                        balanced_triangles += 1

            return balanced_triangles / total_triangles if total_triangles > 0 else 1.0

        except Exception as e:
            logger.debug(f"Error computing structural balance: {e}")
            return 1.0

    def _compute_edge_frustration(self, graph: nx.Graph) -> float:
        """Compute edge frustration metric."""
        try:
            frustrated_triangles = 0
            total_triangles = 0

            for triangle in nx.enumerate_all_cliques(graph):
                if len(triangle) == 3:
                    total_triangles += 1
                    i, j, k = triangle

                    # Get edge weights
                    w_ij = graph[i][j].get("weight", 1.0) if graph.has_edge(i, j) else 0
                    w_jk = graph[j][k].get("weight", 1.0) if graph.has_edge(j, k) else 0
                    w_ik = graph[i][k].get("weight", 1.0) if graph.has_edge(i, k) else 0

                    # Triangle is frustrated if product of signs is negative
                    sign_product = np.sign(w_ij) * np.sign(w_jk) * np.sign(w_ik)
                    if sign_product < 0:
                        frustrated_triangles += 1

            return frustrated_triangles / total_triangles if total_triangles > 0 else 0.0

        except Exception as e:
            logger.debug(f"Error computing edge frustration: {e}")
            return 0.0


class BootstrapEstimator:
    """Bootstrap estimation for confidence intervals and stability assessment."""

    def __init__(self, n_bootstrap: int = 1000, confidence_level: float = 0.95, random_seed: int = 42):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def bootstrap_metric(
        self, data: np.ndarray, metric_function: callable, **kwargs
    ) -> tuple[float, tuple[float, float], list[float]]:
        """Compute bootstrap estimate and confidence interval for a metric."""
        bootstrap_samples = []
        n_samples = data.shape[0] if hasattr(data, "shape") else len(data)

        for _ in range(self.n_bootstrap):
            try:
                # Bootstrap resample
                if hasattr(data, "shape") and len(data.shape) > 1:
                    # For 2D arrays (contexts x neurons)
                    indices = np.random.choice(n_samples, size=n_samples, replace=True)
                    bootstrap_data = data[indices]
                else:
                    # For 1D arrays
                    bootstrap_data = resample(data, n_samples=n_samples, random_state=None)

                # Compute metric on bootstrap sample
                metric_value = metric_function(bootstrap_data, **kwargs)

                # Handle different return types
                if isinstance(metric_value, (int, float)):
                    bootstrap_samples.append(float(metric_value))
                elif isinstance(metric_value, dict) and "value" in metric_value:
                    bootstrap_samples.append(float(metric_value["value"]))
                else:
                    # Skip invalid samples
                    continue

            except Exception as e:
                logger.debug(f"Bootstrap sample failed: {e}")
                continue

        if not bootstrap_samples:
            logger.warning("No valid bootstrap samples generated")
            return 0.0, (0.0, 0.0), []

        # Compute statistics
        mean_estimate = np.mean(bootstrap_samples)

        # Confidence interval
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_samples, lower_percentile)
        ci_upper = np.percentile(bootstrap_samples, upper_percentile)

        return mean_estimate, (ci_lower, ci_upper), bootstrap_samples


class SignificanceTester:
    """Perform various statistical significance tests."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def permutation_test(
        self, observed_value: float, null_distribution: list[float], alternative: str = "two-sided"
    ) -> StatisticalTest:
        """Perform permutation test against null distribution."""
        if not null_distribution:
            return self._create_invalid_test("permutation", "Empty null distribution")

        null_array = np.array(null_distribution)
        n_null = len(null_array)

        # Compute p-value based on alternative hypothesis
        if alternative == "two-sided":
            null_mean = np.mean(null_array)
            p_value = np.mean(np.abs(null_array - null_mean) >= abs(observed_value - null_mean))
        elif alternative == "greater":
            p_value = np.mean(null_array >= observed_value)
        elif alternative == "less":
            p_value = np.mean(null_array <= observed_value)
        else:
            raise ValueError(f"Unknown alternative: {alternative}")

        # Effect size (Cohen's d)
        null_std = np.std(null_array)
        effect_size = (observed_value - np.mean(null_array)) / null_std if null_std > 0 else 0.0

        # Confidence interval for effect size (approximate)
        se_effect = np.sqrt((1 + effect_size**2 / 2) / n_null)
        ci_lower = effect_size - 1.96 * se_effect
        ci_upper = effect_size + 1.96 * se_effect

        return StatisticalTest(
            test_name="permutation_test",
            p_value=float(p_value),
            effect_size=float(effect_size),
            confidence_interval=(ci_lower, ci_upper),
            is_significant=p_value < self.alpha,
            test_statistic=float(observed_value),
            description=f"Permutation test ({alternative}) with {n_null} null samples",
        )

    def bootstrap_test(
        self, bootstrap_samples: list[float], null_value: float = 0.0, alternative: str = "two-sided"
    ) -> StatisticalTest:
        """Perform bootstrap-based significance test."""
        if not bootstrap_samples:
            return self._create_invalid_test("bootstrap", "Empty bootstrap samples")

        samples = np.array(bootstrap_samples)
        observed_mean = np.mean(samples)

        # Center bootstrap samples around null value
        centered_samples = samples - observed_mean + null_value

        # Compute p-value
        if alternative == "two-sided":
            p_value = 2 * min(np.mean(centered_samples <= null_value), np.mean(centered_samples >= null_value))
        elif alternative == "greater":
            p_value = np.mean(centered_samples <= null_value)
        elif alternative == "less":
            p_value = np.mean(centered_samples >= null_value)
        else:
            raise ValueError(f"Unknown alternative: {alternative}")

        # Effect size
        bootstrap_std = np.std(samples)
        effect_size = (observed_mean - null_value) / bootstrap_std if bootstrap_std > 0 else 0.0

        # Confidence interval
        ci_lower = np.percentile(samples, 2.5)
        ci_upper = np.percentile(samples, 97.5)

        return StatisticalTest(
            test_name="bootstrap_test",
            p_value=float(p_value),
            effect_size=float(effect_size),
            confidence_interval=(ci_lower, ci_upper),
            is_significant=p_value < self.alpha,
            test_statistic=float(observed_mean),
            description=f"Bootstrap test ({alternative}) with {len(samples)} samples",
        )

    def _create_invalid_test(self, test_name: str, reason: str) -> StatisticalTest:
        """Create invalid test result."""
        return StatisticalTest(
            test_name=test_name,
            p_value=1.0,
            effect_size=0.0,
            confidence_interval=(0.0, 0.0),
            is_significant=False,
            test_statistic=0.0,
            description=f"Invalid test: {reason}",
        )


class StatisticalValidator:
    """Comprehensive statistical validation framework for network analysis results.

    Orchestrates null model generation, bootstrap estimation, and significance testing
    to provide rigorous statistical validation of network metrics.
    """

    def __init__(
        self,
        null_model_type: str = "permutation",
        n_null_samples: int = 1000,
        n_bootstrap: int = 1000,
        significance_level: float = 0.05,
        multiple_testing_correction: str = "bonferroni",
        random_seed: int = 42,
    ):
        """Initialize StatisticalValidator.

        Args:
            null_model_type: Type of null model ("permutation", "configuration", "signed")
            n_null_samples: Number of null model samples
            n_bootstrap: Number of bootstrap samples
            significance_level: Statistical significance threshold
            multiple_testing_correction: Method for multiple testing correction
            random_seed: Random seed for reproducibility

        """
        self.null_model_type = null_model_type
        self.n_null_samples = n_null_samples
        self.n_bootstrap = n_bootstrap
        self.significance_level = significance_level
        self.multiple_testing_correction = multiple_testing_correction
        self.random_seed = random_seed

        # Initialize components
        self._initialize_null_model_generator()
        self.bootstrap_estimator = BootstrapEstimator(
            n_bootstrap=n_bootstrap, confidence_level=1 - significance_level, random_seed=random_seed
        )
        self.significance_tester = SignificanceTester(alpha=significance_level)

        logger.info(
            f"StatisticalValidator initialized: {null_model_type} null model, "
            f"{n_null_samples} null samples, {n_bootstrap} bootstrap samples"
        )

    def _initialize_null_model_generator(self):
        """Initialize appropriate null model generator."""
        if self.null_model_type == "permutation":
            self.null_generator = PermutationNullModel(n_samples=self.n_null_samples, random_seed=self.random_seed)
        elif self.null_model_type == "configuration":
            self.null_generator = ConfigurationNullModel(n_samples=self.n_null_samples, random_seed=self.random_seed)
        elif self.null_model_type == "signed":
            self.null_generator = SignedNullModel(n_samples=self.n_null_samples, random_seed=self.random_seed)
        else:
            raise ValueError(f"Unknown null model type: {self.null_model_type}")

    def validate_network_metrics(
        self, observed_metrics: dict[str, float], data: np.ndarray, graph: nx.Graph | None = None
    ) -> dict[str, ValidationResult]:
        """Validate network metrics against null models.

        Args:
            observed_metrics: Dictionary of observed metric values
            data: Original activation data
            graph: NetworkX graph (required for some null models)

        Returns:
            Dictionary mapping metric names to ValidationResult objects

        """
        logger.debug(f"Validating {len(observed_metrics)} metrics against {self.null_model_type} null model")

        # Generate null distribution
        null_samples = self._generate_null_distribution(data, graph)

        if not null_samples:
            logger.warning("Failed to generate null samples")
            return self._create_empty_validation_results(observed_metrics)

        # Validate each metric
        validation_results = {}

        for metric_name, observed_value in observed_metrics.items():
            if not isinstance(observed_value, (int, float)):
                logger.debug(f"Skipping non-numeric metric: {metric_name}")
                continue

            try:
                validation_result = self._validate_single_metric(metric_name, observed_value, null_samples)
                validation_results[metric_name] = validation_result

            except Exception as e:
                logger.warning(f"Failed to validate {metric_name}: {e}")
                validation_results[metric_name] = self._create_failed_validation(metric_name, observed_value, str(e))

        # Apply multiple testing correction
        if len(validation_results) > 1:
            validation_results = self._apply_multiple_testing_correction(validation_results)

        logger.debug(
            f"Validation complete: {sum(1 for r in validation_results.values() if r.is_significant)} "
            f"significant out of {len(validation_results)} metrics"
        )

        return validation_results

    def validate_hypothesis_results(
        self, hypothesis_results: dict[str, Any], data: np.ndarray, graph: nx.Graph | None = None
    ) -> dict[str, ValidationResult]:
        """Validate hypothesis test results with statistical significance testing.

        Args:
            hypothesis_results: Results from HypothesisTestSuite
            data: Original activation data
            graph: NetworkX graph

        Returns:
            Dictionary of validation results for hypothesis evidence metrics

        """
        # Extract evidence metrics from hypothesis results
        evidence_metrics = {}

        for hyp_name, result in hypothesis_results.items():
            if hasattr(result, "evidence_details") and isinstance(result.evidence_details, dict):
                for metric_name, value in result.evidence_details.items():
                    if isinstance(value, (int, float)):
                        evidence_metrics[f"{hyp_name}_{metric_name}"] = value

        # Validate evidence metrics
        return self.validate_network_metrics(evidence_metrics, data, graph)

    def compare_networks_statistically(
        self,
        network1_metrics: dict[str, float],
        network2_metrics: dict[str, float],
        data1: np.ndarray,
        data2: np.ndarray,
        graph1: nx.Graph | None = None,
        graph2: nx.Graph | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Statistical comparison between two networks."""
        comparison_results = {}

        # Find common metrics
        common_metrics = set(network1_metrics.keys()) & set(network2_metrics.keys())

        for metric_name in common_metrics:
            if not (
                isinstance(network1_metrics[metric_name], (int, float))
                and isinstance(network2_metrics[metric_name], (int, float))
            ):
                continue

            try:
                # Generate bootstrap distributions for both networks
                bootstrap1 = self._generate_bootstrap_distribution(metric_name, data1, graph1)
                bootstrap2 = self._generate_bootstrap_distribution(metric_name, data2, graph2)

                # Compare distributions
                comparison = self._compare_distributions(
                    bootstrap1, bootstrap2, network1_metrics[metric_name], network2_metrics[metric_name]
                )

                comparison_results[metric_name] = comparison

            except Exception as e:
                logger.warning(f"Failed to compare {metric_name}: {e}")
                comparison_results[metric_name] = {"error": str(e)}

        return comparison_results

    def _generate_null_distribution(self, data: np.ndarray, graph: nx.Graph | None = None) -> list[dict[str, float]]:
        """Generate null distribution using configured null model."""
        try:
            if self.null_model_type in ["configuration", "signed"] and graph is None:
                logger.warning(f"{self.null_model_type} null model requires graph, falling back to permutation")
                fallback_generator = PermutationNullModel(self.n_null_samples, self.random_seed)
                return fallback_generator.generate_null_samples(data)

            if self.null_model_type == "permutation":
                return self.null_generator.generate_null_samples(data)
            if self.null_model_type == "configuration" or self.null_model_type == "signed":
                return self.null_generator.generate_null_samples(data, graph=graph)
            logger.warning(f"Unknown null model type: {self.null_model_type}")
            return []

        except Exception as e:
            logger.warning(f"Failed to generate null distribution: {e}")
            return []

    def _validate_single_metric(
        self, metric_name: str, observed_value: float, null_samples: list[dict[str, float]]
    ) -> ValidationResult:
        """Validate a single metric against null distribution."""
        # Extract null values for this metric
        null_values = []
        for sample in null_samples:
            if metric_name in sample and isinstance(sample[metric_name], (int, float)):
                null_values.append(float(sample[metric_name]))

        if not null_values:
            return self._create_failed_validation(metric_name, observed_value, "No valid null values found")

        # Compute null distribution statistics
        null_stats = {
            "mean": float(np.mean(null_values)),
            "std": float(np.std(null_values)),
            "min": float(np.min(null_values)),
            "max": float(np.max(null_values)),
            "median": float(np.median(null_values)),
            "n_samples": len(null_values),
        }

        # Perform statistical tests
        tests = []

        # Permutation test (two-sided)
        perm_test = self.significance_tester.permutation_test(observed_value, null_values, alternative="two-sided")
        tests.append(perm_test)

        # One-sided tests for directional hypotheses
        if observed_value > null_stats["mean"]:
            perm_test_greater = self.significance_tester.permutation_test(
                observed_value, null_values, alternative="greater"
            )
            tests.append(perm_test_greater)
        else:
            perm_test_less = self.significance_tester.permutation_test(observed_value, null_values, alternative="less")
            tests.append(perm_test_less)

        # Determine overall significance
        is_significant = any(test.is_significant for test in tests)

        # Interpret effect size
        effect_size = perm_test.effect_size
        effect_interpretation = self._interpret_effect_size(effect_size)

        # Generate summary
        summary = self._generate_metric_summary(metric_name, observed_value, null_stats, tests[0], is_significant)

        return ValidationResult(
            metric_name=metric_name,
            observed_value=observed_value,
            null_distribution_stats=null_stats,
            statistical_tests=tests,
            is_significant=is_significant,
            effect_size_interpretation=effect_interpretation,
            summary=summary,
        )

    def _generate_bootstrap_distribution(
        self, metric_name: str, data: np.ndarray, graph: nx.Graph | None = None
    ) -> list[float]:
        """Generate bootstrap distribution for a specific metric."""

        # Define metric computation function
        def compute_metric(bootstrap_data):
            try:
                if metric_name in ["density", "avg_clustering", "global_efficiency"]:
                    # Graph-based metrics
                    if bootstrap_data.shape[0] < 2:
                        return 0.0

                    corr_matrix = np.corrcoef(bootstrap_data.T)
                    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

                    # Create graph
                    threshold = 0.3
                    adj_matrix = (np.abs(corr_matrix) > threshold).astype(float)
                    np.fill_diagonal(adj_matrix, 0)
                    G = nx.from_numpy_array(adj_matrix)

                    if metric_name == "density":
                        return nx.density(G)
                    if metric_name == "avg_clustering":
                        return nx.average_clustering(G)
                    if metric_name == "global_efficiency":
                        return nx.global_efficiency(G)

                elif metric_name == "modularity":
                    # Community-based metric
                    if bootstrap_data.shape[0] < 3:
                        return 0.0

                    corr_matrix = np.corrcoef(bootstrap_data.T)
                    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

                    threshold = 0.3
                    adj_matrix = (np.abs(corr_matrix) > threshold).astype(float)
                    np.fill_diagonal(adj_matrix, 0)
                    G = nx.from_numpy_array(adj_matrix)

                    try:
                        communities = nx.community.greedy_modularity_communities(G)
                        if len(communities) > 1:
                            return nx.community.modularity(G, communities)
                        return 0.0
                    except:
                        return 0.0

                else:
                    # Default: return simple statistic
                    return float(np.mean(bootstrap_data))

            except Exception as e:
                logger.debug(f"Bootstrap metric computation failed: {e}")
                return 0.0

        # Generate bootstrap samples
        try:
            mean_est, ci, bootstrap_samples = self.bootstrap_estimator.bootstrap_metric(data, compute_metric)
            return bootstrap_samples
        except Exception as e:
            logger.warning(f"Bootstrap generation failed for {metric_name}: {e}")
            return []

    def _compare_distributions(
        self, dist1: list[float], dist2: list[float], obs1: float, obs2: float
    ) -> dict[str, Any]:
        """Compare two bootstrap distributions statistically."""
        if not dist1 or not dist2:
            return {"error": "Empty distributions"}

        # Convert to arrays
        arr1 = np.array(dist1)
        arr2 = np.array(dist2)

        # Compute difference in means
        diff_obs = obs2 - obs1
        diff_bootstrap = arr2 - arr1

        # Permutation test for difference
        pooled = np.concatenate([diff_bootstrap, -diff_bootstrap])
        p_value = np.mean(np.abs(pooled) >= abs(diff_obs))

        # Effect size for difference
        pooled_std = np.std(pooled)
        effect_size = diff_obs / pooled_std if pooled_std > 0 else 0.0

        # Confidence interval for difference
        ci_lower = np.percentile(diff_bootstrap, 2.5)
        ci_upper = np.percentile(diff_bootstrap, 97.5)

        return {
            "difference": float(diff_obs),
            "p_value": float(p_value),
            "effect_size": float(effect_size),
            "confidence_interval": (float(ci_lower), float(ci_upper)),
            "is_significant": p_value < self.significance_level,
            "interpretation": "Network 2 significantly different"
            if p_value < self.significance_level
            else "No significant difference",
        }

    def _apply_multiple_testing_correction(
        self, validation_results: dict[str, ValidationResult]
    ) -> dict[str, ValidationResult]:
        """Apply multiple testing correction to p-values."""
        if self.multiple_testing_correction == "none":
            return validation_results

        # Extract p-values
        p_values = []
        result_keys = []

        for key, result in validation_results.items():
            if result.statistical_tests:
                p_values.append(result.statistical_tests[0].p_value)
                result_keys.append(key)

        if not p_values:
            return validation_results

        # Apply correction
        if self.multiple_testing_correction == "bonferroni":
            corrected_p_values = [min(1.0, p * len(p_values)) for p in p_values]
        elif self.multiple_testing_correction == "holm":
            corrected_p_values = self._holm_correction(p_values)
        elif self.multiple_testing_correction == "fdr":
            corrected_p_values = self._benjamini_hochberg_correction(p_values)
        else:
            logger.warning(f"Unknown correction method: {self.multiple_testing_correction}")
            return validation_results

        # Update results with corrected p-values
        corrected_results = validation_results.copy()

        for i, key in enumerate(result_keys):
            original_result = corrected_results[key]

            # Create corrected test
            corrected_test = StatisticalTest(
                test_name=original_result.statistical_tests[0].test_name + "_corrected",
                p_value=corrected_p_values[i],
                effect_size=original_result.statistical_tests[0].effect_size,
                confidence_interval=original_result.statistical_tests[0].confidence_interval,
                is_significant=corrected_p_values[i] < self.significance_level,
                test_statistic=original_result.statistical_tests[0].test_statistic,
                description=f"{original_result.statistical_tests[0].description} ({self.multiple_testing_correction} corrected)",
            )

            # Update result
            updated_tests = original_result.statistical_tests + [corrected_test]
            updated_significance = corrected_test.is_significant

            corrected_results[key] = ValidationResult(
                metric_name=original_result.metric_name,
                observed_value=original_result.observed_value,
                null_distribution_stats=original_result.null_distribution_stats,
                statistical_tests=updated_tests,
                is_significant=updated_significance,
                effect_size_interpretation=original_result.effect_size_interpretation,
                summary=original_result.summary + f" (Corrected p={corrected_p_values[i]:.4f})",
            )

        return corrected_results

    def _holm_correction(self, p_values: list[float]) -> list[float]:
        """Apply Holm-Bonferroni correction."""
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        corrected = [0.0] * n

        for i, idx in enumerate(sorted_indices):
            corrected[idx] = min(1.0, p_values[idx] * (n - i))

        # Ensure monotonicity
        for i in range(1, n):
            idx = sorted_indices[i]
            prev_idx = sorted_indices[i - 1]
            corrected[idx] = max(corrected[idx], corrected[prev_idx])

        return corrected

    def _benjamini_hochberg_correction(self, p_values: list[float]) -> list[float]:
        """Apply Benjamini-Hochberg FDR correction."""
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        corrected = [0.0] * n

        for i in range(n - 1, -1, -1):
            idx = sorted_indices[i]
            corrected[idx] = min(1.0, p_values[idx] * n / (i + 1))

            # Ensure monotonicity
            if i < n - 1:
                next_idx = sorted_indices[i + 1]
                corrected[idx] = min(corrected[idx], corrected[next_idx])

        return corrected

    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_effect = abs(effect_size)

        if abs_effect < 0.2:
            return "negligible"
        if abs_effect < 0.5:
            return "small"
        if abs_effect < 0.8:
            return "medium"
        return "large"

    def _generate_metric_summary(
        self,
        metric_name: str,
        observed_value: float,
        null_stats: dict[str, float],
        test: StatisticalTest,
        is_significant: bool,
    ) -> str:
        """Generate human-readable summary for metric validation."""
        null_mean = null_stats["mean"]
        direction = "higher" if observed_value > null_mean else "lower"

        if is_significant:
            summary = (
                f"{metric_name}: {observed_value:.4f} is significantly {direction} "
                f"than null expectation {null_mean:.4f} "
                f"(p={test.p_value:.4f}, effect size: {test.effect_size:.2f})"
            )
        else:
            summary = (
                f"{metric_name}: {observed_value:.4f} is not significantly different "
                f"from null expectation {null_mean:.4f} "
                f"(p={test.p_value:.4f})"
            )

        return summary

    def _create_empty_validation_results(self, observed_metrics: dict[str, float]) -> dict[str, ValidationResult]:
        """Create empty validation results for failed null model generation."""
        results = {}
        for metric_name, observed_value in observed_metrics.items():
            results[metric_name] = self._create_failed_validation(
                metric_name, observed_value, "Failed to generate null distribution"
            )
        return results

    def _create_failed_validation(self, metric_name: str, observed_value: float, reason: str) -> ValidationResult:
        """Create failed validation result."""
        failed_test = StatisticalTest(
            test_name="failed",
            p_value=1.0,
            effect_size=0.0,
            confidence_interval=(0.0, 0.0),
            is_significant=False,
            test_statistic=observed_value,
            description=f"Test failed: {reason}",
        )

        return ValidationResult(
            metric_name=metric_name,
            observed_value=observed_value,
            null_distribution_stats={"mean": 0.0, "std": 0.0, "n_samples": 0},
            statistical_tests=[failed_test],
            is_significant=False,
            effect_size_interpretation="unknown",
            summary=f"{metric_name} validation failed: {reason}",
        )

    def generate_validation_report(self, validation_results: dict[str, ValidationResult]) -> str:
        """Generate comprehensive validation report."""
        lines = ["Statistical Validation Report", "=" * 50, ""]

        # Summary statistics
        total_metrics = len(validation_results)
        significant_metrics = sum(1 for r in validation_results.values() if r.is_significant)

        lines.extend(
            [
                f"Total metrics tested: {total_metrics}",
                f"Statistically significant: {significant_metrics}",
                f"Significance rate: {significant_metrics / total_metrics * 100:.1f}%",
                f"Null model: {self.null_model_type}",
                f"Multiple testing correction: {self.multiple_testing_correction}",
                "",
            ]
        )

        # Detailed results
        lines.append("Detailed Results:")
        lines.append("-" * 30)

        for metric_name, result in sorted(validation_results.items()):
            status = "SIGNIFICANT" if result.is_significant else "NOT SIGNIFICANT"
            effect = result.effect_size_interpretation

            lines.extend(
                [
                    f"{metric_name}: {status}",
                    f"  Observed: {result.observed_value:.4f}",
                    f"  Null mean: {result.null_distribution_stats['mean']:.4f}",
                    f"  P-value: {result.statistical_tests[0].p_value:.4f}",
                    f"  Effect size: {result.statistical_tests[0].effect_size:.3f} ({effect})",
                    "",
                ]
            )

        return "\n".join(lines)

    def export_validation_data(self, validation_results: dict[str, ValidationResult]) -> dict[str, Any]:
        """Export validation results as structured data."""
        export_data = {
            "validation_summary": {
                "total_metrics": len(validation_results),
                "significant_metrics": sum(1 for r in validation_results.values() if r.is_significant),
                "null_model_type": self.null_model_type,
                "n_null_samples": self.n_null_samples,
                "significance_level": self.significance_level,
                "multiple_testing_correction": self.multiple_testing_correction,
            },
            "metric_results": {},
        }

        for metric_name, result in validation_results.items():
            export_data["metric_results"][metric_name] = {
                "observed_value": result.observed_value,
                "is_significant": result.is_significant,
                "p_value": result.statistical_tests[0].p_value if result.statistical_tests else 1.0,
                "effect_size": result.statistical_tests[0].effect_size if result.statistical_tests else 0.0,
                "effect_size_interpretation": result.effect_size_interpretation,
                "null_distribution_stats": result.null_distribution_stats,
                "summary": result.summary,
            }

        return export_data
