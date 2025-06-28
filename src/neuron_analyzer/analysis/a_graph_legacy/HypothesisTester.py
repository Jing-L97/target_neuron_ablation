import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


@dataclass
class HypothesisResult:
    """Container for hypothesis test results."""

    hypothesis_name: str
    supported: bool
    evidence_strength: float  # 0.0 to 1.0
    evidence_details: dict[str, Any]
    summary: str


class BaseHypothesisTester(ABC):
    """Abstract base class for hypothesis testing."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def test_hypothesis(self, analysis_results: dict[str, Any], graph: nx.Graph, **kwargs) -> HypothesisResult:
        """Test the hypothesis and return results."""

    def _compute_evidence_strength(self, metrics: dict[str, float], thresholds: dict[str, float]) -> float:
        """Compute overall evidence strength from multiple metrics."""
        if not metrics or not thresholds:
            return 0.0  # Return float instead of metrics dict

        # Compute evidence strength as average of normalized metric scores
        evidence_scores = []

        for metric_name, metric_value in metrics.items():
            if metric_name in thresholds:
                threshold = thresholds[metric_name]
                if threshold > 0:
                    # Normalize score: metric/threshold, capped at 1.0
                    normalized_score = min(metric_value / threshold, 1.0)
                else:
                    # For zero threshold, use binary scoring
                    normalized_score = 1.0 if metric_value > 0 else 0.0
                evidence_scores.append(normalized_score)

        # Return average evidence strength, or 0.0 if no valid scores
        return float(np.mean(evidence_scores)) if evidence_scores else 0.0

    def _generate_optimization_summary(self, evidence: dict[str, float], supported: bool) -> str:
        """Generate human-readable summary of optimization evidence."""
        if supported:
            summary = "Optimized information flow detected. "
            global_eff = evidence.get("global_efficiency", 0)
            cost_eff = evidence.get("cost_efficiency", 0)
            summary += f"Global efficiency: {global_eff:.3f}, "
            summary += f"Cost-efficiency: {cost_eff:.3f}"

            if "small_world_sigma" in evidence:
                summary += f", Small-world sigma: {evidence['small_world_sigma']:.2f}"

            if "balance_optimization" in evidence:
                summary += f", Balance optimization: {evidence['balance_optimization']:.3f}"
        else:
            summary = "Limited optimization detected. "
            summary += "Network efficiency below optimal thresholds."

        return summary


class HypothesisTestSuite:
    """Comprehensive hypothesis testing suite for neural coordination patterns.

    Orchestrates multiple hypothesis tests and provides unified reporting.
    """

    def __init__(
        self,
        hierarchy_config: dict[str, Any] | None = None,
        modularity_config: dict[str, Any] | None = None,
        adaptivity_config: dict[str, Any] | None = None,
        optimization_config: dict[str, Any] | None = None,
    ):
        """Initialize hypothesis test suite with configurable testers."""
        # Initialize hypothesis testers
        self.hierarchy_tester = HierarchyTester(**(hierarchy_config or {}))
        self.modularity_tester = ModularityTester(**(modularity_config or {}))
        self.adaptivity_tester = AdaptivityTester(**(adaptivity_config or {}))
        self.optimization_tester = OptimizationTester(**(optimization_config or {}))

        # Store testers for iteration
        self.testers = {
            "H1_hierarchy": self.hierarchy_tester,
            "H2_modularity": self.modularity_tester,
            "H3_adaptivity": self.adaptivity_tester,
            "H4_optimization": self.optimization_tester,
        }

    def run_all_tests(
        self,
        analysis_results: dict[str, Any],
        graph: nx.Graph,
        context_data: dict[str, Any] | None = None,
        random_baseline: dict[str, Any] | None = None,
    ) -> dict[str, HypothesisResult]:
        """Run all hypothesis tests and return results.

        Args:
            analysis_results: Network analysis results from NetworkAnalyzer
            graph: NetworkX graph being analyzed
            context_data: Optional context-specific analysis results
            random_baseline: Optional random network baseline for comparison

        Returns:
            Dictionary mapping hypothesis names to test results

        """
        results = {}

        logger.info("Running comprehensive hypothesis tests...")

        # H1: Hierarchical Organization
        logger.debug("Testing H1: Hierarchical Organization")
        try:
            results["H1_hierarchy"] = self.hierarchy_tester.test_hypothesis(analysis_results, graph)
        except Exception as e:
            logger.warning(f"H1 test failed: {e}")
            results["H1_hierarchy"] = self._create_failed_result("H1_hierarchy", str(e))

        # H2: Modular Architecture
        logger.debug("Testing H2: Modular Architecture")
        try:
            results["H2_modularity"] = self.modularity_tester.test_hypothesis(analysis_results, graph)
        except Exception as e:
            logger.warning(f"H2 test failed: {e}")
            results["H2_modularity"] = self._create_failed_result("H2_modularity", str(e))

        # H3: Context-Dependent Topology
        logger.debug("Testing H3: Context-Dependent Topology")
        try:
            kwargs = {}
            if context_data:
                kwargs.update(
                    {
                        "rare_context_results": context_data.get("rare_context"),
                        "common_context_results": context_data.get("common_context"),
                    }
                )

            results["H3_adaptivity"] = self.adaptivity_tester.test_hypothesis(analysis_results, graph, **kwargs)
        except Exception as e:
            logger.warning(f"H3 test failed: {e}")
            results["H3_adaptivity"] = self._create_failed_result("H3_adaptivity", str(e))

        # H4: Optimized Information Flow
        logger.debug("Testing H4: Optimized Information Flow")
        try:
            kwargs = {}
            if random_baseline:
                kwargs["random_baseline"] = random_baseline

            results["H4_optimization"] = self.optimization_tester.test_hypothesis(analysis_results, graph, **kwargs)
        except Exception as e:
            logger.warning(f"H4 test failed: {e}")
            results["H4_optimization"] = self._create_failed_result("H4_optimization", str(e))

        logger.info(
            f"Completed hypothesis tests: {sum(r.supported for r in results.values())}/{len(results)} supported"
        )

        return results

    def run_single_test(
        self, hypothesis_name: str, analysis_results: dict[str, Any], graph: nx.Graph, **kwargs
    ) -> HypothesisResult:
        """Run a single hypothesis test.

        Args:
            hypothesis_name: Name of hypothesis to test (H1_hierarchy, H2_modularity, etc.)
            analysis_results: Network analysis results
            graph: NetworkX graph
            **kwargs: Additional arguments for specific tests

        Returns:
            HypothesisResult for the specified test

        """
        if hypothesis_name not in self.testers:
            raise ValueError(f"Unknown hypothesis: {hypothesis_name}")

        tester = self.testers[hypothesis_name]

        try:
            return tester.test_hypothesis(analysis_results, graph, **kwargs)
        except Exception as e:
            logger.warning(f"Test {hypothesis_name} failed: {e}")
            return self._create_failed_result(hypothesis_name, str(e))

    def run_tests_for_multiple_networks(
        self,
        network_analyses: dict[str, dict[str, Any]],
        graphs: dict[str, nx.Graph],
        context_data: dict[str, dict[str, Any]] | None = None,
        random_baselines: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, dict[str, HypothesisResult]]:
        """Run hypothesis tests for multiple networks.

        Args:
            network_analyses: Network analysis results for each network
            graphs: NetworkX graphs for each network
            context_data: Optional context-specific data for each network
            random_baselines: Optional random baselines for each network

        Returns:
            Nested dictionary: network_name -> hypothesis_name -> result

        """
        all_results = {}

        for network_name in network_analyses:
            if network_name not in graphs:
                logger.warning(f"No graph provided for {network_name}, skipping")
                continue

            logger.debug(f"Testing hypotheses for network: {network_name}")

            # Prepare context data for this network
            network_context = context_data.get(network_name) if context_data else None
            network_baseline = random_baselines.get(network_name) if random_baselines else None

            # Run tests for this network
            network_results = self.run_all_tests(
                network_analyses[network_name], graphs[network_name], network_context, network_baseline
            )

            all_results[network_name] = network_results

        return all_results

    def generate_comparative_report(self, test_results: dict[str, dict[str, HypothesisResult]]) -> dict[str, Any]:
        """Generate comparative report across multiple networks.

        Args:
            test_results: Results from run_tests_for_multiple_networks

        Returns:
            Comparative analysis report

        """
        report = {
            "summary": {},
            "hypothesis_comparison": {},
            "network_profiles": {},
            "strongest_evidence": {},
            "recommendations": [],
        }

        if not test_results:
            return report

        # Overall summary
        network_names = list(test_results.keys())
        hypothesis_names = list(next(iter(test_results.values())).keys())

        report["summary"] = {
            "n_networks": len(network_names),
            "n_hypotheses": len(hypothesis_names),
            "networks": network_names,
            "hypotheses": hypothesis_names,
        }

        # Hypothesis-by-hypothesis comparison
        for hyp_name in hypothesis_names:
            hyp_results = {net: test_results[net][hyp_name] for net in network_names if hyp_name in test_results[net]}

            supported_networks = [net for net, result in hyp_results.items() if result.supported]
            avg_evidence = np.mean([result.evidence_strength for result in hyp_results.values()])

            report["hypothesis_comparison"][hyp_name] = {
                "supported_networks": supported_networks,
                "support_ratio": len(supported_networks) / len(network_names),
                "avg_evidence_strength": float(avg_evidence),
                "strongest_network": max(hyp_results.items(), key=lambda x: x[1].evidence_strength)[0]
                if hyp_results
                else None,
            }

        # Network profiles
        for net_name in network_names:
            net_results = test_results[net_name]
            supported_hypotheses = [hyp for hyp, result in net_results.items() if result.supported]
            avg_evidence = np.mean([result.evidence_strength for result in net_results.values()])

            report["network_profiles"][net_name] = {
                "supported_hypotheses": supported_hypotheses,
                "n_supported": len(supported_hypotheses),
                "avg_evidence_strength": float(avg_evidence),
                "network_type": self._classify_network_type(supported_hypotheses),
            }

        # Strongest evidence
        all_results = [
            (net, hyp, result) for net, hyp_results in test_results.items() for hyp, result in hyp_results.items()
        ]

        if all_results:
            strongest = max(all_results, key=lambda x: x[2].evidence_strength)
            report["strongest_evidence"] = {
                "network": strongest[0],
                "hypothesis": strongest[1],
                "evidence_strength": strongest[2].evidence_strength,
                "summary": strongest[2].summary,
            }

        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(report)

        return report

    def _create_failed_result(self, hypothesis_name: str, error_msg: str) -> HypothesisResult:
        """Create a failed result for error cases."""
        return HypothesisResult(
            hypothesis_name=hypothesis_name,
            supported=False,
            evidence_strength=0.0,
            evidence_details={"error": error_msg},
            summary=f"Test failed: {error_msg}",
        )

    def _classify_network_type(self, supported_hypotheses: list[str]) -> str:
        """Classify network type based on supported hypotheses."""
        if len(supported_hypotheses) == 0:
            return "unstructured"
        if len(supported_hypotheses) == 4:
            return "highly_organized"
        if "H1_hierarchy" in supported_hypotheses and "H2_modularity" in supported_hypotheses:
            return "hierarchical_modular"
        if "H1_hierarchy" in supported_hypotheses:
            return "hierarchical"
        if "H2_modularity" in supported_hypotheses:
            return "modular"
        if "H3_adaptivity" in supported_hypotheses:
            return "adaptive"
        if "H4_optimization" in supported_hypotheses:
            return "optimized"
        return "partially_structured"

    def _generate_recommendations(self, report: dict[str, Any]) -> list[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        # Check for consistent patterns
        hyp_comparison = report.get("hypothesis_comparison", {})

        for hyp_name, hyp_data in hyp_comparison.items():
            support_ratio = hyp_data.get("support_ratio", 0)

            if support_ratio > 0.7:
                recommendations.append(
                    f"{hyp_name} shows strong support across networks - investigate common mechanisms"
                )
            elif support_ratio < 0.3:
                recommendations.append(f"{hyp_name} shows limited support - consider alternative explanations")

        # Check for network-specific patterns
        network_profiles = report.get("network_profiles", {})
        highly_organized = [
            net for net, profile in network_profiles.items() if profile.get("network_type") == "highly_organized"
        ]

        if highly_organized:
            recommendations.append(
                f"Networks {highly_organized} show highly organized structure - focus detailed analysis here"
            )

        # Evidence strength recommendations
        strongest = report.get("strongest_evidence", {})
        if strongest:
            recommendations.append(
                f"Strongest evidence found in {strongest.get('network')} for {strongest.get('hypothesis')} - "
                f"use as reference case"
            )

        return recommendations

    def export_results_summary(self, test_results: dict[str, HypothesisResult]) -> str:
        """Export test results as human-readable summary."""
        summary_lines = ["Hypothesis Test Results Summary", "=" * 40, ""]

        for hyp_name, result in test_results.items():
            status = "SUPPORTED" if result.supported else "NOT SUPPORTED"
            strength = result.evidence_strength

            summary_lines.extend(
                [
                    f"{result.hypothesis_name}: {status}",
                    f"  Evidence Strength: {strength:.3f}",
                    f"  Summary: {result.summary}",
                    "",
                ]
            )

        # Overall summary
        supported_count = sum(1 for r in test_results.values() if r.supported)
        total_count = len(test_results)
        avg_strength = np.mean([r.evidence_strength for r in test_results.values()])

        summary_lines.extend(
            [
                "Overall Summary:",
                f"  Supported Hypotheses: {supported_count}/{total_count}",
                f"  Average Evidence Strength: {avg_strength:.3f}",
                "",
            ]
        )

        return "\n".join(summary_lines)


class HierarchyTester(BaseHypothesisTester):
    """Tests H1: Hierarchical Network Organization hypothesis."""

    def __init__(
        self, hub_threshold: float = 0.1, betweenness_threshold: float = 0.05, degree_distribution_alpha: float = 2.5
    ):
        super().__init__(
            "Hierarchical Organization", "Tests for hierarchical structure with hub nodes and scale-free properties"
        )
        self.hub_threshold = hub_threshold
        self.betweenness_threshold = betweenness_threshold
        self.degree_distribution_alpha = degree_distribution_alpha

    def test_hypothesis(self, analysis_results: dict[str, Any], graph: nx.Graph, **kwargs) -> HypothesisResult:
        """Test hierarchical organization hypothesis."""
        centralities = analysis_results.get("centralities", {})
        hubs = analysis_results.get("hubs", {})
        topology = analysis_results.get("topology", {})

        # Evidence metrics
        evidence_metrics = {}

        # 1. Hub presence and strength
        n_hubs = hubs.get("n_hubs", 0)
        total_nodes = topology.get("n_nodes", 1)
        hub_ratio = n_hubs / total_nodes if total_nodes > 0 else 0.0
        evidence_metrics["hub_ratio"] = hub_ratio

        # 2. Betweenness centrality concentration
        betweenness_values = centralities.get("betweenness", [])
        max_betweenness = max(betweenness_values) if betweenness_values else 0.0
        evidence_metrics["max_betweenness"] = max_betweenness

        # 3. Degree distribution analysis
        degree_analysis = self._analyze_degree_distribution(graph)
        evidence_metrics.update(degree_analysis)

        # 4. Signed network specific hierarchy
        if graph.graph.get("signed", False):
            signed_hierarchy = self._analyze_signed_hierarchy(hubs, centralities)
            evidence_metrics.update(signed_hierarchy)

        # Define evidence thresholds
        thresholds = {
            "hub_ratio": 0.1,  # At least 10% hubs
            "max_betweenness": self.betweenness_threshold,
            "degree_concentration": 0.3,
            "scale_free_fit": 0.7,
            "hub_diversity": 0.5,  # For signed networks
        }

        # Compute evidence strength
        evidence_strength = self._compute_evidence_strength(evidence_metrics, thresholds)

        # Determine support
        supported = evidence_strength > 0.6

        logger.info("The evidence metrics are ")
        logger.info(evidence_metrics)

        # Generate summary
        summary = self._generate_hierarchy_summary(evidence_metrics, supported)

        return HypothesisResult(
            hypothesis_name=self.name,
            supported=supported,
            evidence_strength=evidence_strength,
            evidence_details=evidence_metrics,
            summary=summary,
        )

    def _analyze_degree_distribution(self, graph: nx.Graph) -> dict[str, float]:
        """Analyze degree distribution for scale-free properties."""
        if graph.number_of_nodes() < 10:
            return {"degree_concentration": 0.0, "scale_free_fit": 0.0}

        degrees = [d for n, d in graph.degree()]
        if not degrees:
            return {"degree_concentration": 0.0, "scale_free_fit": 0.0}

        # Degree concentration (Gini coefficient)
        degrees_sorted = sorted(degrees)
        n = len(degrees_sorted)
        cumsum = np.cumsum(degrees_sorted)
        gini = (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(cumsum))) / (n * sum(degrees_sorted))

        # Scale-free fit (simplified power-law assessment)
        unique_degrees, counts = np.unique(degrees, return_counts=True)
        if len(unique_degrees) < 3:
            scale_free_fit = 0.0
        else:
            # Log-log correlation as proxy for power-law fit
            log_degrees = np.log(unique_degrees[unique_degrees > 0])
            log_counts = np.log(counts[unique_degrees > 0])

            if len(log_degrees) >= 3:
                correlation = abs(np.corrcoef(log_degrees, log_counts)[0, 1])
                scale_free_fit = correlation if not np.isnan(correlation) else 0.0
            else:
                scale_free_fit = 0.0

        return {"degree_concentration": float(gini), "scale_free_fit": float(scale_free_fit)}

    def _analyze_signed_hierarchy(self, hubs: dict[str, Any], centralities: dict[str, Any]) -> dict[str, float]:
        """Analyze hierarchy in signed networks."""
        metrics = {}

        # Hub type diversity
        n_excitatory = len(hubs.get("excitatory_hubs", []))
        n_inhibitory = len(hubs.get("inhibitory_hubs", []))
        n_balanced = len(hubs.get("balanced_hubs", []))
        total_hubs = n_excitatory + n_inhibitory + n_balanced

        if total_hubs > 0:
            hub_types = sum([1 for x in [n_excitatory, n_inhibitory, n_balanced] if x > 0])
            metrics["hub_diversity"] = hub_types / 3.0  # Normalize by max possible types
        else:
            metrics["hub_diversity"] = 0.0

        # Excitation-inhibition balance in hubs
        balance_ratio = hubs.get("hub_balance_ratio", 0.5)
        # Favor balanced networks (closer to 0.5 is better)
        metrics["hub_ei_balance"] = 1.0 - abs(balance_ratio - 0.5) * 2

        return metrics

    def _generate_hierarchy_summary(self, evidence: dict[str, float], supported: bool) -> str:
        """Generate human-readable summary of hierarchy evidence."""
        if supported:
            summary = "Strong hierarchical organization detected. "
            summary += f"Hub ratio: {evidence.get('hub_ratio', 0):.2f}, "
            summary += f"Max betweenness: {evidence.get('max_betweenness', 0):.3f}, "
            summary += f"Degree concentration: {evidence.get('degree_concentration', 0):.2f}"

            if "hub_diversity" in evidence:
                summary += f", Hub diversity: {evidence['hub_diversity']:.2f}"
        else:
            summary = "Limited hierarchical structure. "
            summary += f"Hub ratio: {evidence.get('hub_ratio', 0):.2f}, "
            summary += "Evidence strength insufficient for strong hierarchy."

        return summary


class ModularityTester(BaseHypothesisTester):
    """Tests H2: Modular Coordination Architecture hypothesis."""

    def __init__(
        self, modularity_threshold: float = 0.3, density_ratio_threshold: float = 2.0, min_communities: int = 2
    ):
        super().__init__(
            "Modular Architecture",
            "Tests for modular structure with distinct communities and inter/intra-community patterns",
        )
        self.modularity_threshold = modularity_threshold
        self.density_ratio_threshold = density_ratio_threshold
        self.min_communities = min_communities

    def test_hypothesis(self, analysis_results: dict[str, Any], graph: nx.Graph, **kwargs) -> HypothesisResult:
        """Test modular architecture hypothesis."""
        communities = analysis_results.get("communities", {})
        topology = analysis_results.get("topology", {})

        # Evidence metrics
        evidence_metrics = {}

        # 1. Basic modularity
        modularity = communities.get("modularity", 0.0)
        evidence_metrics["modularity"] = modularity

        # 2. Number of communities
        n_communities = communities.get("n_communities", 1)
        n_nodes = topology.get("n_nodes", 1)
        evidence_metrics["community_ratio"] = n_communities / n_nodes if n_nodes > 0 else 0.0

        # 3. Community structure quality
        community_quality = self._assess_community_quality(communities, graph)
        evidence_metrics.update(community_quality)

        # 4. Signed network modularity
        if graph.graph.get("signed", False):
            signed_modularity = self._analyze_signed_modularity(communities, graph)
            evidence_metrics.update(signed_modularity)

        # Define thresholds
        thresholds = {
            "modularity": self.modularity_threshold,
            "community_ratio": 0.1,  # At least 10% as many communities as nodes
            "density_ratio": self.density_ratio_threshold,
            "community_cohesion": 0.5,
            "signed_modularity": 0.25,  # Lower threshold for signed networks
        }

        # Compute evidence strength
        evidence_strength = self._compute_evidence_strength(evidence_metrics, thresholds)

        # Determine support
        supported = (
            evidence_strength > 0.6
            and n_communities >= self.min_communities
            and modularity > self.modularity_threshold * 0.7  # Allow some tolerance
        )

        # Generate summary
        summary = self._generate_modularity_summary(evidence_metrics, supported)

        return HypothesisResult(
            hypothesis_name=self.name,
            supported=supported,
            evidence_strength=evidence_strength,
            evidence_details=evidence_metrics,
            summary=summary,
        )

    def _assess_community_quality(self, communities: dict[str, Any], graph: nx.Graph) -> dict[str, float]:
        """Assess the quality of detected communities."""
        metrics = {}

        # Community size distribution
        community_sizes = communities.get("community_sizes", [])
        if community_sizes:
            size_variance = np.var(community_sizes)
            avg_size = np.mean(community_sizes)
            metrics["size_homogeneity"] = 1.0 / (1.0 + size_variance / max(avg_size, 1)) if avg_size > 0 else 0.0
        else:
            metrics["size_homogeneity"] = 0.0

        # Density ratio (intra vs inter community)
        if hasattr(communities, "labels") or "labels" in communities:
            density_ratio = self._compute_density_ratio(graph, communities.get("labels", []))
            metrics["density_ratio"] = min(density_ratio, 10.0) / 10.0  # Normalize, cap at 10
        else:
            metrics["density_ratio"] = 0.0

        # Community cohesion
        metrics["community_cohesion"] = (
            min(communities.get("avg_community_size", 0) / graph.number_of_nodes(), 1.0)
            if graph.number_of_nodes() > 0
            else 0.0
        )

        return metrics

    def _compute_density_ratio(self, graph: nx.Graph, labels: list[int]) -> float:
        """Compute ratio of intra-community to inter-community edge density."""
        if not labels or len(set(labels)) == 1:
            return 1.0

        intra_edges = 0
        inter_edges = 0
        intra_possible = 0
        inter_possible = 0

        nodes = list(graph.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if i < len(labels) and j < len(labels):
                    same_community = labels[i] == labels[j]

                    if same_community:
                        intra_possible += 1
                        if graph.has_edge(nodes[i], nodes[j]):
                            intra_edges += 1
                    else:
                        inter_possible += 1
                        if graph.has_edge(nodes[i], nodes[j]):
                            inter_edges += 1

        intra_density = intra_edges / intra_possible if intra_possible > 0 else 0.0
        inter_density = inter_edges / inter_possible if inter_possible > 0 else 0.0

        return intra_density / inter_density if inter_density > 0 else float("inf")

    def _analyze_signed_modularity(self, communities: dict[str, Any], graph: nx.Graph) -> dict[str, float]:
        """Analyze modularity in signed networks."""
        metrics = {}

        # Signed modularity
        signed_mod = communities.get("signed_modularity", 0.0)
        metrics["signed_modularity"] = signed_mod

        # Positive vs negative modularity
        pos_mod = communities.get("positive_modularity", 0.0)
        neg_mod = communities.get("negative_modularity", 0.0)

        if pos_mod + abs(neg_mod) > 0:
            metrics["modularity_balance"] = pos_mod / (pos_mod + abs(neg_mod))
        else:
            metrics["modularity_balance"] = 0.5

        return metrics

    def _generate_modularity_summary(self, evidence: dict[str, float], supported: bool) -> str:
        """Generate human-readable summary of modularity evidence."""
        if supported:
            summary = "Strong modular architecture detected. "
            summary += f"Modularity: {evidence.get('modularity', 0):.3f}, "
            summary += f"Density ratio: {evidence.get('density_ratio', 0):.2f}"

            if "signed_modularity" in evidence:
                summary += f", Signed modularity: {evidence['signed_modularity']:.3f}"
        else:
            summary = "Limited modular structure. "
            summary += f"Modularity: {evidence.get('modularity', 0):.3f}, "
            summary += "Evidence insufficient for strong modularity."

        return summary


class AdaptivityTester(BaseHypothesisTester):
    """Tests H3: Context-Dependent Network Topology hypothesis."""

    def __init__(
        self,
        topology_change_threshold: float = 0.1,
        edge_turnover_threshold: float = 0.15,
        sign_change_threshold: float = 0.1,
    ):
        super().__init__("Adaptive Topology", "Tests for context-dependent changes in network structure")
        self.topology_change_threshold = topology_change_threshold
        self.edge_turnover_threshold = edge_turnover_threshold
        self.sign_change_threshold = sign_change_threshold

    def test_hypothesis(
        self,
        analysis_results: dict[str, Any],
        graph: nx.Graph,
        rare_context_results: dict[str, Any] | None = None,
        common_context_results: dict[str, Any] | None = None,
        **kwargs,
    ) -> HypothesisResult:
        """Test adaptive topology hypothesis."""
        evidence_metrics = {}

        if rare_context_results and common_context_results:
            # Compare rare vs common contexts
            context_differences = self._compute_context_differences(rare_context_results, common_context_results)
            evidence_metrics.update(context_differences)

            # Signed network specific adaptivity
            if graph.graph.get("signed", False):
                sign_adaptivity = self._analyze_sign_adaptivity(rare_context_results, common_context_results)
                evidence_metrics.update(sign_adaptivity)
        else:
            # Fallback: analyze temporal or structural variability within single graph
            structural_variability = self._analyze_structural_variability(analysis_results, graph)
            evidence_metrics.update(structural_variability)

        # Define thresholds
        thresholds = {
            "topology_change": self.topology_change_threshold,
            "edge_turnover": self.edge_turnover_threshold,
            "sign_change_rate": self.sign_change_threshold,
            "clustering_change": 0.1,
            "efficiency_change": 0.1,
        }

        # Compute evidence strength
        evidence_strength = self._compute_evidence_strength(evidence_metrics, thresholds)

        # Determine support
        supported = evidence_strength > 0.5  # Lower threshold since adaptivity is harder to detect

        # Generate summary
        summary = self._generate_adaptivity_summary(evidence_metrics, supported)

        return HypothesisResult(
            hypothesis_name=self.name,
            supported=supported,
            evidence_strength=evidence_strength,
            evidence_details=evidence_metrics,
            summary=summary,
        )

    def _compute_context_differences(
        self, rare_results: dict[str, Any], common_results: dict[str, Any]
    ) -> dict[str, float]:
        """Compute differences between rare and common context networks."""
        differences = {}

        # Topology differences
        rare_topo = rare_results.get("topology", {})
        common_topo = common_results.get("topology", {})

        for metric in ["density", "avg_clustering", "global_efficiency", "avg_path_length"]:
            rare_val = rare_topo.get(metric, 0)
            common_val = common_topo.get(metric, 0)

            abs_diff = abs(rare_val - common_val)
            differences[f"{metric}_change"] = abs_diff

            # Relative change
            if common_val != 0:
                rel_change = abs_diff / abs(common_val)
                differences[f"{metric}_relative_change"] = rel_change

        # Overall topology change score
        topology_changes = [
            differences.get(f"{m}_change", 0) for m in ["density", "avg_clustering", "global_efficiency"]
        ]
        differences["topology_change"] = float(np.mean(topology_changes))

        # Community differences
        rare_comm = rare_results.get("communities", {})
        common_comm = common_results.get("communities", {})

        mod_diff = abs(rare_comm.get("modularity", 0) - common_comm.get("modularity", 0))
        comm_diff = abs(rare_comm.get("n_communities", 1) - common_comm.get("n_communities", 1))

        differences["modularity_change"] = mod_diff
        differences["community_structure_change"] = comm_diff / max(
            rare_comm.get("n_communities", 1), common_comm.get("n_communities", 1)
        )

        return differences

    def _analyze_sign_adaptivity(
        self, rare_results: dict[str, Any], common_results: dict[str, Any]
    ) -> dict[str, float]:
        """Analyze sign-specific adaptivity in signed networks."""
        metrics = {}

        # Balance changes
        rare_balance = rare_results.get("topology", {}).get("edge_balance_ratio", 0.5)
        common_balance = common_results.get("topology", {}).get("edge_balance_ratio", 0.5)

        balance_change = abs(rare_balance - common_balance)
        metrics["balance_change"] = balance_change

        # Structural balance changes
        rare_struct_balance = rare_results.get("topology", {}).get("structural_balance", 1.0)
        common_struct_balance = common_results.get("topology", {}).get("structural_balance", 1.0)

        struct_balance_change = abs(rare_struct_balance - common_struct_balance)
        metrics["structural_balance_change"] = struct_balance_change

        # Estimate sign change rate (simplified)
        metrics["sign_change_rate"] = max(balance_change, struct_balance_change)

        return metrics

    def _analyze_structural_variability(self, analysis_results: dict[str, Any], graph: nx.Graph) -> dict[str, float]:
        """Analyze structural variability within a single network."""
        metrics = {}

        # Degree distribution variance
        degrees = [d for n, d in graph.degree()]
        if degrees:
            degree_cv = np.std(degrees) / (np.mean(degrees) + 1e-10)
            metrics["degree_variability"] = min(degree_cv, 2.0) / 2.0  # Normalize
        else:
            metrics["degree_variability"] = 0.0

        # Community size variance
        communities = analysis_results.get("communities", {})
        community_sizes = communities.get("community_sizes", [])
        if len(community_sizes) > 1:
            size_cv = np.std(community_sizes) / (np.mean(community_sizes) + 1e-10)
            metrics["community_size_variability"] = min(size_cv, 2.0) / 2.0
        else:
            metrics["community_size_variability"] = 0.0

        # Estimate adaptivity from structural heterogeneity
        metrics["topology_change"] = (metrics["degree_variability"] + metrics["community_size_variability"]) / 2

        return metrics

    def _generate_adaptivity_summary(self, evidence: dict[str, float], supported: bool) -> str:
        """Generate human-readable summary of adaptivity evidence."""
        if supported:
            summary = "Context-dependent topology detected. "
            topology_change = evidence.get("topology_change", 0)
            summary += f"Topology change: {topology_change:.3f}"

            if "sign_change_rate" in evidence:
                summary += f", Sign change rate: {evidence['sign_change_rate']:.3f}"
        else:
            summary = "Limited context-dependent adaptation. "
            summary += "Topology appears relatively stable across contexts."

        return summary


class OptimizationTester(BaseHypothesisTester):
    """Tests H4: Optimized Information Flow hypothesis."""

    def __init__(
        self,
        efficiency_threshold: float = 0.5,
        cost_efficiency_threshold: float = 0.3,
        small_world_threshold: float = 1.5,
    ):
        super().__init__(
            "Optimized Information Flow",
            "Tests for optimized network properties including efficiency and cost-effectiveness",
        )
        self.efficiency_threshold = efficiency_threshold
        self.cost_efficiency_threshold = cost_efficiency_threshold
        self.small_world_threshold = small_world_threshold

    def test_hypothesis(
        self, analysis_results: dict[str, Any], graph: nx.Graph, random_baseline: dict[str, Any] | None = None, **kwargs
    ) -> HypothesisResult:
        """Test optimized information flow hypothesis."""
        topology = analysis_results.get("topology", {})
        evidence_metrics = {}

        # 1. Basic efficiency metrics
        global_eff = topology.get("global_efficiency", 0.0)
        local_eff = topology.get("local_efficiency", 0.0)
        evidence_metrics["global_efficiency"] = global_eff
        evidence_metrics["local_efficiency"] = local_eff

        # 2. Cost-efficiency trade-off
        density = topology.get("density", 0.0)
        cost_efficiency = global_eff / (density + 1e-10)  # Avoid division by zero
        evidence_metrics["cost_efficiency"] = min(cost_efficiency, 5.0) / 5.0  # Normalize

        # 3. Small-world properties
        small_world_metrics = self._assess_small_world_properties(topology, random_baseline)
        evidence_metrics.update(small_world_metrics)

        # 4. Signed network optimization
        if graph.graph.get("signed", False):
            signed_optimization = self._analyze_signed_optimization(topology, analysis_results)
            evidence_metrics.update(signed_optimization)

        # 5. Relative performance vs random networks
        if random_baseline:
            relative_performance = self._compute_relative_performance(topology, random_baseline)
            evidence_metrics.update(relative_performance)

        # Define thresholds
        thresholds = {
            "global_efficiency": self.efficiency_threshold,
            "cost_efficiency": self.cost_efficiency_threshold,
            "small_world_sigma": self.small_world_threshold,
            "efficiency_ratio": 1.2,  # At least 20% better than random
            "balance_optimization": 0.6,  # For signed networks
        }

        # Compute evidence strength
        evidence_strength = self._compute_evidence_strength(evidence_metrics, thresholds)

        # Determine support
        supported = (
            evidence_strength > 0.6 and global_eff > self.efficiency_threshold * 0.8  # Some tolerance
        )

        # Generate summary
        summary = self._generate_optimization_summary(evidence_metrics, supported)

        return HypothesisResult(
            hypothesis_name=self.name,
            supported=supported,
            evidence_strength=evidence_strength,
            evidence_details=evidence_metrics,
            summary=summary,
        )

    def _assess_small_world_properties(
        self, topology: dict[str, Any], random_baseline: dict[str, Any] | None
    ) -> dict[str, float]:
        """Assess small-world network properties."""
        metrics = {}

        clustering = topology.get("avg_clustering", 0.0)
        path_length = topology.get("avg_path_length", 0.0)

        if random_baseline:
            random_clustering = random_baseline.get("avg_clustering", clustering)
            random_path_length = random_baseline.get("avg_path_length", path_length)

            # Small-world sigma: (C/C_random) / (L/L_random)
            if random_clustering > 0 and random_path_length > 0 and path_length > 0:
                clustering_ratio = clustering / random_clustering
                path_ratio = path_length / random_path_length
                sigma = clustering_ratio / path_ratio if path_ratio > 0 else 0.0
                metrics["small_world_sigma"] = min(sigma, 10.0) / 10.0  # Normalize
            else:
                metrics["small_world_sigma"] = 0.0
        else:
            # Simplified small-world assessment without random baseline
            metrics["small_world_sigma"] = min(clustering * 2, 1.0)  # High clustering indicates small-world tendency

        metrics["clustering_coefficient"] = clustering

        return metrics

    def _analyze_signed_optimization(
        self, topology: dict[str, Any], analysis_results: dict[str, Any]
    ) -> dict[str, float]:
        """Analyze optimization in signed networks."""
        metrics = {}

        # Balance optimization
        edge_balance = topology.get("edge_balance_ratio", 0.5)
        balance_score = 1.0 - abs(edge_balance - 0.5) * 2  # Closer to 0.5 is better
        metrics["balance_optimization"] = balance_score

        # Structural balance vs frustration trade-off
        structural_balance = topology.get("structural_balance", 1.0)
        frustration = topology.get("edge_frustration", 0.0)
        balance_frustration_score = structural_balance * (1.0 - frustration)
        metrics["balance_frustration_optimization"] = balance_frustration_score

        # Hub diversity optimization (if available)
        hubs = analysis_results.get("hubs", {})
        hub_diversity = hubs.get("hub_diversity", 1.0)
        metrics["hub_diversity_optimization"] = hub_diversity / 3.0  # Normalize by max diversity

        return metrics

    def _compute_relative_performance(
        self, topology: dict[str, Any], random_baseline: dict[str, Any]
    ) -> dict[str, float]:
        """Compute performance relative to random networks."""
        metrics = {}

        # Efficiency ratio
        observed_eff = topology.get("global_efficiency", 0.0)
        random_eff = random_baseline.get("global_efficiency", observed_eff)

        if random_eff > 0:
            efficiency_ratio = observed_eff / random_eff
            metrics["efficiency_ratio"] = min(efficiency_ratio, 3.0) / 3.0  # Normalize, cap at 3x
        else:
            metrics["efficiency_ratio"] = 1.0

        # Clustering ratio
        observed_clustering = topology.get("avg_clustering", 0.0)
        random_clustering = random_baseline.get("avg_clustering", observed_clustering)

        if random_clustering > 0:
            clustering_ratio = observed_clustering / random_clustering
            metrics["clustering_ratio"] = min(clustering_ratio, 5.0) / 5.0  # Normalize
        else:
            metrics["clustering_ratio"] = 1.0

        return metrics
