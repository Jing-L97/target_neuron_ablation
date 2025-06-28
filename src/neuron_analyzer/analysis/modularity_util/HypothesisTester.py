import logging
import typing as t
from dataclasses import dataclass

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HypothesisResult:
    """Result of a single hypothesis test."""

    supported: bool
    evidence_strength: float
    evidence_details: dict[str, t.Any]
    summary: str


class ModularityTester:
    """Test modularity hypothesis for neural coordination patterns."""

    def __init__(
        self, modularity_threshold: float = 0.3, significance_level: float = 0.05, min_communities: int = 2, **kwargs
    ):
        self.modularity_threshold = modularity_threshold
        self.significance_level = significance_level
        self.min_communities = min_communities
        self.params = kwargs

    def test_hypothesis(self, analysis_results: dict[str, t.Any], graph: nx.Graph, **kwargs) -> HypothesisResult:
        """Test the modularity hypothesis.

        Hypothesis: Special neuron groups exhibit significant modular organization
        compared to random baselines.

        Args:
            analysis_results: Community detection results
            graph: NetworkX graph

        Returns:
            HypothesisResult with test outcome

        """
        try:
            # Extract key metrics
            modularity = analysis_results.get("modularity", 0.0)
            signed_modularity = analysis_results.get("signed_modularity", 0.0)
            n_communities = analysis_results.get("n_communities", 1)
            community_sizes = analysis_results.get("community_sizes", [])

            # Initialize evidence components
            evidence_components = {}
            evidence_score = 0.0

            # Test 1: Modularity threshold
            modularity_test = modularity > self.modularity_threshold
            evidence_components["modularity_above_threshold"] = {
                "value": modularity,
                "threshold": self.modularity_threshold,
                "passed": modularity_test,
            }
            if modularity_test:
                evidence_score += 0.3

            # Test 2: Signed modularity (if applicable)
            if graph.graph.get("signed", False):
                signed_modularity_test = signed_modularity > self.modularity_threshold
                evidence_components["signed_modularity_above_threshold"] = {
                    "value": signed_modularity,
                    "threshold": self.modularity_threshold,
                    "passed": signed_modularity_test,
                }
                if signed_modularity_test:
                    evidence_score += 0.3
            else:
                # Use regular modularity if not signed
                evidence_components["signed_modularity_above_threshold"] = {
                    "value": modularity,
                    "threshold": self.modularity_threshold,
                    "passed": modularity_test,
                    "note": "Graph not signed, using regular modularity",
                }
                if modularity_test:
                    evidence_score += 0.3

            # Test 3: Minimum number of communities
            community_test = n_communities >= self.min_communities
            evidence_components["sufficient_communities"] = {
                "value": n_communities,
                "threshold": self.min_communities,
                "passed": community_test,
            }
            if community_test:
                evidence_score += 0.2

            # Test 4: Community size distribution
            if community_sizes:
                # Check for reasonably balanced communities (no single giant component)
                max_community_ratio = max(community_sizes) / sum(community_sizes)
                balanced_test = max_community_ratio < 0.8  # No community dominates
                evidence_components["balanced_communities"] = {
                    "max_community_ratio": max_community_ratio,
                    "threshold": 0.8,
                    "passed": balanced_test,
                }
                if balanced_test:
                    evidence_score += 0.2

            # Normalize evidence score to [0, 1]
            max_possible_score = 1.0
            evidence_strength = min(evidence_score / max_possible_score, 1.0)

            # Determine if hypothesis is supported
            supported = modularity_test and community_test and evidence_strength > 0.5

            # Generate summary
            summary = self._generate_summary(supported, modularity, signed_modularity, n_communities, evidence_strength)

            return HypothesisResult(
                supported=supported,
                evidence_strength=evidence_strength,
                evidence_details=evidence_components,
                summary=summary,
            )

        except Exception as e:
            logger.error(f"Modularity test failed: {e}")
            return HypothesisResult(
                supported=False,
                evidence_strength=0.0,
                evidence_details={"error": str(e)},
                summary=f"Test failed due to error: {e}",
            )

    def _generate_summary(
        self, supported: bool, modularity: float, signed_modularity: float, n_communities: int, evidence_strength: float
    ) -> str:
        """Generate human-readable summary of test results."""
        if supported:
            return (
                f"SUPPORTED: Network shows significant modular organization "
                f"(Q={modularity:.3f}, Q_signed={signed_modularity:.3f}, "
                f"{n_communities} communities, evidence={evidence_strength:.3f})"
            )
        return (
            f"NOT SUPPORTED: Network lacks significant modular organization "
            f"(Q={modularity:.3f}, Q_signed={signed_modularity:.3f}, "
            f"{n_communities} communities, evidence={evidence_strength:.3f})"
        )


class HypothesisTestSuite:
    """Simplified hypothesis testing suite for single modularity hypothesis."""

    def __init__(self, modularity_config: dict[str, t.Any] | None = None):
        """Initialize hypothesis test suite."""
        self.modularity_tester = ModularityTester(**(modularity_config or {}))

    def test_modularity_hypothesis(
        self, analysis_results: dict[str, t.Any], graph: nx.Graph, **kwargs
    ) -> HypothesisResult:
        """Test the modularity hypothesis for a single network.

        Args:
            analysis_results: Community detection results from CommunityAnalyzer
            graph: NetworkX graph being analyzed
            **kwargs: Additional arguments for the test

        Returns:
            HypothesisResult with test outcome

        """
        logger.info("Testing modularity hypothesis...")

        try:
            return self.modularity_tester.test_hypothesis(analysis_results, graph, **kwargs)
        except Exception as e:
            logger.warning(f"Modularity test failed: {e}")
            return HypothesisResult(
                supported=False, evidence_strength=0.0, evidence_details={"error": str(e)}, summary=f"Test failed: {e}"
            )

    def test_multiple_networks(
        self, network_analyses: dict[str, dict[str, t.Any]], graphs: dict[str, nx.Graph], **kwargs
    ) -> dict[str, HypothesisResult]:
        """Test modularity hypothesis for multiple networks.

        Args:
            network_analyses: Community detection results for each network
            graphs: NetworkX graphs for each network
            **kwargs: Additional arguments for tests

        Returns:
            Dictionary mapping network names to HypothesisResult

        """
        results = {}

        for network_name in network_analyses:
            if network_name not in graphs:
                logger.warning(f"No graph provided for {network_name}, skipping")
                continue

            logger.debug(f"Testing modularity hypothesis for: {network_name}")

            results[network_name] = self.test_modularity_hypothesis(
                network_analyses[network_name], graphs[network_name], **kwargs
            )

        return results

    def compare_networks(self, test_results: dict[str, HypothesisResult]) -> dict[str, t.Any]:
        """Generate comparative analysis across multiple networks.

        Args:
            test_results: Results from test_multiple_networks

        Returns:
            Comparative analysis report

        """
        if not test_results:
            return {"error": "No test results provided"}

        # Basic statistics
        supported_networks = [name for name, result in test_results.items() if result.supported]
        evidence_strengths = [result.evidence_strength for result in test_results.values()]

        # Find best and worst networks
        best_network = max(test_results.items(), key=lambda x: x[1].evidence_strength)
        worst_network = min(test_results.items(), key=lambda x: x[1].evidence_strength)

        report = {
            "summary": {
                "total_networks": len(test_results),
                "supported_networks": len(supported_networks),
                "support_ratio": len(supported_networks) / len(test_results),
                "avg_evidence_strength": float(np.mean(evidence_strengths)),
                "std_evidence_strength": float(np.std(evidence_strengths)),
            },
            "supported_networks": supported_networks,
            "best_network": {
                "name": best_network[0],
                "evidence_strength": best_network[1].evidence_strength,
                "summary": best_network[1].summary,
            },
            "worst_network": {
                "name": worst_network[0],
                "evidence_strength": worst_network[1].evidence_strength,
                "summary": worst_network[1].summary,
            },
            "individual_results": {
                name: {
                    "supported": result.supported,
                    "evidence_strength": result.evidence_strength,
                    "summary": result.summary,
                }
                for name, result in test_results.items()
            },
        }

        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(report)

        return report

    def _generate_recommendations(self, report: dict[str, t.Any]) -> list[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        summary = report["summary"]
        support_ratio = summary["support_ratio"]
        avg_evidence = summary["avg_evidence_strength"]

        # Overall pattern recommendations
        if support_ratio > 0.8:
            recommendations.append(
                "Strong modularity evidence across networks - investigate common organizational principles"
            )
        elif support_ratio > 0.5:
            recommendations.append(
                "Moderate modularity evidence - examine differences between supported and unsupported networks"
            )
        elif support_ratio < 0.3:
            recommendations.append("Limited modularity evidence - consider alternative organizational hypotheses")

        # Evidence strength recommendations
        if avg_evidence > 0.7:
            recommendations.append("High average evidence strength indicates robust modular organization")
        elif avg_evidence < 0.3:
            recommendations.append("Low average evidence strength suggests weak or absent modular organization")

        # Specific network recommendations
        best_network = report["best_network"]
        if best_network["evidence_strength"] > 0.8:
            recommendations.append(
                f"Network '{best_network['name']}' shows exceptional modularity - use as reference case"
            )

        return recommendations

    def export_summary(self, test_result: HypothesisResult) -> str:
        """Export single test result as human-readable summary."""
        lines = [
            "Modularity Hypothesis Test Results",
            "=" * 40,
            "",
            f"Hypothesis Supported: {'YES' if test_result.supported else 'NO'}",
            f"Evidence Strength: {test_result.evidence_strength:.3f}",
            f"Summary: {test_result.summary}",
            "",
            "Evidence Details:",
        ]

        for component, details in test_result.evidence_details.items():
            if isinstance(details, dict) and "passed" in details:
                status = "PASS" if details["passed"] else "FAIL"
                value = details.get("value", "N/A")
                threshold = details.get("threshold", "N/A")
                lines.append(f"  {component}: {status} (value={value}, threshold={threshold})")
            else:
                lines.append(f"  {component}: {details}")

        return "\n".join(lines)

    def export_comparative_summary(self, comparison_report: dict[str, t.Any]) -> str:
        """Export comparative analysis as human-readable summary."""
        lines = [
            "Comparative Modularity Analysis",
            "=" * 40,
            "",
            "Overall Summary:",
            f"  Total Networks: {comparison_report['summary']['total_networks']}",
            f"  Supported Networks: {comparison_report['summary']['supported_networks']}",
            f"  Support Ratio: {comparison_report['summary']['support_ratio']:.2%}",
            f"  Average Evidence: {comparison_report['summary']['avg_evidence_strength']:.3f}",
            "",
            "Best Network:",
            f"  {comparison_report['best_network']['name']} "
            f"(evidence: {comparison_report['best_network']['evidence_strength']:.3f})",
            "",
            "Individual Results:",
        ]

        for name, result in comparison_report["individual_results"].items():
            status = "SUPPORTED" if result["supported"] else "NOT SUPPORTED"
            lines.append(f"  {name}: {status} (evidence: {result['evidence_strength']:.3f})")

        lines.extend(["", "Recommendations:"])
        for rec in comparison_report["recommendations"]:
            lines.append(f"  • {rec}")

        return "\n".join(lines)
