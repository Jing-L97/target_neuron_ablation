import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AnalysisMetadata:
    """Metadata for analysis run."""

    timestamp: str
    analysis_id: str
    data_description: str
    neuron_groups: dict[str, list[int]]
    configuration: dict[str, Any]
    software_version: str = "1.0.0"
    random_seed: int | None = None


@dataclass
class AnalysisSummary:
    """High-level summary of analysis results."""

    total_networks: int
    supported_hypotheses: dict[str, int]
    significant_metrics: list[str]
    key_findings: list[str]
    recommendations: list[str]
    overall_score: float


class ResultsManager:
    """Comprehensive results management system for neural coordination analysis."""

    def __init__(self):
        """Initialize ResultsManager."""
        # Storage for results
        self.results_store = {
            "metadata": None,
            "network_analyses": {},
            "hypothesis_tests": {},
            "statistical_validation": {},
            "comparative_analysis": {},
            "summary": None,
        }

        logger.info("ResultsManager initialized for in-memory processing")

    def store_analysis_metadata(
        self,
        data_description: str,
        neuron_groups: dict[str, list[int]],
        configuration: dict[str, Any],
        analysis_id: str | None = None,
        random_seed: int | None = None,
    ) -> str:
        if analysis_id is None:
            analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        metadata = AnalysisMetadata(
            timestamp=datetime.now().isoformat(),
            analysis_id=analysis_id,
            data_description=data_description,
            neuron_groups=neuron_groups,
            configuration=configuration,
            random_seed=random_seed,
        )

        self.results_store["metadata"] = metadata

        logger.info(f"Stored analysis metadata with ID: {analysis_id}")
        return analysis_id

    def store_network_analysis_results(self, network_name: str, analysis_results: dict[str, Any]) -> None:
        """Store network analysis results from NetworkAnalyzer."""
        self.results_store["network_analyses"][network_name] = analysis_results
        logger.debug(f"Stored network analysis results for: {network_name}")

    def store_hypothesis_test_results(self, network_name: str, hypothesis_results: dict[str, Any]) -> None:
        """Store hypothesis test results from HypothesisTestSuite."""
        self.results_store["hypothesis_tests"][network_name] = hypothesis_results
        logger.debug(f"Stored hypothesis test results for: {network_name}")

    def store_statistical_validation_results(self, network_name: str, validation_results: dict[str, Any]) -> None:
        """Store statistical validation results from StatisticalValidator."""
        self.results_store["statistical_validation"][network_name] = validation_results
        logger.debug(f"Stored statistical validation results for: {network_name}")

    def aggregate_results(self) -> dict[str, Any]:
        """Aggregate all stored results into comprehensive summary.

        Returns:
            dictionary containing all aggregated results

        """
        logger.info("Aggregating all analysis results...")

        # Compute comparative analysis
        comparative_analysis = self._compute_comparative_analysis()
        self.results_store["comparative_analysis"] = comparative_analysis

        # Generate summary
        summary = self._generate_analysis_summary()
        self.results_store["summary"] = summary

        # Create complete results package
        aggregated_results = {
            "metadata": asdict(self.results_store["metadata"]) if self.results_store["metadata"] else None,
            "network_analyses": self.results_store["network_analyses"],
            "hypothesis_tests": self._serialize_hypothesis_results(self.results_store["hypothesis_tests"]),
            "statistical_validation": self.results_store["statistical_validation"],
            "comparative_analysis": comparative_analysis,
            "summary": asdict(summary) if summary else None,
        }

        logger.info(f"Aggregated results for {len(self.results_store['network_analyses'])} networks")
        return aggregated_results

    def generate_summary_report(self, include_recommendations: bool = True) -> str:
        """Generate comprehensive text summary report.

        Args:
            include_recommendations: Whether to include analysis recommendations

        Returns:
            Formatted summary report string

        """
        if not self.results_store["summary"]:
            self.aggregate_results()

        summary = self.results_store["summary"]
        metadata = self.results_store["metadata"]

        report_lines = ["=" * 80, "NEURAL COORDINATION ANALYSIS SUMMARY REPORT", "=" * 80, ""]

        # Metadata section
        if metadata:
            report_lines.extend(
                [
                    "ANALYSIS METADATA",
                    "-" * 40,
                    f"Analysis ID: {metadata.analysis_id}",
                    f"Timestamp: {metadata.timestamp}",
                    f"Data: {metadata.data_description}",
                    f"Networks Analyzed: {', '.join(metadata.neuron_groups.keys())}",
                    f"Random Seed: {metadata.random_seed}",
                    "",
                ]
            )

        # Summary statistics
        if summary:
            report_lines.extend(
                [
                    "SUMMARY STATISTICS",
                    "-" * 40,
                    f"Total Networks: {summary.total_networks}",
                    f"Overall Analysis Score: {summary.overall_score:.3f}",
                    "",
                ]
            )

            # Hypothesis support
            report_lines.append("Hypothesis Support:")
            for hyp_name, support_count in summary.supported_hypotheses.items():
                percentage = (support_count / summary.total_networks) * 100
                report_lines.append(f"  {hyp_name}: {support_count}/{summary.total_networks} ({percentage:.1f}%)")

            report_lines.append("")

            # Significant metrics
            if summary.significant_metrics:
                report_lines.extend(
                    ["Statistically Significant Metrics:", "  " + ", ".join(summary.significant_metrics), ""]
                )

            # Key findings
            if summary.key_findings:
                report_lines.extend(["KEY FINDINGS", "-" * 40])
                for i, finding in enumerate(summary.key_findings, 1):
                    report_lines.append(f"{i}. {finding}")
                report_lines.append("")

            # Recommendations
            if include_recommendations and summary.recommendations:
                report_lines.extend(["RECOMMENDATIONS", "-" * 40])
                for i, recommendation in enumerate(summary.recommendations, 1):
                    report_lines.append(f"{i}. {recommendation}")
                report_lines.append("")

        # Network-specific results
        report_lines.extend(["NETWORK-SPECIFIC RESULTS", "-" * 40])

        for net_name in self.results_store["network_analyses"]:
            report_lines.append(f"\n{net_name.upper()} Network:")

            # Network analysis summary
            if net_name in self.results_store["network_analyses"]:
                analysis = self.results_store["network_analyses"][net_name]
                if "summary" in analysis:
                    net_summary = analysis["summary"]
                    properties = net_summary.get("key_properties", [])
                    complexity = net_summary.get("complexity_score", 0)

                    report_lines.extend(
                        [
                            f"  Properties: {', '.join(properties) if properties else 'basic connectivity'}",
                            f"  Complexity Score: {complexity:.3f}",
                        ]
                    )

            # Hypothesis results
            if net_name in self.results_store["hypothesis_tests"]:
                hyp_results = self.results_store["hypothesis_tests"][net_name]
                supported_hyps = [hyp for hyp, result in hyp_results.items() if result.supported]

                report_lines.append(
                    f"  Supported Hypotheses: {', '.join(supported_hyps) if supported_hyps else 'None'}"
                )

            # Statistical validation
            if net_name in self.results_store["statistical_validation"]:
                validation = self.results_store["statistical_validation"][net_name]
                significant_metrics = [
                    metric
                    for metric, result in validation.items()
                    if isinstance(result, dict) and result.get("is_significant", False)
                ]

                report_lines.append(f"  Significant Metrics: {len(significant_metrics)}/{len(validation)}")

        report_lines.extend(
            ["", "=" * 80, f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "=" * 80]
        )

        return "\n".join(report_lines)

    def get_results_for_network(self, network_name: str) -> dict[str, Any]:
        """Get all results for a specific network."""
        network_results = {
            "network_analysis": self.results_store["network_analyses"].get(network_name),
            "hypothesis_tests": self.results_store["hypothesis_tests"].get(network_name),
            "statistical_validation": self.results_store["statistical_validation"].get(network_name),
        }

        return {k: v for k, v in network_results.items() if v is not None}

    def compare_networks(self, network1: str, network2: str) -> dict[str, Any]:
        """Generate detailed comparison between two networks."""
        comparison = {
            "networks": [network1, network2],
            "network_metrics_comparison": {},
            "hypothesis_comparison": {},
            "statistical_comparison": {},
        }

        # Compare network metrics
        if network1 in self.results_store["network_analyses"] and network2 in self.results_store["network_analyses"]:
            analysis1 = self.results_store["network_analyses"][network1]
            analysis2 = self.results_store["network_analyses"][network2]

            comparison["network_metrics_comparison"] = self._compare_network_metrics(analysis1, analysis2)

        # Compare hypothesis results
        if network1 in self.results_store["hypothesis_tests"] and network2 in self.results_store["hypothesis_tests"]:
            hyp1 = self.results_store["hypothesis_tests"][network1]
            hyp2 = self.results_store["hypothesis_tests"][network2]

            comparison["hypothesis_comparison"] = self._compare_hypothesis_results(hyp1, hyp2)

        # Compare statistical validation
        if (
            network1 in self.results_store["statistical_validation"]
            and network2 in self.results_store["statistical_validation"]
        ):
            val1 = self.results_store["statistical_validation"][network1]
            val2 = self.results_store["statistical_validation"][network2]

            comparison["statistical_comparison"] = self._compare_validation_results(val1, val2)

        return comparison

    def get_storage_summary(self) -> dict[str, Any]:
        """Get summary of stored results."""
        return {
            "has_metadata": self.results_store["metadata"] is not None,
            "network_analyses_count": len(self.results_store["network_analyses"]),
            "hypothesis_tests_count": len(self.results_store["hypothesis_tests"]),
            "statistical_validation_count": len(self.results_store["statistical_validation"]),
            "has_summary": self.results_store["summary"] is not None,
            "networks": list(self.results_store["network_analyses"].keys()),
        }

    def _serialize_hypothesis_results(self, hypothesis_results: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Convert HypothesisResult objects to dictionaries for serialization."""
        serialized = {}

        for network_name, network_hyps in hypothesis_results.items():
            serialized[network_name] = {}
            for hyp_name, result in network_hyps.items():
                if hasattr(result, "__dict__"):
                    # Convert dataclass or object to dictionary
                    if hasattr(result, "__dataclass_fields__"):
                        serialized[network_name][hyp_name] = asdict(result)
                    else:
                        serialized[network_name][hyp_name] = result.__dict__
                else:
                    serialized[network_name][hyp_name] = result

        return serialized

    def _compute_comparative_analysis(self) -> dict[str, Any]:
        """Compute comparative analysis across all networks."""
        comparative = {"network_rankings": {}, "hypothesis_consistency": {}, "metric_correlations": {}}

        try:
            # Network rankings by different criteria
            if self.results_store["network_analyses"]:
                comparative["network_rankings"] = self._rank_networks_by_criteria()

            # Hypothesis consistency across networks
            if self.results_store["hypothesis_tests"]:
                comparative["hypothesis_consistency"] = self._analyze_hypothesis_consistency()

            # Metric correlations
            if self.results_store["network_analyses"]:
                comparative["metric_correlations"] = self._compute_metric_correlations()

        except Exception as e:
            logger.warning(f"Error in comparative analysis: {e}")

        return comparative

    def _generate_analysis_summary(self) -> AnalysisSummary:
        """Generate high-level analysis summary."""
        total_networks = len(self.results_store["network_analyses"])

        # Count hypothesis support
        supported_hypotheses = {"H1_hierarchy": 0, "H2_modularity": 0, "H3_adaptivity": 0, "H4_optimization": 0}

        for network_hyps in self.results_store["hypothesis_tests"].values():
            for hyp_name, result in network_hyps.items():
                if hyp_name in supported_hypotheses and result.supported:
                    supported_hypotheses[hyp_name] += 1

        # Collect significant metrics
        significant_metrics = set()
        for network_val in self.results_store["statistical_validation"].values():
            for metric_name, result in network_val.items():
                if isinstance(result, dict) and result.get("is_significant", False):
                    significant_metrics.add(metric_name)

        # Generate key findings
        key_findings = self._extract_key_findings()

        # Generate recommendations
        recommendations = self._generate_recommendations()

        # Compute overall score
        overall_score = self._compute_overall_analysis_score(supported_hypotheses, significant_metrics)

        return AnalysisSummary(
            total_networks=total_networks,
            supported_hypotheses=supported_hypotheses,
            significant_metrics=list(significant_metrics),
            key_findings=key_findings,
            recommendations=recommendations,
            overall_score=overall_score,
        )

    def _extract_key_findings(self) -> list[str]:
        """Extract key findings from analysis results."""
        findings = []

        # Check for strong hypothesis support
        hyp_support = {}
        for network_hyps in self.results_store["hypothesis_tests"].values():
            for hyp_name, result in network_hyps.items():
                if hyp_name not in hyp_support:
                    hyp_support[hyp_name] = []
                hyp_support[hyp_name].append(result.supported)

        for hyp_name, support_list in hyp_support.items():
            support_rate = sum(support_list) / len(support_list)
            if support_rate >= 0.7:
                findings.append(
                    f"Strong evidence for {hyp_name.replace('_', ' ')} across networks ({support_rate:.1%} support)"
                )
            elif support_rate <= 0.3:
                findings.append(
                    f"Limited evidence for {hyp_name.replace('_', ' ')} across networks ({support_rate:.1%} support)"
                )

        # Check for network-specific patterns
        if len(self.results_store["network_analyses"]) > 1:
            # Find the most complex network
            complexity_scores = {}
            for net_name, analysis in self.results_store["network_analyses"].items():
                complexity = analysis.get("summary", {}).get("complexity_score", 0)
                complexity_scores[net_name] = complexity

            if complexity_scores:
                most_complex = max(complexity_scores.items(), key=lambda x: x[1])
                if most_complex[1] > 0.5:
                    findings.append(
                        f"{most_complex[0]} network shows highest structural complexity (score: {most_complex[1]:.3f})"
                    )

        return findings[:5]  # Limit to top 5 findings

    def _generate_recommendations(self) -> list[str]:
        """Generate analysis recommendations."""
        recommendations = []

        # Check hypothesis support patterns
        hyp_support = {}
        for network_hyps in self.results_store["hypothesis_tests"].values():
            for hyp_name, result in network_hyps.items():
                if hyp_name not in hyp_support:
                    hyp_support[hyp_name] = []
                hyp_support[hyp_name].append(result.evidence_strength)

        # Recommend further investigation for strong patterns
        for hyp_name, strengths in hyp_support.items():
            avg_strength = sum(strengths) / len(strengths)
            if avg_strength > 0.7:
                recommendations.append(
                    f"Investigate mechanisms underlying {hyp_name.replace('_', ' ')} - strong evidence detected"
                )

        # Check for statistical validation issues
        validation_rates = {}
        for net_name, validation in self.results_store["statistical_validation"].items():
            significant_count = sum(
                1 for result in validation.values() if isinstance(result, dict) and result.get("is_significant", False)
            )
            validation_rates[net_name] = significant_count / len(validation) if validation else 0

        if validation_rates:
            avg_validation = sum(validation_rates.values()) / len(validation_rates)
            if avg_validation < 0.3:
                recommendations.append(
                    "Consider larger sample sizes or different null models - low statistical significance rates"
                )

        # Network-specific recommendations
        if len(self.results_store["network_analyses"]) > 1:
            recommendations.append(
                "Perform detailed comparative analysis between network types to identify differential mechanisms"
            )

        return recommendations[:5]  # Limit to top 5 recommendations

    def _compute_overall_analysis_score(
        self, supported_hypotheses: dict[str, int], significant_metrics: list[str]
    ) -> float:
        """Compute overall quality score for the analysis."""
        total_networks = len(self.results_store["network_analyses"])
        if total_networks == 0:
            return 0.0

        # Hypothesis support score (0-0.4)
        hyp_score = sum(supported_hypotheses.values()) / (len(supported_hypotheses) * total_networks) * 0.4

        # Statistical significance score (0-0.3)
        total_possible_metrics = len(significant_metrics) * total_networks if significant_metrics else 1
        actual_significant = len(significant_metrics)
        sig_score = min(actual_significant / max(total_possible_metrics * 0.3, 1), 0.3)

        # Data quality score (0-0.3)
        data_quality = 0.3  # Base score, could be enhanced with actual data quality metrics

        overall_score = hyp_score + sig_score + data_quality
        return min(overall_score, 1.0)

    def _rank_networks_by_criteria(self) -> dict[str, list[str]]:
        """Rank networks by different criteria."""
        rankings = {}

        network_scores = {}
        for net_name, analysis in self.results_store["network_analyses"].items():
            network_scores[net_name] = {
                "complexity": analysis.get("summary", {}).get("complexity_score", 0),
                "efficiency": analysis.get("topology", {}).get("global_efficiency", 0),
                "modularity": analysis.get("communities", {}).get("modularity", 0),
                "clustering": analysis.get("topology", {}).get("avg_clustering", 0),
            }

        # Create rankings for each criterion
        for criterion in ["complexity", "efficiency", "modularity", "clustering"]:
            sorted_networks = sorted(network_scores.items(), key=lambda x: x[1][criterion], reverse=True)
            rankings[f"by_{criterion}"] = [net[0] for net in sorted_networks]

        return rankings

    def _analyze_hypothesis_consistency(self) -> dict[str, Any]:
        """Analyze consistency of hypothesis support across networks."""
        consistency = {}

        all_hypotheses = set()
        for network_hyps in self.results_store["hypothesis_tests"].values():
            all_hypotheses.update(network_hyps.keys())

        for hyp_name in all_hypotheses:
            support_values = []
            strength_values = []

            for network_hyps in self.results_store["hypothesis_tests"].values():
                if hyp_name in network_hyps:
                    support_values.append(network_hyps[hyp_name].supported)
                    strength_values.append(network_hyps[hyp_name].evidence_strength)

            if support_values:
                consistency[hyp_name] = {
                    "support_rate": sum(support_values) / len(support_values),
                    "avg_strength": sum(strength_values) / len(strength_values),
                    "strength_std": np.std(strength_values),
                    "consistency_score": 1.0 - np.std([1 if s else 0 for s in support_values]),
                }

        return consistency

    def _compute_metric_correlations(self) -> dict[str, float]:
        """Compute correlations between network metrics across networks."""
        correlations = {}

        # Collect metrics from all networks
        all_metrics = {}
        for net_name, analysis in self.results_store["network_analyses"].items():
            topology = analysis.get("topology", {})
            communities = analysis.get("communities", {})

            metrics = {**topology, **communities}
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(value)

        # Compute correlations between metrics
        metric_names = list(all_metrics.keys())
        for i, metric1 in enumerate(metric_names):
            for metric2 in metric_names[i + 1 :]:
                if len(all_metrics[metric1]) == len(all_metrics[metric2]) and len(all_metrics[metric1]) > 1:
                    corr = np.corrcoef(all_metrics[metric1], all_metrics[metric2])[0, 1]
                    if not np.isnan(corr):
                        correlations[f"{metric1}_vs_{metric2}"] = float(corr)

        return correlations

    def _compare_network_metrics(self, analysis1: dict[str, Any], analysis2: dict[str, Any]) -> dict[str, Any]:
        """Compare metrics between two network analyses."""
        comparison = {}

        # Compare topology metrics
        topo1 = analysis1.get("topology", {})
        topo2 = analysis2.get("topology", {})

        common_metrics = set(topo1.keys()) & set(topo2.keys())
        for metric in common_metrics:
            if isinstance(topo1[metric], (int, float)) and isinstance(topo2[metric], (int, float)):
                diff = topo2[metric] - topo1[metric]
                rel_diff = diff / abs(topo1[metric]) if topo1[metric] != 0 else float("inf")

                comparison[metric] = {
                    "network1_value": topo1[metric],
                    "network2_value": topo2[metric],
                    "absolute_difference": diff,
                    "relative_difference": rel_diff,
                }

        return comparison

    def _compare_hypothesis_results(self, hyp1: dict[str, Any], hyp2: dict[str, Any]) -> dict[str, Any]:
        """Compare hypothesis results between two networks."""
        comparison = {}

        common_hypotheses = set(hyp1.keys()) & set(hyp2.keys())
        for hyp_name in common_hypotheses:
            result1 = hyp1[hyp_name]
            result2 = hyp2[hyp_name]

            comparison[hyp_name] = {
                "network1_supported": result1.supported,
                "network2_supported": result2.supported,
                "network1_strength": result1.evidence_strength,
                "network2_strength": result2.evidence_strength,
                "agreement": result1.supported == result2.supported,
                "strength_difference": result2.evidence_strength - result1.evidence_strength,
            }

        return comparison

    def _compare_validation_results(self, val1: dict[str, Any], val2: dict[str, Any]) -> dict[str, Any]:
        """Compare statistical validation results between two networks."""
        comparison = {}

        common_metrics = set(val1.keys()) & set(val2.keys())
        for metric_name in common_metrics:
            result1 = val1[metric_name]
            result2 = val2[metric_name]

            if isinstance(result1, dict) and isinstance(result2, dict):
                comparison[metric_name] = {
                    "network1_significant": result1.get("is_significant", False),
                    "network2_significant": result2.get("is_significant", False),
                    "network1_p_value": result1.get("p_value", 1.0),
                    "network2_p_value": result2.get("p_value", 1.0),
                    "significance_agreement": result1.get("is_significant", False)
                    == result2.get("is_significant", False),
                }

        return comparison
