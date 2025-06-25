import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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


class BaseExporter(ABC):
    """Abstract base class for results exporters."""

    @abstractmethod
    def export(self, data: Any, filepath: Path, **kwargs) -> bool:
        """Export data to specified filepath."""


class VisualizationGenerator:
    """Generate visualizations for analysis results."""

    def __init__(self, style: str = "seaborn", figsize: tuple[int, int] = (10, 6)):
        self.style = style
        self.figsize = figsize
        plt.style.use(style)
        sns.set_palette("husl")

    def create_hypothesis_support_chart(
        self, hypothesis_results: dict[str, dict[str, Any]], output_path: Path | None = None
    ) -> plt.Figure:
        """Create bar chart showing hypothesis support across networks."""
        # Extract support data
        networks = list(hypothesis_results.keys())
        hypotheses = ["H1_hierarchy", "H2_modularity", "H3_adaptivity", "H4_optimization"]

        support_matrix = []
        for network in networks:
            network_support = []
            for hyp in hypotheses:
                if hyp in hypothesis_results[network]:
                    supported = hypothesis_results[network][hyp].supported
                    network_support.append(1 if supported else 0)
                else:
                    network_support.append(0)
            support_matrix.append(network_support)

        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)

        x = np.arange(len(hypotheses))
        width = 0.8 / len(networks)

        for i, network in enumerate(networks):
            offset = (i - len(networks) / 2 + 0.5) * width
            bars = ax.bar(x + offset, support_matrix[i], width, label=network, alpha=0.8)

            # Add value labels on bars
            for bar, value in zip(bars, support_matrix[i], strict=False):
                if value > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        "✓",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

        ax.set_xlabel("Hypotheses")
        ax.set_ylabel("Support")
        ax.set_title("Hypothesis Support Across Networks")
        ax.set_xticks(x)
        ax.set_xticklabels([h.replace("_", "\n").replace("H", "H") for h in hypotheses])
        ax.set_ylim(0, 1.2)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved hypothesis support chart to {output_path}")

        return fig

    def create_evidence_strength_heatmap(
        self, hypothesis_results: dict[str, dict[str, Any]], output_path: Path | None = None
    ) -> plt.Figure:
        """Create heatmap of evidence strength across networks and hypotheses."""
        networks = list(hypothesis_results.keys())
        hypotheses = ["H1_hierarchy", "H2_modularity", "H3_adaptivity", "H4_optimization"]

        # Create evidence strength matrix
        evidence_matrix = []
        for network in networks:
            network_evidence = []
            for hyp in hypotheses:
                if hyp in hypothesis_results[network]:
                    strength = hypothesis_results[network][hyp].evidence_strength
                    network_evidence.append(strength)
                else:
                    network_evidence.append(0.0)
            evidence_matrix.append(network_evidence)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(evidence_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

        # Set ticks and labels
        ax.set_xticks(np.arange(len(hypotheses)))
        ax.set_yticks(np.arange(len(networks)))
        ax.set_xticklabels([h.replace("_", "\n") for h in hypotheses])
        ax.set_yticklabels(networks)

        # Add text annotations
        for i in range(len(networks)):
            for j in range(len(hypotheses)):
                text = ax.text(
                    j, i, f"{evidence_matrix[i][j]:.2f}", ha="center", va="center", color="black", fontweight="bold"
                )

        ax.set_title("Evidence Strength Across Networks and Hypotheses")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Evidence Strength", rotation=270, labelpad=20)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved evidence strength heatmap to {output_path}")

        return fig

    def create_metric_comparison_plot(
        self, network_metrics: dict[str, dict[str, float]], metrics_to_plot: list[str], output_path: Path | None = None
    ) -> plt.Figure:
        """Create comparison plot for selected metrics across networks."""
        networks = list(network_metrics.keys())
        n_metrics = len(metrics_to_plot)

        # Create subplots
        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]

            # Extract metric values
            values = []
            labels = []
            for network in networks:
                if metric in network_metrics[network]:
                    values.append(network_metrics[network][metric])
                    labels.append(network)

            # Create bar plot
            bars = ax.bar(labels, values, alpha=0.7)

            # Add value labels on bars
            for bar, value in zip(bars, values, strict=False):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.3f}", ha="center", va="bottom"
                )

            ax.set_title(metric.replace("_", " ").title())
            ax.set_ylabel("Value")
            ax.tick_params(axis="x", rotation=45)

        plt.suptitle("Network Metrics Comparison")
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved metrics comparison plot to {output_path}")

        return fig

    def create_statistical_significance_plot(
        self, validation_results: dict[str, dict[str, Any]], output_path: Path | None = None
    ) -> plt.Figure:
        """Create plot showing statistical significance of metrics."""
        # Extract p-values and effect sizes
        metrics = []
        p_values = []
        effect_sizes = []
        significance = []

        for metric_name, result in validation_results.items():
            if isinstance(result, dict) and "p_value" in result:
                metrics.append(metric_name)
                p_values.append(result["p_value"])
                effect_sizes.append(abs(result.get("effect_size", 0)))
                significance.append(result.get("is_significant", False))

        if not metrics:
            # Fallback: create empty plot
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(
                0.5, 0.5, "No statistical validation data available", ha="center", va="center", transform=ax.transAxes
            )
            return fig

        # Create scatter plot
        fig, ax = plt.subplots(figsize=self.figsize)

        colors = ["red" if sig else "blue" for sig in significance]
        sizes = [100 * es for es in effect_sizes]

        scatter = ax.scatter(range(len(metrics)), p_values, c=colors, s=sizes, alpha=0.6)

        # Add significance line
        ax.axhline(y=0.05, color="red", linestyle="--", alpha=0.5, label="α = 0.05")

        # Customize plot
        ax.set_xlabel("Metrics")
        ax.set_ylabel("P-value")
        ax.set_title("Statistical Significance of Network Metrics")
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([m.replace("_", "\n") for m in metrics], rotation=45)
        ax.set_yscale("log")
        ax.legend(["α = 0.05", "Significant", "Not Significant"])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved statistical significance plot to {output_path}")

        return fig


class ResultsManager:
    """Comprehensive results management system for neural coordination analysis.

    Handles aggregation, export, visualization, and reporting of analysis results
    from all components of the restructured framework.
    """

    def __init__(
        self,
        output_directory: str | Path = "analysis_results",
        auto_export: bool = True,
        create_visualizations: bool = True,
    ):
        """Initialize ResultsManager.

        Args:
            output_directory: Directory for saving results
            auto_export: Whether to automatically export results
            create_visualizations: Whether to create visualizations

        """
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.auto_export = auto_export
        self.create_visualizations = create_visualizations

        # Initialize exporters
        self.exporters = {
            "json": JSONExporter(),
            "csv": CSVExporter(),
            "pickle": PickleExporter(),
            "html": HTMLReportExporter(),
        }

        # Initialize visualization generator
        if create_visualizations:
            self.viz_generator = VisualizationGenerator()

        # Storage for results
        self.results_store = {
            "metadata": None,
            "network_analyses": {},
            "hypothesis_tests": {},
            "statistical_validation": {},
            "comparative_analysis": {},
            "summary": None,
        }

        logger.info(f"ResultsManager initialized with output directory: {self.output_dir}")

    def store_analysis_metadata(
        self,
        data_description: str,
        neuron_groups: dict[str, list[int]],
        configuration: dict[str, Any],
        analysis_id: str | None = None,
        random_seed: int | None = None,
    ) -> str:
        """Store metadata for the analysis run.

        Args:
            data_description: Description of the data being analyzed
            neuron_groups: Dictionary of neuron group definitions
            configuration: Analysis configuration parameters
            analysis_id: Optional custom analysis ID
            random_seed: Random seed used for analysis

        Returns:
            Generated analysis ID

        """
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

        if self.auto_export:
            self._auto_export_network_results(network_name, analysis_results)

    def store_hypothesis_test_results(self, network_name: str, hypothesis_results: dict[str, Any]) -> None:
        """Store hypothesis test results from HypothesisTestSuite."""
        self.results_store["hypothesis_tests"][network_name] = hypothesis_results
        logger.debug(f"Stored hypothesis test results for: {network_name}")

        if self.auto_export:
            self._auto_export_hypothesis_results(network_name, hypothesis_results)

    def store_statistical_validation_results(self, network_name: str, validation_results: dict[str, Any]) -> None:
        """Store statistical validation results from StatisticalValidator."""
        self.results_store["statistical_validation"][network_name] = validation_results
        logger.debug(f"Stored statistical validation results for: {network_name}")

        if self.auto_export:
            self._auto_export_validation_results(network_name, validation_results)

    def aggregate_results(self) -> dict[str, Any]:
        """Aggregate all stored results into comprehensive summary.

        Returns:
            Dictionary containing all aggregated results

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
            "metadata": self.results_store["metadata"],
            "network_analyses": self.results_store["network_analyses"],
            "hypothesis_tests": self.results_store["hypothesis_tests"],
            "statistical_validation": self.results_store["statistical_validation"],
            "comparative_analysis": comparative_analysis,
            "summary": summary,
        }

        logger.info(f"Aggregated results for {len(self.results_store['network_analyses'])} networks")
        return aggregated_results

    def export_results(
        self, export_formats: list[str] = ["json", "html"], filename_prefix: str = "neural_coordination_analysis"
    ) -> dict[str, bool]:
        """Export aggregated results in specified formats.

        Args:
            export_formats: List of export formats ('json', 'csv', 'pickle', 'html')
            filename_prefix: Prefix for output filenames

        Returns:
            Dictionary mapping formats to success status

        """
        aggregated_results = self.aggregate_results()
        export_status = {}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for format_name in export_formats:
            if format_name not in self.exporters:
                logger.warning(f"Unknown export format: {format_name}")
                export_status[format_name] = False
                continue

            filename = f"{filename_prefix}_{timestamp}.{format_name}"
            filepath = self.output_dir / filename

            success = self.exporters[format_name].export(aggregated_results, filepath)
            export_status[format_name] = success

        return export_status

    def create_visualizations(self, output_subdir: str = "visualizations") -> dict[str, Path]:
        """Create comprehensive visualizations of results.

        Args:
            output_subdir: Subdirectory name for visualizations

        Returns:
            Dictionary mapping visualization names to file paths

        """
        if not self.create_visualizations:
            logger.warning("Visualization creation is disabled")
            return {}

        viz_dir = self.output_dir / output_subdir
        viz_dir.mkdir(exist_ok=True)

        created_plots = {}

        try:
            # Hypothesis support chart
            if self.results_store["hypothesis_tests"]:
                fig = self.viz_generator.create_hypothesis_support_chart(
                    self.results_store["hypothesis_tests"], viz_dir / "hypothesis_support.png"
                )
                created_plots["hypothesis_support"] = viz_dir / "hypothesis_support.png"
                plt.close(fig)

            # Evidence strength heatmap
            if self.results_store["hypothesis_tests"]:
                fig = self.viz_generator.create_evidence_strength_heatmap(
                    self.results_store["hypothesis_tests"], viz_dir / "evidence_strength_heatmap.png"
                )
                created_plots["evidence_heatmap"] = viz_dir / "evidence_strength_heatmap.png"
                plt.close(fig)

            # Network metrics comparison
            if self.results_store["network_analyses"]:
                network_metrics = {}
                for net_name, analysis in self.results_store["network_analyses"].items():
                    if "topology" in analysis:
                        network_metrics[net_name] = analysis["topology"]

                if network_metrics:
                    metrics_to_plot = ["density", "avg_clustering", "global_efficiency", "modularity"]
                    available_metrics = [
                        m for m in metrics_to_plot if any(m in metrics for metrics in network_metrics.values())
                    ]

                    if available_metrics:
                        fig = self.viz_generator.create_metric_comparison_plot(
                            network_metrics,
                            available_metrics[:4],  # Limit to 4 metrics
                            viz_dir / "network_metrics_comparison.png",
                        )
                        created_plots["metrics_comparison"] = viz_dir / "network_metrics_comparison.png"
                        plt.close(fig)

            # Statistical significance plot
            if self.results_store["statistical_validation"]:
                # Combine validation results from all networks
                combined_validation = {}
                for net_name, validation in self.results_store["statistical_validation"].items():
                    for metric_name, result in validation.items():
                        combined_key = f"{net_name}_{metric_name}"
                        combined_validation[combined_key] = result

                if combined_validation:
                    fig = self.viz_generator.create_statistical_significance_plot(
                        combined_validation, viz_dir / "statistical_significance.png"
                    )
                    created_plots["statistical_significance"] = viz_dir / "statistical_significance.png"
                    plt.close(fig)

            logger.info(f"Created {len(created_plots)} visualizations in {viz_dir}")

        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")

        return created_plots

    def generate_summary_report(self, include_recommendations: bool = True) -> str:
        """Generate comprehensive text summary report."""
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

    def export_summary_report(self, filename: str = "summary_report.txt") -> Path:
        """Export summary report as text file."""
        report_content = self.generate_summary_report()
        output_path = self.output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"Exported summary report to {output_path}")
        return output_path

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

    def _compute_comparative_analysis(self) -> dict[str, Any]:
        """Compute comparative analysis across all networks."""
        comparative = {
            "network_rankings": {},
            "hypothesis_consistency": {},
            "metric_correlations": {},
            "cluster_analysis": {},
        }

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
            network_names = list(self.results_store["network_analyses"].keys())

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

    def _auto_export_network_results(self, network_name: str, results: dict[str, Any]) -> None:
        """Auto-export network analysis results."""
        filename = f"network_analysis_{network_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename

        self.exporters["json"].export(results, filepath)

    def _auto_export_hypothesis_results(self, network_name: str, results: dict[str, Any]) -> None:
        """Auto-export hypothesis test results."""
        filename = f"hypothesis_tests_{network_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename

        # Convert HypothesisResult objects to dictionaries
        serializable_results = {}
        for hyp_name, result in results.items():
            if hasattr(result, "__dict__"):
                serializable_results[hyp_name] = (
                    asdict(result) if hasattr(result, "__dataclass_fields__") else result.__dict__
                )
            else:
                serializable_results[hyp_name] = result

        self.exporters["json"].export(serializable_results, filepath)

    def _auto_export_validation_results(self, network_name: str, results: dict[str, Any]) -> None:
        """Auto-export statistical validation results."""
        filename = f"statistical_validation_{network_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename

        self.exporters["json"].export(results, filepath)

    def cleanup_temporary_files(self, keep_final_results: bool = True) -> None:
        """Clean up temporary files, optionally keeping final aggregated results."""
        if not keep_final_results:
            # Remove all files
            for file in self.output_dir.glob("*"):
                if file.is_file():
                    file.unlink()
            logger.info("Cleaned up all temporary files")
        else:
            # Keep only aggregated results and visualizations
            final_patterns = ["*aggregated*", "*summary*", "visualizations/*"]
            files_to_keep = set()

            for pattern in final_patterns:
                files_to_keep.update(self.output_dir.glob(pattern))

            for file in self.output_dir.glob("*"):
                if file.is_file() and file not in files_to_keep:
                    file.unlink()

            logger.info("Cleaned up temporary files, kept final results")

    def get_output_directory(self) -> Path:
        """Get the output directory path."""
        return self.output_dir

    def list_generated_files(self) -> list[Path]:
        """List all files generated in the output directory."""
        return list(self.output_dir.rglob("*"))

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
