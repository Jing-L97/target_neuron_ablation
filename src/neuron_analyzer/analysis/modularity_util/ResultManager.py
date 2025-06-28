import logging
import typing as t
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AnalysisMetadata:
    """Metadata for modularity analysis run."""

    timestamp: str
    analysis_id: str
    data_description: str
    neuron_groups: dict[str, list[int]]
    algorithm_config: dict[str, t.Any]
    random_seed: int | None = None


@dataclass
class ModularitySummary:
    """Summary of modularity analysis results."""

    total_groups: int
    significant_groups: int
    best_group: str
    best_modularity: float
    worst_group: str
    worst_modularity: float
    overall_finding: str
    recommendations: list[str]


class ResultsManager:
    """Simplified results manager for modularity analysis."""

    def __init__(self):
        """Initialize results manager for modularity analysis."""
        self.results = {
            "metadata": None,
            "group_modularity": {},
            "community_details": {},
            "statistical_comparisons": {},
            "summary": None,
        }

        logger.info("ModularityResultsManager initialized")

    def store_metadata(
        self,
        data_description: str,
        neuron_groups: dict[str, list[int]],
        algorithm_config: dict[str, t.Any],
        analysis_id: str | None = None,
        random_seed: int | None = None,
    ) -> str:
        """Store analysis metadata."""
        if analysis_id is None:
            analysis_id = f"modularity_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        metadata = AnalysisMetadata(
            timestamp=datetime.now().isoformat(),
            analysis_id=analysis_id,
            data_description=data_description,
            neuron_groups=neuron_groups,
            algorithm_config=algorithm_config,
            random_seed=random_seed,
        )

        self.results["metadata"] = metadata
        logger.info(f"Stored metadata for analysis: {analysis_id}")
        return analysis_id

    def store_group_modularity(
        self, group_name: str, modularity: float, signed_modularity: float, community_details: dict[str, t.Any]
    ) -> None:
        """Store modularity results for a neuron group.

        Args:
            group_name: Name of the neuron group
            modularity: Regular modularity value
            signed_modularity: Signed modularity value
            community_details: Additional community detection details

        """
        self.results["group_modularity"][group_name] = {
            "modularity": modularity,
            "signed_modularity": signed_modularity,
            "n_communities": community_details.get("n_communities", 1),
            "algorithm": community_details.get("algorithm", "unknown"),
        }

        self.results["community_details"][group_name] = community_details

        logger.debug(
            f"Stored modularity results for {group_name}: Q={modularity:.3f}, Q_signed={signed_modularity:.3f}"
        )

    def store_statistical_comparison(
        self,
        comparison_name: str,
        comparison_result: t.Any,  # GroupComparisonResult from ModularityGroupValidator
    ) -> None:
        """Store statistical comparison results."""
        # Convert result to dictionary if it's a dataclass
        if hasattr(comparison_result, "__dataclass_fields__"):
            result_dict = asdict(comparison_result)
        else:
            result_dict = comparison_result

        self.results["statistical_comparisons"][comparison_name] = result_dict

        logger.debug(f"Stored statistical comparison: {comparison_name}")

    def get_modularity_ranking(self, metric: str = "signed_modularity") -> list[tuple[str, float]]:
        """Get groups ranked by modularity.

        Args:
            metric: Either "modularity" or "signed_modularity"

        Returns:
            List of (group_name, modularity_value) tuples, ranked highest to lowest

        """
        if metric not in ["modularity", "signed_modularity"]:
            raise ValueError("Metric must be 'modularity' or 'signed_modularity'")

        rankings = []
        for group_name, results in self.results["group_modularity"].items():
            if metric in results:
                rankings.append((group_name, results[metric]))

        # Sort by modularity value (highest first)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def get_significant_groups(self) -> list[str]:
        """Get list of groups with statistically significant modularity."""
        significant_groups = []

        for comparison_name, result in self.results["statistical_comparisons"].items():
            if result.get("is_significant", False):
                # Extract group name from comparison name (e.g., "boost_vs_random" -> "boost")
                group_name = comparison_name.split("_vs_")[0]
                if group_name not in significant_groups:
                    significant_groups.append(group_name)

        return significant_groups

    def compare_groups(self, group1: str, group2: str) -> dict[str, t.Any]:
        """Compare modularity between two groups."""
        if group1 not in self.results["group_modularity"]:
            raise ValueError(f"Group {group1} not found in results")
        if group2 not in self.results["group_modularity"]:
            raise ValueError(f"Group {group2} not found in results")

        results1 = self.results["group_modularity"][group1]
        results2 = self.results["group_modularity"][group2]

        comparison = {
            "groups": [group1, group2],
            "modularity_difference": results1["modularity"] - results2["modularity"],
            "signed_modularity_difference": results1["signed_modularity"] - results2["signed_modularity"],
            "group1_values": {
                "modularity": results1["modularity"],
                "signed_modularity": results1["signed_modularity"],
                "n_communities": results1["n_communities"],
            },
            "group2_values": {
                "modularity": results2["modularity"],
                "signed_modularity": results2["signed_modularity"],
                "n_communities": results2["n_communities"],
            },
        }

        # Determine winner
        if results1["signed_modularity"] > results2["signed_modularity"]:
            comparison["higher_modularity"] = group1
        elif results2["signed_modularity"] > results1["signed_modularity"]:
            comparison["higher_modularity"] = group2
        else:
            comparison["higher_modularity"] = "tie"

        return comparison

    def generate_summary(self) -> ModularitySummary:
        """Generate comprehensive summary of modularity analysis."""
        if not self.results["group_modularity"]:
            raise ValueError("No modularity results stored")

        # Basic statistics
        total_groups = len(self.results["group_modularity"])
        significant_groups = len(self.get_significant_groups())

        # Find best and worst groups
        rankings = self.get_modularity_ranking("signed_modularity")
        best_group, best_modularity = rankings[0] if rankings else ("unknown", 0.0)
        worst_group, worst_modularity = rankings[-1] if rankings else ("unknown", 0.0)

        # Generate overall finding
        overall_finding = self._generate_overall_finding(significant_groups, total_groups)

        # Generate recommendations
        recommendations = self._generate_recommendations()

        summary = ModularitySummary(
            total_groups=total_groups,
            significant_groups=significant_groups,
            best_group=best_group,
            best_modularity=best_modularity,
            worst_group=worst_group,
            worst_modularity=worst_modularity,
            overall_finding=overall_finding,
            recommendations=recommendations,
        )

        self.results["summary"] = summary
        return summary

    def generate_report(self, include_details: bool = True) -> str:
        """Generate comprehensive text report."""
        if not self.results["summary"]:
            self.generate_summary()

        summary = self.results["summary"]
        metadata = self.results["metadata"]

        lines = ["=" * 70, "MODULARITY ANALYSIS REPORT", "=" * 70, ""]

        # Metadata section
        if metadata:
            lines.extend(
                [
                    "ANALYSIS INFORMATION",
                    "-" * 30,
                    f"Analysis ID: {metadata.analysis_id}",
                    f"Timestamp: {metadata.timestamp}",
                    f"Data: {metadata.data_description}",
                    f"Algorithm: {metadata.algorithm_config.get('algorithm', 'unknown')}",
                    f"Groups analyzed: {', '.join(metadata.neuron_groups.keys())}",
                    f"Random seed: {metadata.random_seed}",
                    "",
                ]
            )

        # Summary section
        lines.extend(
            [
                "SUMMARY RESULTS",
                "-" * 30,
                f"Total groups analyzed: {summary.total_groups}",
                f"Groups with significant modularity: {summary.significant_groups}",
                f"Significance rate: {summary.significant_groups / summary.total_groups * 100:.1f}%",
                f"Best performing group: {summary.best_group} (Q_signed = {summary.best_modularity:.3f})",
                f"Worst performing group: {summary.worst_group} (Q_signed = {summary.worst_modularity:.3f})",
                "",
                f"Overall finding: {summary.overall_finding}",
                "",
            ]
        )

        # Group rankings
        lines.extend(["GROUP RANKINGS (by signed modularity)", "-" * 40])

        rankings = self.get_modularity_ranking("signed_modularity")
        for i, (group_name, modularity) in enumerate(rankings, 1):
            group_type = "special" if not group_name.startswith("random") else "random"
            significance = "***" if group_name in self.get_significant_groups() else ""
            lines.append(f"{i}. {group_name} ({group_type}): {modularity:.4f} {significance}")

        lines.extend(["", "(*** indicates statistically significant)", ""])

        # Statistical comparisons
        if self.results["statistical_comparisons"]:
            lines.extend(["STATISTICAL COMPARISONS", "-" * 30])

            for comparison_name, result in self.results["statistical_comparisons"].items():
                status = "SIGNIFICANT" if result.get("is_significant", False) else "NOT SIGNIFICANT"
                p_val = result.get("p_value", 1.0)
                effect = result.get("effect_size", 0.0)

                lines.extend(
                    [
                        f"{comparison_name.replace('_', ' ').title()}: {status}",
                        f"  P-value: {p_val:.4f}",
                        f"  Effect size: {effect:.3f}",
                        f"  Summary: {result.get('summary', 'No summary available')}",
                        "",
                    ]
                )

        # Detailed results
        if include_details:
            lines.extend(["DETAILED GROUP RESULTS", "-" * 30])

            for group_name in sorted(self.results["group_modularity"].keys()):
                group_data = self.results["group_modularity"][group_name]
                community_data = self.results["community_details"].get(group_name, {})

                lines.extend(
                    [
                        f"{group_name.upper()}:",
                        f"  Modularity (Q): {group_data['modularity']:.4f}",
                        f"  Signed Modularity (Q_signed): {group_data['signed_modularity']:.4f}",
                        f"  Number of communities: {group_data['n_communities']}",
                        f"  Algorithm used: {group_data['algorithm']}",
                        f"  Community sizes: {community_data.get('community_sizes', 'N/A')}",
                        "",
                    ]
                )

        # Recommendations
        if summary.recommendations:
            lines.extend(["RECOMMENDATIONS", "-" * 20])

            for i, rec in enumerate(summary.recommendations, 1):
                lines.append(f"{i}. {rec}")

            lines.append("")

        lines.extend(["=" * 70, f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "=" * 70])

        return "\n".join(lines)

    def export_results(self) -> dict[str, t.Any]:
        """Export all results as a structured dictionary."""
        # Ensure summary is generated
        if not self.results["summary"]:
            self.generate_summary()

        export_data = {
            "metadata": asdict(self.results["metadata"]) if self.results["metadata"] else None,
            "group_modularity": self.results["group_modularity"].copy(),
            "community_details": self.results["community_details"].copy(),
            "statistical_comparisons": self.results["statistical_comparisons"].copy(),
            "summary": asdict(self.results["summary"]) if self.results["summary"] else None,
            "rankings": {
                "by_modularity": self.get_modularity_ranking("modularity"),
                "by_signed_modularity": self.get_modularity_ranking("signed_modularity"),
            },
            "significant_groups": self.get_significant_groups(),
        }

        return export_data

    def get_group_results(self, group_name: str) -> dict[str, t.Any]:
        """Get all results for a specific group."""
        if group_name not in self.results["group_modularity"]:
            raise ValueError(f"Group {group_name} not found in results")

        return {
            "modularity_results": self.results["group_modularity"][group_name],
            "community_details": self.results["community_details"].get(group_name, {}),
            "statistical_comparisons": {
                name: result for name, result in self.results["statistical_comparisons"].items() if group_name in name
            },
        }

    def _generate_overall_finding(self, significant_groups: int, total_groups: int) -> str:
        """Generate overall finding statement."""
        if significant_groups == 0:
            return "No groups show significantly different modularity from random baseline"
        if significant_groups == 1:
            return "One group shows significantly different modularity from random baseline"
        if significant_groups == total_groups:
            return "All groups show significantly different modularity from random baseline"
        return f"{significant_groups} out of {total_groups} groups show significantly different modularity"

    def _generate_recommendations(self) -> list[str]:
        """Generate analysis recommendations."""
        recommendations = []

        # Check significance rate
        significant_groups = len(self.get_significant_groups())
        total_groups = len(self.results["group_modularity"])
        significance_rate = significant_groups / total_groups if total_groups > 0 else 0

        if significance_rate >= 0.5:
            recommendations.append(
                "Strong evidence for modular organization - investigate biological mechanisms underlying community structure"
            )
        elif significance_rate > 0:
            recommendations.append(
                "Mixed evidence for modular organization - examine differences between significant and non-significant groups"
            )
        else:
            recommendations.append(
                "Limited evidence for modular organization - consider alternative organizational hypotheses or larger sample sizes"
            )

        # Check for strong modularity values
        rankings = self.get_modularity_ranking("signed_modularity")
        if rankings and rankings[0][1] > 0.5:
            recommendations.append(
                f"Group '{rankings[0][0]}' shows exceptionally high modularity - prioritize for detailed investigation"
            )

        # Check algorithm consistency
        algorithms_used = set()
        for group_data in self.results["group_modularity"].values():
            algorithms_used.add(group_data.get("algorithm", "unknown"))

        if len(algorithms_used) > 1:
            recommendations.append("Multiple algorithms used - verify consistency of results across detection methods")

        # Statistical recommendations
        if self.results["statistical_comparisons"]:
            all_effects = [result.get("effect_size", 0) for result in self.results["statistical_comparisons"].values()]
            avg_effect = np.mean([abs(e) for e in all_effects])

            if avg_effect > 0.8:
                recommendations.append("Large effect sizes detected - results likely to be biologically meaningful")
            elif avg_effect < 0.2:
                recommendations.append(
                    "Small effect sizes detected - consider biological significance vs statistical significance"
                )

        return recommendations[:5]  # Limit to top 5
