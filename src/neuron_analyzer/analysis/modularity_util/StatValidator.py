import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class GroupComparisonResult:
    """Result of statistical comparison between neuron groups."""

    special_group_modularity: float
    random_group_modularity: float
    modularity_difference: float
    p_value: float
    effect_size: float
    confidence_interval: tuple[float, float]
    is_significant: bool
    interpretation: str
    summary: str


class StatisticalValidator:
    """Simplified statistical validator for comparing modularity between neuron groups."""

    def __init__(
        self,
        significance_level: float = 0.05,
        n_permutations: int = 1000,
        confidence_level: float = 0.95,
        random_seed: int = 42,
    ):
        """Initialize validator for group modularity comparison.

        Args:
            significance_level: Alpha level for statistical tests
            n_permutations: Number of permutations for permutation test
            confidence_level: Confidence level for effect size CI
            random_seed: Random seed for reproducibility

        """
        self.significance_level = significance_level
        self.n_permutations = n_permutations
        self.confidence_level = confidence_level
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def compare_special_vs_random(
        self, special_modularity: float, random_modularities: list[float], test_type: str = "permutation"
    ) -> GroupComparisonResult:
        """Compare special group modularity against random baseline groups.

        Args:
            special_modularity: Signed modularity of special neuron group
            random_modularities: List of modularity values from random groups
            test_type: Type of statistical test ("permutation", "ttest", "both")

        Returns:
            GroupComparisonResult with statistical comparison

        """
        if not random_modularities:
            raise ValueError("No random group modularities provided")

        # Basic statistics
        random_mean = np.mean(random_modularities)
        random_std = np.std(random_modularities)
        modularity_diff = special_modularity - random_mean

        # Choose appropriate test
        if test_type == "ttest" or (test_type == "permutation" and len(random_modularities) < 3):
            # Use t-test when few random groups
            result = self._perform_ttest(special_modularity, random_modularities)
        elif test_type == "permutation":
            # Use permutation test with sufficient random groups
            result = self._perform_permutation_test(special_modularity, random_modularities)
        elif test_type == "both":
            # Perform both tests for robustness
            ttest_result = self._perform_ttest(special_modularity, random_modularities)
            perm_result = self._perform_permutation_test(special_modularity, random_modularities)
            # Use more conservative p-value
            result = ttest_result if ttest_result.p_value > perm_result.p_value else perm_result
        else:
            raise ValueError(f"Unknown test type: {test_type}")

        # Enhance result with additional info
        result.special_group_modularity = special_modularity
        result.random_group_modularity = random_mean
        result.modularity_difference = modularity_diff

        # Generate interpretation and summary
        result.interpretation = self._interpret_effect_size(result.effect_size)
        result.summary = self._generate_summary(result)

        return result

    def compare_multiple_groups(
        self, group_modularities: dict[str, float], random_baseline_modularities: list[float]
    ) -> dict[str, GroupComparisonResult]:
        """Compare multiple special groups against random baselines.

        Args:
            group_modularities: Dict mapping group names to modularity values
            random_baseline_modularities: List of random group modularities

        Returns:
            Dict mapping group names to comparison results

        """
        results = {}

        for group_name, modularity in group_modularities.items():
            if group_name.startswith("random"):
                continue  # Skip random groups in the comparison

            try:
                result = self.compare_special_vs_random(
                    modularity, random_baseline_modularities, test_type="permutation"
                )
                results[group_name] = result

            except Exception as e:
                logger.warning(f"Failed to compare {group_name}: {e}")
                results[group_name] = self._create_failed_result(group_name, modularity, str(e))

        # Apply multiple testing correction if needed
        if len(results) > 1:
            results = self._apply_bonferroni_correction(results)

        return results

    def bootstrap_confidence_interval(
        self, modularity_values: list[float], confidence_level: float | None = None
    ) -> tuple[float, float]:
        """Compute bootstrap confidence interval for modularity values.

        Args:
            modularity_values: List of modularity values
            confidence_level: Confidence level (uses instance default if None)

        Returns:
            Tuple of (lower_bound, upper_bound)

        """
        if confidence_level is None:
            confidence_level = self.confidence_level

        if len(modularity_values) < 2:
            logger.warning("Insufficient data for bootstrap CI")
            return (0.0, 0.0)

        # Bootstrap resampling
        n_bootstrap = 1000
        bootstrap_means = []

        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(modularity_values, size=len(modularity_values), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))

        # Compute confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)

        return (float(ci_lower), float(ci_upper))

    def _perform_ttest(self, special_modularity: float, random_modularities: list[float]) -> GroupComparisonResult:
        """Perform one-sample t-test against random baseline mean."""
        random_mean = np.mean(random_modularities)
        random_std = np.std(random_modularities)
        n_random = len(random_modularities)

        if random_std == 0:
            # Handle case where all random values are identical
            p_value = 0.0 if special_modularity != random_mean else 1.0
            t_stat = float("inf") if special_modularity != random_mean else 0.0
        else:
            # One-sample t-test
            t_stat = (special_modularity - random_mean) / (random_std / np.sqrt(n_random))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_random - 1))

        # Effect size (Cohen's d)
        effect_size = (special_modularity - random_mean) / random_std if random_std > 0 else 0.0

        # Confidence interval for the difference
        if random_std > 0:
            margin_error = stats.t.ppf(1 - self.significance_level / 2, n_random - 1) * (random_std / np.sqrt(n_random))
            ci_lower = (special_modularity - random_mean) - margin_error
            ci_upper = (special_modularity - random_mean) + margin_error
        else:
            ci_lower = ci_upper = special_modularity - random_mean

        return GroupComparisonResult(
            special_group_modularity=special_modularity,
            random_group_modularity=random_mean,
            modularity_difference=special_modularity - random_mean,
            p_value=float(p_value),
            effect_size=float(effect_size),
            confidence_interval=(ci_lower, ci_upper),
            is_significant=p_value < self.significance_level,
            interpretation="",
            summary="",
        )

    def _perform_permutation_test(
        self, special_modularity: float, random_modularities: list[float]
    ) -> GroupComparisonResult:
        """Perform permutation test by comparing to null distribution."""
        # Create combined dataset
        all_values = [special_modularity] + random_modularities
        n_total = len(all_values)
        observed_diff = special_modularity - np.mean(random_modularities)

        # Generate null distribution
        null_diffs = []
        for _ in range(self.n_permutations):
            # Randomly assign one value as "special"
            shuffled = np.random.permutation(all_values)
            null_special = shuffled[0]
            null_random = shuffled[1:]
            null_diff = null_special - np.mean(null_random)
            null_diffs.append(null_diff)

        # Compute p-value (two-tailed)
        p_value = np.mean(np.abs(null_diffs) >= abs(observed_diff))

        # Effect size
        null_std = np.std(null_diffs)
        effect_size = observed_diff / null_std if null_std > 0 else 0.0

        # Confidence interval from null distribution
        ci_lower = np.percentile(null_diffs, 2.5)
        ci_upper = np.percentile(null_diffs, 97.5)

        return GroupComparisonResult(
            special_group_modularity=special_modularity,
            random_group_modularity=np.mean(random_modularities),
            modularity_difference=observed_diff,
            p_value=float(p_value),
            effect_size=float(effect_size),
            confidence_interval=(ci_lower, ci_upper),
            is_significant=p_value < self.significance_level,
            interpretation="",
            summary="",
        )

    def _apply_bonferroni_correction(
        self, results: dict[str, GroupComparisonResult]
    ) -> dict[str, GroupComparisonResult]:
        """Apply Bonferroni correction for multiple comparisons."""
        n_tests = len(results)
        corrected_results = {}

        for group_name, result in results.items():
            # Create corrected result
            corrected_p = min(1.0, result.p_value * n_tests)
            corrected_significant = corrected_p < self.significance_level

            corrected_result = GroupComparisonResult(
                special_group_modularity=result.special_group_modularity,
                random_group_modularity=result.random_group_modularity,
                modularity_difference=result.modularity_difference,
                p_value=corrected_p,
                effect_size=result.effect_size,
                confidence_interval=result.confidence_interval,
                is_significant=corrected_significant,
                interpretation=result.interpretation,
                summary=result.summary + f" (Bonferroni corrected p={corrected_p:.4f})",
            )

            corrected_results[group_name] = corrected_result

        return corrected_results

    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_effect = abs(effect_size)

        if abs_effect < 0.2:
            magnitude = "negligible"
        elif abs_effect < 0.5:
            magnitude = "small"
        elif abs_effect < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"

        direction = "higher" if effect_size > 0 else "lower"
        return f"{magnitude} effect ({direction} than random)"

    def _generate_summary(self, result: GroupComparisonResult) -> str:
        """Generate human-readable summary of comparison result."""
        special_mod = result.special_group_modularity
        random_mod = result.random_group_modularity
        p_val = result.p_value
        effect = result.interpretation

        significance = "significantly" if result.is_significant else "not significantly"
        direction = "higher" if special_mod > random_mod else "lower"

        summary = (
            f"Special group modularity ({special_mod:.3f}) is {significance} "
            f"{direction} than random baseline ({random_mod:.3f}), "
            f"p={p_val:.4f}, {effect}"
        )

        return summary

    def _create_failed_result(self, group_name: str, modularity: float, error_msg: str) -> GroupComparisonResult:
        """Create failed result for error cases."""
        return GroupComparisonResult(
            special_group_modularity=modularity,
            random_group_modularity=0.0,
            modularity_difference=0.0,
            p_value=1.0,
            effect_size=0.0,
            confidence_interval=(0.0, 0.0),
            is_significant=False,
            interpretation="test failed",
            summary=f"{group_name} comparison failed: {error_msg}",
        )

    def generate_comparison_report(self, results: dict[str, GroupComparisonResult]) -> str:
        """Generate comprehensive comparison report."""
        lines = ["Modularity Group Comparison Report", "=" * 50, ""]

        # Summary statistics
        total_groups = len(results)
        significant_groups = sum(1 for r in results.values() if r.is_significant)

        lines.extend(
            [
                f"Groups tested: {total_groups}",
                f"Significantly different: {significant_groups}",
                f"Significance rate: {significant_groups / total_groups * 100:.1f}%",
                f"Alpha level: {self.significance_level}",
                "",
            ]
        )

        # Individual results
        lines.append("Individual Group Results:")
        lines.append("-" * 30)

        for group_name, result in results.items():
            status = "SIGNIFICANT" if result.is_significant else "NOT SIGNIFICANT"

            lines.extend(
                [
                    f"{group_name}: {status}",
                    f"  Special modularity: {result.special_group_modularity:.4f}",
                    f"  Random baseline: {result.random_group_modularity:.4f}",
                    f"  Difference: {result.modularity_difference:.4f}",
                    f"  P-value: {result.p_value:.4f}",
                    f"  Effect size: {result.effect_size:.3f} ({result.interpretation})",
                    f"  95% CI: ({result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f})",
                    f"  Summary: {result.summary}",
                    "",
                ]
            )

        # Overall interpretation
        lines.append("Overall Interpretation:")
        lines.append("-" * 20)

        if significant_groups == 0:
            lines.append("No groups show significantly different modularity from random baseline.")
        elif significant_groups == total_groups:
            lines.append("All groups show significantly different modularity from random baseline.")
        else:
            significant_names = [name for name, r in results.items() if r.is_significant]
            lines.append(f"Groups with significant modularity: {', '.join(significant_names)}")

        return "\n".join(lines)
