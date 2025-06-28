import logging
import typing as t
from dataclasses import dataclass

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

# TODO:


@dataclass
class ThresholdResult:
    """Result of automatic threshold selection."""

    threshold_value: float
    n_significant_edges: int
    n_positive_edges: int
    n_negative_edges: int
    significance_level: float
    correction_method: str
    critical_t_value: float
    description: str


class StatisticalThresholdSelector:
    """Automatic threshold selection based on statistical significance of correlations.

    Implements Method 1: Statistical significance-based thresholds that preserve
    edges based on their statistical validity rather than arbitrary correlation values.
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        correction_method: str = "fdr_bh",  # "bonferroni", "fdr_bh", "fdr_by", "none"
        min_effect_size: float | None = None,
        two_tailed: bool = True,
    ):
        """Initialize statistical threshold selector.

        Args:
            significance_level: Alpha level for statistical tests (default: 0.05)
            correction_method: Multiple testing correction method
            min_effect_size: Minimum absolute correlation for practical significance
            two_tailed: Whether to use two-tailed test (tests for any relationship)

        """
        self.significance_level = significance_level
        self.correction_method = correction_method
        self.min_effect_size = min_effect_size
        self.two_tailed = two_tailed

        logger.info(
            f"StatisticalThresholdSelector initialized: α={significance_level}, "
            f"correction={correction_method}, two_tailed={two_tailed}"
        )

    def select_threshold(self, correlation_matrix: np.ndarray) -> float:
        """Select threshold based on statistical significance of correlations.

        Args:
            correlation_matrix: Square correlation matrix (n_neurons x n_neurons)

        Returns:
            float: Minimum absolute correlation threshold for statistical significance

        """
        if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
            raise ValueError("Correlation matrix must be square")

        n_neurons = correlation_matrix.shape[0]

        if n_neurons < 3:
            logger.warning("Too few neurons for statistical testing")
            return 1.0  # Impossible threshold (no edges)

        # Extract unique correlations (upper triangle, excluding diagonal)
        correlations, p_values = self._extract_correlations_and_pvalues(correlation_matrix)

        if len(correlations) == 0:
            logger.warning("No correlations to test")
            return 1.0  # Impossible threshold (no edges)

        # Apply multiple testing correction
        corrected_results = self._apply_multiple_testing_correction(p_values)
        significant_mask = corrected_results["significant_mask"]

        # Apply effect size filter if specified
        if self.min_effect_size is not None:
            effect_size_mask = np.abs(correlations) >= self.min_effect_size
            significant_mask = significant_mask & effect_size_mask
            logger.info(f"Applied minimum effect size filter: |r| >= {self.min_effect_size}")

        # Determine threshold based on significant correlations
        threshold_value = self._determine_threshold_value(correlations, significant_mask)

        logger.info(
            f"Selected threshold: {threshold_value:.4f} "
            f"({np.sum(significant_mask)}/{len(correlations)} significant correlations)"
        )
        return threshold_value

    def select_threshold_from_data(self, activation_data: np.ndarray) -> float:
        """Select threshold directly from activation data.

        Args:
            activation_data: Activation matrix (n_contexts x n_neurons)

        Returns:
            float: Minimum absolute correlation threshold for statistical significance

        """
        if activation_data.shape[0] < 3:
            raise ValueError("Need at least 3 contexts for correlation analysis")

        # Compute correlation matrix
        correlation_matrix = np.corrcoef(activation_data.T)
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # Select threshold
        threshold_value = self.select_threshold(correlation_matrix)

        logger.info(f"Threshold from data: {threshold_value:.4f} (n_contexts={activation_data.shape[0]})")
        return threshold_value

    def apply_threshold(self, correlation_matrix: np.ndarray, threshold_value: float) -> np.ndarray:
        """Apply statistical threshold to correlation matrix.

        Args:
            correlation_matrix: Original correlation matrix
            threshold_value: Threshold value (minimum absolute correlation)

        Returns:
            Thresholded correlation matrix (non-significant correlations set to 0)

        """
        n_neurons = correlation_matrix.shape[0]
        thresholded_matrix = correlation_matrix.copy()

        # Simple absolute threshold application
        # Set correlations below threshold to zero, preserve signs above threshold
        mask = np.abs(correlation_matrix) < threshold_value
        thresholded_matrix[mask] = 0.0

        # Keep diagonal as 1 (self-correlations)
        np.fill_diagonal(thresholded_matrix, 1.0)

        n_edges_before = np.sum(np.abs(correlation_matrix) > 0) - n_neurons  # Exclude diagonal
        n_edges_after = np.sum(np.abs(thresholded_matrix) > 0) - n_neurons  # Exclude diagonal

        logger.debug(f"Applied threshold {threshold_value:.4f}: {n_edges_after}/{n_edges_before} edges retained")

        return thresholded_matrix

    def _extract_correlations_and_pvalues(self, correlation_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract correlations and compute p-values from correlation matrix."""
        n_neurons = correlation_matrix.shape[0]
        correlations = []
        p_values = []

        # Estimate sample size from the correlation precision
        # This is approximate since we don't have original data
        # Use a reasonable default or try to infer from correlation precision
        n_samples = self._estimate_sample_size_from_matrix(correlation_matrix)

        # Extract upper triangle correlations
        for i in range(n_neurons):
            for j in range(i + 1, n_neurons):
                correlation = correlation_matrix[i, j]

                # Skip perfect correlations (likely diagonal or identical neurons)
                if abs(correlation) >= 0.999:
                    continue

                # Compute p-value for this correlation
                p_value = self._correlation_p_value(correlation, n_samples)

                correlations.append(correlation)
                p_values.append(p_value)

        return np.array(correlations), np.array(p_values)

    def _correlation_p_value(self, correlation: float, n_samples: int) -> float:
        """Compute p-value for a correlation coefficient."""
        if abs(correlation) >= 1.0:
            return 0.0  # Perfect correlation is always significant

        if n_samples <= 2:
            return 1.0  # Cannot test with too few samples

        # Compute t-statistic
        t_stat = correlation * np.sqrt((n_samples - 2) / (1 - correlation**2))

        # Compute p-value
        if self.two_tailed:
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_samples - 2))
        else:
            p_value = 1 - stats.t.cdf(t_stat, df=n_samples - 2)

        return p_value

    def _estimate_sample_size_from_matrix(self, correlation_matrix: np.ndarray) -> int:
        """Estimate sample size from correlation matrix properties."""
        upper_triangle = []
        n = correlation_matrix.shape[0]

        for i in range(n):
            for j in range(i + 1, n):
                if abs(correlation_matrix[i, j]) < 0.999:  # Exclude perfect correlations
                    upper_triangle.append(abs(correlation_matrix[i, j]))

        if not upper_triangle:
            return 30  # Default reasonable sample size

        # Use the fact that correlation precision increases with sample size
        # Smaller typical correlations suggest larger sample size
        median_correlation = np.median(upper_triangle)

        # Heuristic mapping (this could be improved with domain knowledge)
        if median_correlation < 0.1:
            estimated_n = 100  # Large sample, small typical correlations
        elif median_correlation < 0.2:
            estimated_n = 50
        elif median_correlation < 0.3:
            estimated_n = 30
        else:
            estimated_n = 20  # Smaller sample, larger typical correlations

        logger.debug(f"Estimated sample size: {estimated_n} (based on median |r|: {median_correlation:.3f})")
        return estimated_n

    def _estimate_sample_size_from_threshold(self, critical_t: float) -> int:
        """Estimate sample size from critical t-value."""
        # For two-tailed test at alpha=0.05, find df such that t(alpha/2, df) = critical_t
        if critical_t <= 0:
            return 30  # Default

        # Binary search for degrees of freedom
        for df in range(2, 1000):
            if self.two_tailed:
                theoretical_t = stats.t.ppf(1 - self.significance_level / 2, df)
            else:
                theoretical_t = stats.t.ppf(1 - self.significance_level, df)

            if abs(theoretical_t - critical_t) < 0.01:
                return df + 2  # df = n - 2, so n = df + 2

        return 30  # Default if not found

    def _apply_multiple_testing_correction(self, p_values: np.ndarray) -> dict[str, t.Any]:
        """Apply multiple testing correction to p-values."""
        if len(p_values) == 0:
            return {
                "significant_mask": np.array([], dtype=bool),
                "corrected_p_values": np.array([]),
                "n_tests": 0,
                "n_significant": 0,
            }

        if self.correction_method == "none":
            significant_mask = p_values < self.significance_level
            corrected_p_values = p_values
        else:
            # Use statsmodels for multiple testing correction
            significant_mask, corrected_p_values, _, _ = multipletests(
                p_values, alpha=self.significance_level, method=self.correction_method
            )

        n_significant = np.sum(significant_mask)

        logger.info(
            f"Multiple testing correction ({self.correction_method}): "
            f"{n_significant}/{len(p_values)} correlations significant"
        )

        return {
            "significant_mask": significant_mask,
            "corrected_p_values": corrected_p_values,
            "n_tests": len(p_values),
            "n_significant": n_significant,
        }

    def _determine_threshold_value(self, correlations: np.ndarray, significant_mask: np.ndarray) -> float:
        """Determine the correlation threshold based on significant correlations."""
        if len(correlations) == 0 or not np.any(significant_mask):
            return 1.0  # No significant correlations, impossible threshold

        significant_correlations = correlations[significant_mask]

        # The threshold is the minimum absolute correlation that is significant
        # This ensures we include all statistically significant relationships
        if len(significant_correlations) > 0:
            threshold_value = np.min(np.abs(significant_correlations))
        else:
            threshold_value = 1.0  # No significant correlations

        return threshold_value

    def get_threshold_info(self, correlation_matrix: np.ndarray) -> dict[str, t.Any]:
        """Get detailed information about threshold selection (for debugging/reporting).

        Args:
            correlation_matrix: Square correlation matrix

        Returns:
            Dictionary with detailed threshold information

        """
        n_neurons = correlation_matrix.shape[0]

        if n_neurons < 3:
            return {
                "threshold_value": 1.0,
                "n_significant_edges": 0,
                "n_positive_edges": 0,
                "n_negative_edges": 0,
                "significance_level": self.significance_level,
                "correction_method": self.correction_method,
                "description": f"Trivial case: {n_neurons} neurons, no statistical testing possible",
            }

        # Extract correlations and p-values
        correlations, p_values = self._extract_correlations_and_pvalues(correlation_matrix)

        if len(correlations) == 0:
            return {
                "threshold_value": 1.0,
                "n_significant_edges": 0,
                "n_positive_edges": 0,
                "n_negative_edges": 0,
                "significance_level": self.significance_level,
                "correction_method": self.correction_method,
                "description": "No correlations to test",
            }

        # Apply multiple testing correction
        corrected_results = self._apply_multiple_testing_correction(p_values)
        significant_mask = corrected_results["significant_mask"]

        # Apply effect size filter if specified
        if self.min_effect_size is not None:
            effect_size_mask = np.abs(correlations) >= self.min_effect_size
            significant_mask = significant_mask & effect_size_mask

        # Get threshold and statistics
        threshold_value = self._determine_threshold_value(correlations, significant_mask)

        if np.any(significant_mask):
            significant_correlations = correlations[significant_mask]
            n_positive = np.sum(significant_correlations > 0)
            n_negative = np.sum(significant_correlations < 0)
            n_significant_total = len(significant_correlations)
        else:
            n_positive = 0
            n_negative = 0
            n_significant_total = 0

        # Generate description
        correction_desc = (
            f"{self.correction_method} correction" if self.correction_method != "none" else "no correction"
        )
        effect_size_desc = f", min |r|≥{self.min_effect_size}" if self.min_effect_size else ""

        description = (
            f"Statistical threshold: α={self.significance_level} "
            f"({correction_desc}){effect_size_desc}, "
            f"min significant |r|={threshold_value:.4f}"
        )

        return {
            "threshold_value": threshold_value,
            "n_significant_edges": n_significant_total,
            "n_positive_edges": n_positive,
            "n_negative_edges": n_negative,
            "significance_level": self.significance_level,
            "correction_method": self.correction_method,
            "description": description,
            "total_tests": len(correlations),
            "significant_ratio": n_significant_total / len(correlations) if len(correlations) > 0 else 0.0,
        }
