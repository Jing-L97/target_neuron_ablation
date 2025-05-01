import logging
from pathlib import Path

import numpy as np

from neuron_analyzer.load_util import JsonProcessor

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NeuronHypothesisTester:
    """Class for testing the hyperplane separability hypothesis on neuron data."""

    def __init__(self, classifier_results: dict, out_path: Path, resume: bool = False):
        """Initialize the hypothesis tester."""
        self.results = classifier_results
        self.out_path = out_path
        self.resume = resume
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

    def run_pipeline(self) -> dict:
        """Summarize the evidence for hyperplane separation of neuron groups."""
        if self.resume and self.out_path.is_file():
            # load and return the file
            logger.info(f"Resume existing file from {self.out_path}")
            return JsonProcessor.load_json(self.out_path)

        # Check if we have all the necessary results
        required_keys = [
            "linear_svc",
            "svm_linear",
            "cross_validation",
            "permutation_test",
            "margin_statistics",
            "misclassification_rate",
        ]

        for key in required_keys:
            if key not in self.results:
                raise ValueError(f"Missing required result '{key}'. Run the full analysis first.")

        # Extract key metrics from linear_svc
        linear_svc_accuracy = self.results["linear_svc"]["accuracy"]
        linear_svc_f1 = self.results["linear_svc"]["f1_score"]
        linear_svc_silhouette = self.results["linear_svc"].get("silhouette_score")

        # Extract key metrics from svm_linear
        svm_linear_accuracy = self.results["svm_linear"]["accuracy"]
        svm_linear_f1 = self.results["svm_linear"]["f1_score"]
        svm_linear_silhouette = self.results["svm_linear"].get("silhouette_score")

        # Extract cross-validation metrics
        cv_accuracy = self.results["cross_validation"]["mean_accuracy"]
        cv_std = self.results["cross_validation"]["std_accuracy"]

        # Extract permutation test p-value
        perm_pvalue = self.results["permutation_test"]["p_value"]

        # Extract margin statistics
        margin_stats = self.results["margin_statistics"]
        class_margins = margin_stats.get("class_margins", {})

        # Handle missing mean_distance
        if "mean_distance" in margin_stats:
            mean_margin = margin_stats["mean_distance"]
        else:
            # Try to calculate from class margins
            mean_values = []
            for cls, stats in class_margins.items():
                if isinstance(stats, dict) and "mean" in stats:
                    mean_values.append(stats["mean"])

            mean_margin = np.mean(mean_values) if mean_values else 0.0

        # Extract misclassification rate
        misclass_rate = self.results["misclassification_rate"]

        # Calculate average performance metrics for more robust evaluation
        avg_accuracy = (linear_svc_accuracy + svm_linear_accuracy) / 2
        avg_f1 = (linear_svc_f1 + svm_linear_f1) / 2

        # Define thresholds for evidence levels
        strong_evidence_threshold = 0.85
        moderate_evidence_threshold = 0.70
        significant_pvalue = 0.05

        # Determine evidence level
        if avg_accuracy > strong_evidence_threshold and perm_pvalue < significant_pvalue:
            evidence_level = "Strong"
            interpretation = (
                "Both classifiers achieve high accuracy, significantly better than random labeling, "
                "indicating that neuron groups are likely separable by hyperplanes."
            )
        elif avg_accuracy > moderate_evidence_threshold and perm_pvalue < significant_pvalue:
            evidence_level = "Moderate"
            interpretation = (
                "Both classifiers achieve moderate accuracy, significantly better than random labeling, "
                "suggesting that neuron groups may be partially separable by hyperplanes."
            )
        elif perm_pvalue < significant_pvalue:
            evidence_level = "Weak"
            interpretation = (
                "The classifiers perform significantly better than random labeling, but with "
                "relatively low accuracy, indicating limited separability by hyperplanes."
            )
        else:
            evidence_level = "Insufficient"
            interpretation = (
                "The classifiers do not perform significantly better than random labeling, "
                "suggesting that neuron groups may not be separable by hyperplanes."
            )

        # Create summary dictionary with comprehensive metrics
        summary = {
            "evidence_level": evidence_level,
            "interpretation": interpretation,
            "key_metrics": {
                "linear_svc": {
                    "accuracy": linear_svc_accuracy,
                    "f1_score": linear_svc_f1,
                    "silhouette_score": linear_svc_silhouette,
                },
                "svm_linear": {
                    "accuracy": svm_linear_accuracy,
                    "f1_score": svm_linear_f1,
                    "silhouette_score": svm_linear_silhouette,
                },
                "average_performance": {
                    "accuracy": avg_accuracy,
                    "f1_score": avg_f1,
                },
                "cross_validation": {
                    "mean_accuracy": cv_accuracy,
                    "std_accuracy": cv_std,
                },
                "permutation_test": {
                    "p_value": perm_pvalue,
                },
                "margin_statistics": {
                    "mean_margin": mean_margin,
                    "class_margins": class_margins,
                },
                "misclassification_rate": misclass_rate,
            },
            "conclusion": self._generate_conclusion(
                evidence_level, avg_accuracy, perm_pvalue, mean_margin, linear_svc_accuracy, svm_linear_accuracy
            ),
            "class_distribution": self.results.get("sample_counts", {}).get("class_distribution", {}),
            "total_samples": self.results.get("sample_counts", {}).get("total", 0),
            "class_num": self.results.get("class_num", 2),
        }

        # Save to file
        JsonProcessor.save_json(summary, self.out_path)
        logger.info(f"Saved hyperplane separability analysis to {self.out_path}")
        return summary

    def _generate_conclusion(
        self,
        evidence_level: str,
        avg_accuracy: float,
        pvalue: float,
        margin: float,
        linear_svc_accuracy: float,
        svm_linear_accuracy: float,
    ) -> str:
        """Generate a detailed conclusion based on the evidence level and key metrics."""
        if evidence_level == "Strong":
            return (
                f"With LinearSVC accuracy of {linear_svc_accuracy:.2f}, SVM accuracy of {svm_linear_accuracy:.2f} "
                f"(average: {avg_accuracy:.2f}), and permutation test p-value of {pvalue:.4f}, "
                f"we find strong evidence supporting the hyperplane separability hypothesis. "
                f"The mean margin of {margin:.4f} suggests a clear separation between neuron groups."
            )
        if evidence_level == "Moderate":
            return (
                f"With LinearSVC accuracy of {linear_svc_accuracy:.2f}, SVM accuracy of {svm_linear_accuracy:.2f} "
                f"(average: {avg_accuracy:.2f}), and permutation test p-value of {pvalue:.4f}, "
                f"we find moderate evidence supporting the hyperplane separability hypothesis. "
                f"The mean margin of {margin:.4f} suggests some separation between neuron groups, "
                f"but there may be overlap regions or outliers."
            )
        if evidence_level == "Weak":
            return (
                f"With LinearSVC accuracy of {linear_svc_accuracy:.2f}, SVM accuracy of {svm_linear_accuracy:.2f} "
                f"(average: {avg_accuracy:.2f}), and permutation test p-value of {pvalue:.4f}, "
                f"we find weak evidence supporting the hyperplane separability hypothesis. "
                f"The mean margin of {margin:.4f} suggests limited separation between neuron groups, "
                f"with substantial overlap or complex boundaries."
            )
        return (
            f"With LinearSVC accuracy of {linear_svc_accuracy:.2f}, SVM accuracy of {svm_linear_accuracy:.2f} "
            f"(average: {avg_accuracy:.2f}), and permutation test p-value of {pvalue:.4f}, "
            f"we find insufficient evidence supporting the hyperplane separability hypothesis. "
            f"The results suggest that neuron groups may not be linearly separable in activation space."
        )
