from pathlib import Path

from neuron_analyzer.load_util import JsonProcessor


class NeuronHypothesisTester:
    """Class for testing the hyperplane separability hypothesis on neuron data."""

    def __init__(self, classifier_results, out_path: Path):
        """Initialize the hypothesis tester."""
        self.results = classifier_results
        self.out_path = out_path
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

    def run_pipeline(self) -> dict:
        """Summarize the evidence for hyperplane separation of neuron groups."""
        # Check if we have all the necessary results
        required_keys = [
            "linear_svc",
            "cv_svm_linear",
            "perm_test_linear_svc",
            "margin_stats_linear_svc",
            "misclass_analysis_linear_svc",
        ]

        for key in required_keys:
            if key not in self.results:
                raise ValueError(f"Missing required result '{key}'. Run the full analysis first.")

        # Extract key metrics
        accuracy = self.results["linear_svc"]["accuracy"]
        f1_score = self.results["linear_svc"]["f1_score"]
        cv_accuracy = self.results["cv_svm_linear"]["mean_accuracy"]
        cv_std = self.results["cv_svm_linear"]["std_accuracy"]
        perm_pvalue = self.results["perm_test_linear_svc"]["p_value"]

        # Margin statistics
        margin_stats = self.results["margin_stats_linear_svc"]
        mean_margin = margin_stats["mean_distance"]
        class_margins = margin_stats.get("class_margins", {})

        # Misclassification analysis
        misclass_rate = self.results["misclass_analysis_linear_svc"]["misclassification_rate"]

        # Interpret the evidence
        strong_evidence_threshold = 0.85  # Accuracy/F1 threshold for strong evidence
        moderate_evidence_threshold = 0.70  # For moderate evidence
        significant_pvalue = 0.05  # Standard significance level

        # Determine evidence level
        if accuracy > strong_evidence_threshold and perm_pvalue < significant_pvalue:
            evidence_level = "Strong"
            interpretation = (
                "The classifier achieves high accuracy, significantly better than random labeling, "
                "indicating that neuron groups are likely separable by hyperplanes."
            )
        elif accuracy > moderate_evidence_threshold and perm_pvalue < significant_pvalue:
            evidence_level = "Moderate"
            interpretation = (
                "The classifier achieves moderate accuracy, significantly better than random labeling, "
                "suggesting that neuron groups may be partially separable by hyperplanes."
            )
        elif perm_pvalue < significant_pvalue:
            evidence_level = "Weak"
            interpretation = (
                "The classifier performs significantly better than random labeling, but with "
                "relatively low accuracy, indicating limited separability by hyperplanes."
            )
        else:
            evidence_level = "Insufficient"
            interpretation = (
                "The classifier does not perform significantly better than random labeling, "
                "suggesting that neuron groups may not be separable by hyperplanes."
            )

        # Summary dictionary
        summary = {
            "evidence_level": evidence_level,
            "interpretation": interpretation,
            "key_metrics": {
                "accuracy": accuracy,
                "f1_score": f1_score,
                "cross_validation": {"mean_accuracy": cv_accuracy, "std_accuracy": cv_std},
                "permutation_test_pvalue": perm_pvalue,
                "misclassification_rate": misclass_rate,
                "margin_statistics": {"mean_margin": mean_margin, "class_specific_margins": class_margins},
            },
            "conclusion": self._generate_conclusion(evidence_level, accuracy, perm_pvalue, mean_margin),
        }

        # Save to file
        JsonProcessor.save_json(summary, self.out_path)
        return summary

    def _generate_conclusion(self, evidence_level, accuracy, pvalue, margin):
        """Generate a conclusion based on the evidence level and key metrics."""
        if evidence_level == "Strong":
            return (
                f"With an accuracy of {accuracy:.2f} and p-value of {pvalue:.4f}, "
                f"we find strong evidence supporting the hyperplane separability hypothesis. "
                f"The mean margin of {margin:.4f} suggests a clear separation between neuron groups."
            )
        if evidence_level == "Moderate":
            return (
                f"With an accuracy of {accuracy:.2f} and p-value of {pvalue:.4f}, "
                f"we find moderate evidence supporting the hyperplane separability hypothesis. "
                f"The mean margin of {margin:.4f} suggests some separation between neuron groups, "
                f"but there may be overlap regions or outliers."
            )
        if evidence_level == "Weak":
            return (
                f"With an accuracy of {accuracy:.2f} and p-value of {pvalue:.4f}, "
                f"we find weak evidence supporting the hyperplane separability hypothesis. "
                f"The mean margin of {margin:.4f} suggests limited separation between neuron groups, "
                f"with substantial overlap or complex boundaries."
            )
        return (
            f"With an accuracy of {accuracy:.2f} and p-value of {pvalue:.4f}, "
            f"we find insufficient evidence supporting the hyperplane separability hypothesis. "
            f"The results suggest that neuron groups may not be linearly separable in activation space."
        )
