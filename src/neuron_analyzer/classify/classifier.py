import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, permutation_test_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NeuronClassifier:
    """Class for classifying neurons based on their feature vectors."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        neuron_indices: list[str],
        metadata: dict | None = None,
        classification_mode: str = "three_class",
        random_state: int = 42,
    ):
        """Initialize the NeuronClassifier."""
        self.X = X
        self.y_original = y
        self.neuron_indices = neuron_indices
        self.metadata = metadata or {}
        self.classification_mode = classification_mode
        self.random_state = random_state

        # Transform labels if using two-class mode
        if classification_mode == "binary":
            # Convert to binary classification: 0 for common, 1 for special (boost or suppress)
            self.y = np.array([0 if label == 0 else 1 for label in y])
        else:
            self.y = y

        # Classifier containers
        self.classifiers = {}
        self.results = {}
        self.feature_importance = {}
        self.hyperplanes = {}

    def prepare_data(self, test_size: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and testing sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.random_state, stratify=self.y
        )

        return X_train, X_test, y_train, y_test

    def train_svm(self, kernel: str = "linear", C: float = 1.0, gamma: str = "scale", test_size: float = 0.2) -> dict:
        """Train an SVM classifier and evaluate its performance."""
        # TODO: check parameter setting
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(test_size)

        # Initialize and train the SVM classifier
        clf = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=self.random_state)
        clf.fit(X_train, y_train)

        # Make predictions
        y_pred = clf.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(y_test, y_pred, output_dict=True)

        # Calculate silhouette score if there are enough samples
        silhouette = None
        if len(np.unique(y_test)) > 1:
            try:
                silhouette = silhouette_score(X_test, y_pred)
            except Exception:
                silhouette = None

        # Store model and results
        model_name = f"svm_{kernel}"
        self.classifiers[model_name] = clf

        # Extract hyperplane for linear SVM
        if kernel == "linear":
            self.hyperplanes[model_name] = {
                "w": clf.coef_[0] if len(clf.coef_) == 1 else clf.coef_,
                "b": clf.intercept_[0] if len(clf.intercept_) == 1 else clf.intercept_,
                "support_vectors": clf.support_vectors_,
            }

        # Store results
        results = {
            "accuracy": accuracy,
            "f1_score": f1,
            "classification_report": report,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "silhouette_score": silhouette,
            "model_params": {"kernel": kernel, "C": C, "gamma": gamma},
        }

        self.results[model_name] = results
        return results

    def train_linear_svc(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        loss: str = "squared_hinge",
        dual: bool = True,
        test_size: float = 0.2,
    ) -> dict:
        """Train a LinearSVC classifier (optimized for linear kernel)."""
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(test_size)

        # Initialize and train the LinearSVC classifier
        clf = LinearSVC(
            C=C,
            penalty=penalty,
            loss=loss,
            dual=dual,
            random_state=self.random_state,
            max_iter=10000,  # Increased to ensure convergence
        )
        clf.fit(X_train, y_train)

        # Make predictions
        y_pred = clf.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(y_test, y_pred, output_dict=True)

        # Calculate silhouette score if possible
        silhouette = None
        if len(np.unique(y_test)) > 1:
            try:
                silhouette = silhouette_score(X_test, y_pred)
            except Exception:
                silhouette = None

        # Store model and results
        model_name = "linear_svc"
        self.classifiers[model_name] = clf

        # Extract hyperplane
        self.hyperplanes[model_name] = {
            "w": clf.coef_[0] if len(clf.coef_) == 1 else clf.coef_,
            "b": clf.intercept_[0] if len(clf.intercept_) == 1 else clf.intercept_,
        }

        # Store results
        results = {
            "accuracy": accuracy,
            "f1_score": f1,
            "classification_report": report,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "silhouette_score": silhouette,
            "model_params": {"C": C, "penalty": penalty, "loss": loss, "dual": dual},
        }

        self.results[model_name] = results
        return results

    def train_comparison_classifiers(self, test_size: float = 0.2) -> dict:
        """Train additional classifiers to verify the hyperplane."""
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(test_size)

        # Initialize classifiers
        classifiers = {
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            "mlp": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=self.random_state),
        }

        comparison_results = {}

        # Train and evaluate each classifier
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # Store the model
            self.classifiers[name] = clf

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            report = classification_report(y_test, y_pred, output_dict=True)

            # Calculate silhouette score if possible
            silhouette = None
            if len(np.unique(y_test)) > 1:
                try:
                    silhouette = silhouette_score(X_test, y_pred)
                except Exception:
                    silhouette = None

            # Calculate feature importance for Random Forest
            if name == "random_forest":
                self.feature_importance[name] = clf.feature_importances_

            # Store results
            results = {
                "accuracy": accuracy,
                "f1_score": f1,
                "classification_report": report,
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "silhouette_score": silhouette,
            }

            self.results[name] = results
            comparison_results[name] = results

        return comparison_results

    def cross_validate_svm(self, kernel: str = "linear", C: float = 1.0, n_splits: int = 5) -> dict:
        """Perform cross-validation to check hyperplane robustness."""
        # Initialize the classifier
        clf = SVC(kernel=kernel, C=C, random_state=self.random_state)

        # Initialize k-fold cross-validation
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        # Lists to store results
        accuracies = []
        f1_scores = []

        # Perform cross-validation
        for train_idx, test_idx in kfold.split(self.X, self.y):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            # Train the model
            clf.fit(X_train, y_train)

            # Make predictions
            y_pred = clf.predict(X_test)

            # Calculate metrics
            accuracies.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred, average="weighted"))

        # Calculate mean and std of metrics
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)

        # Store cross-validation results
        cv_results = {
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy,
            "mean_f1": mean_f1,
            "std_f1": std_f1,
            "fold_accuracies": accuracies,
            "fold_f1_scores": f1_scores,
        }

        model_name = f"cv_svm_{kernel}"
        self.results[model_name] = cv_results

        return cv_results

    def perform_permutation_test(
        self, classifier_type: str = "linear_svc", n_permutations: int = 1000, test_size: float = 0.2
    ) -> dict:
        """Perform permutation test to validate statistical significance of hyperplanes."""
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(test_size)

        # Initialize the classifier
        if classifier_type == "linear_svc":
            clf = LinearSVC(random_state=self.random_state, max_iter=10000)
        else:
            clf = SVC(kernel="linear", random_state=self.random_state)

        # Perform permutation test
        score, perm_scores, pvalue = permutation_test_score(
            clf,
            X_train,
            y_train,
            scoring="accuracy",
            cv=5,
            n_permutations=n_permutations,
            random_state=self.random_state,
        )

        # Store permutation test results
        perm_test_results = {
            "score": score,
            "perm_scores_mean": np.mean(perm_scores),
            "perm_scores_std": np.std(perm_scores),
            "p_value": pvalue,
            "n_permutations": n_permutations,
        }

        model_name = f"perm_test_{classifier_type}"
        self.results[model_name] = perm_test_results

        return perm_test_results

    def calculate_margin_statistics(self, clf_name: str = "linear_svc") -> dict:
        """Calculate statistics about the margin distribution."""
        if clf_name not in self.classifiers:
            raise ValueError(f"Classifier '{clf_name}' not found. Train it first.")

        clf = self.classifiers[clf_name]

        # For linear SVM, calculate distance to hyperplane
        distances = []
        if clf_name == "linear_svc" or (clf_name == "svm_linear"):
            # Get w and b from the hyperplane
            w = self.hyperplanes[clf_name]["w"]
            b = self.hyperplanes[clf_name]["b"]

            # Calculate distances for each sample
            w_norm = np.linalg.norm(w)

            # Handle multi-class case
            if len(w.shape) > 1 and w.shape[0] > 1:
                # For multiclass, we'll calculate distance to the closest hyperplane
                all_distances = []

                for i in range(w.shape[0]):
                    w_i = w[i]
                    b_i = b[i] if isinstance(b, np.ndarray) else b
                    dist_i = (np.dot(self.X, w_i) + b_i) / np.linalg.norm(w_i)
                    all_distances.append(dist_i)

                # Stack distances and get the minimum absolute distance for each sample
                all_distances = np.column_stack(all_distances)
                distances = np.min(np.abs(all_distances), axis=1)
            else:
                # For binary classification
                distances = np.abs(np.dot(self.X, w) + b) / w_norm
        # For non-linear SVM with decision_function method
        elif hasattr(clf, "decision_function"):
            # Get raw decision function values
            decision_values = clf.decision_function(self.X)

            # For multi-class, take the minimum absolute distance to any hyperplane
            if len(decision_values.shape) > 1:
                distances = np.min(np.abs(decision_values), axis=1)
            else:
                distances = np.abs(decision_values)
        # For other classifiers, use probability as a proxy for confidence
        elif hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(self.X)
            distances = np.max(proba, axis=1)  # Use max probability as confidence

        # Calculate statistics
        margin_stats = {
            "mean_distance": np.mean(distances),
            "median_distance": np.median(distances),
            "std_distance": np.std(distances),
            "min_distance": np.min(distances),
            "max_distance": np.max(distances),
        }

        # Store margin statistics in results
        margin_key = f"margin_stats_{clf_name}"
        self.results[margin_key] = margin_stats

        # Create DataFrame with neuron indices, classes, and distances
        margin_df = pd.DataFrame(
            {
                "neuron_id": self.neuron_indices,
                "class": self.y_original,  # Original class labels
                "distance": distances,
            }
        )

        # Group by class and calculate statistics
        class_margins = margin_df.groupby("class")["distance"].agg(["mean", "median", "std", "min", "max"]).to_dict()

        # Add class-specific margin statistics
        self.results[margin_key]["class_margins"] = class_margins

        return margin_stats

    def bootstrap_confidence_intervals(
        self, clf_name: str = "linear_svc", n_bootstraps: int = 1000, confidence_level: float = 0.95
    ) -> dict:
        """Calculate bootstrap confidence intervals for classifier metrics."""
        if clf_name not in self.classifiers:
            raise ValueError(f"Classifier '{clf_name}' not found. Train it first.")

        clf = self.classifiers[clf_name]

        # Lists to store bootstrap metrics
        accuracies = []
        f1_scores = []

        # Perform bootstrap sampling
        for _ in range(n_bootstraps):
            # Sample with replacement
            indices = np.random.choice(range(len(self.X)), size=len(self.X), replace=True)
            X_bootstrap = self.X[indices]
            y_bootstrap = self.y[indices]

            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X_bootstrap, y_bootstrap, test_size=0.2, random_state=np.random.randint(1000)
            )

            # Clone and train the classifier
            clf_bootstrap = clone_from_sklearn_estimator(clf)
            clf_bootstrap.fit(X_train, y_train)

            # Make predictions
            y_pred = clf_bootstrap.predict(X_test)

            # Calculate metrics
            accuracies.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred, average="weighted"))

        # Calculate confidence intervals
        alpha = (1 - confidence_level) / 2

        # For accuracy
        acc_lower = np.percentile(accuracies, 100 * alpha)
        acc_upper = np.percentile(accuracies, 100 * (1 - alpha))
        acc_mean = np.mean(accuracies)

        # For F1-score
        f1_lower = np.percentile(f1_scores, 100 * alpha)
        f1_upper = np.percentile(f1_scores, 100 * (1 - alpha))
        f1_mean = np.mean(f1_scores)

        # Store bootstrap results
        bootstrap_results = {
            "accuracy_ci": {"lower": acc_lower, "mean": acc_mean, "upper": acc_upper},
            "f1_score_ci": {"lower": f1_lower, "mean": f1_mean, "upper": f1_upper},
            "n_bootstraps": n_bootstraps,
            "confidence_level": confidence_level,
        }

        bootstrap_key = f"bootstrap_{clf_name}"
        self.results[bootstrap_key] = bootstrap_results

        return bootstrap_results

    def analyze_misclassifications(self, clf_name: str = "linear_svc", test_size: float = 0.2) -> dict:
        """Analyze misclassified samples to identify potential subclusters or outliers."""
        if clf_name not in self.classifiers:
            raise ValueError(f"Classifier '{clf_name}' not found. Train it first.")

        clf = self.classifiers[clf_name]

        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(test_size)

        # Make predictions
        y_pred = clf.predict(X_test)

        # Find misclassified samples
        misclassified_indices = np.where(y_test != y_pred)[0]
        correctly_classified_indices = np.where(y_test == y_pred)[0]

        # Get the original indices
        indices_in_test = np.arange(len(self.X))[len(X_train) :]
        misclassified_original_indices = indices_in_test[misclassified_indices]

        # Get the corresponding neuron indices
        misclassified_neuron_indices = [self.neuron_indices[i] for i in misclassified_original_indices]

        # Analyze misclassifications by class
        true_vs_pred = {}
        for i in misclassified_indices:
            true_class = int(y_test[i])
            pred_class = int(y_pred[i])
            key = f"{true_class}_{pred_class}"
            if key not in true_vs_pred:
                true_vs_pred[key] = 0
            true_vs_pred[key] += 1

        # Create misclassification summary
        misclass_analysis = {
            "total_test_samples": len(y_test),
            "misclassified_count": len(misclassified_indices),
            "misclassification_rate": len(misclassified_indices) / len(y_test),
            "class_confusion": true_vs_pred,
            "misclassified_neuron_indices": misclassified_neuron_indices,
        }

        misclass_key = f"misclass_analysis_{clf_name}"
        self.results[misclass_key] = misclass_analysis

        return misclass_analysis

    def run_all_analyses(self, test_size: float = 0.2) -> dict:
        """Run all classification and analysis methods."""
        # Train classifiers
        logger.info("Training Linear SVC...")
        self.train_linear_svc(test_size=test_size)

        logger.info("Training SVM with linear kernel...")
        self.train_svm(kernel="linear", test_size=test_size)

        logger.info("Training comparison classifiers...")
        self.train_comparison_classifiers(test_size=test_size)

        logger.info("Performing cross-validation...")
        self.cross_validate_svm(kernel="linear", n_splits=5)

        logger.info("Performing permutation test...")
        self.perform_permutation_test(n_permutations=100)  # Use a smaller number for speed

        logger.info("Calculating margin statistics...")
        self.calculate_margin_statistics(clf_name="linear_svc")

        logger.info("Analyzing misclassifications...")
        self.analyze_misclassifications(clf_name="linear_svc", test_size=test_size)

        # Compile summary of key results
        summary = {
            "classification_mode": self.classification_mode,
            "sample_counts": {
                "total": len(self.y),
                "class_distribution": {str(c): int(np.sum(self.y == c)) for c in np.unique(self.y)},
            },
            "linear_svc": {
                "accuracy": self.results["linear_svc"]["accuracy"],
                "f1_score": self.results["linear_svc"]["f1_score"],
                "silhouette_score": self.results["linear_svc"]["silhouette_score"],
            },
            "svm_linear": {
                "accuracy": self.results["svm_linear"]["accuracy"],
                "f1_score": self.results["svm_linear"]["f1_score"],
                "silhouette_score": self.results["svm_linear"]["silhouette_score"],
            },
            "cross_validation": {
                "mean_accuracy": self.results["cv_svm_linear"]["mean_accuracy"],
                "std_accuracy": self.results["cv_svm_linear"]["std_accuracy"],
            },
            "permutation_test": {"p_value": self.results["perm_test_linear_svc"]["p_value"]},
            "margin_statistics": {
                "mean_distance": self.results["margin_stats_linear_svc"]["mean_distance"],
                "class_margins": self.results["margin_stats_linear_svc"]["class_margins"],
            },
            "misclassification_rate": self.results["misclass_analysis_linear_svc"]["misclassification_rate"],
        }

        self.results["summary"] = summary
        return summary

    def save_results(self, output_path: Path) -> None:
        """Save all results and models to disk."""
        # Create output directory
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save results JSON
        results_copy = {k: v for k, v in self.results.items()}

        # Convert numpy arrays in hyperplanes to lists for JSON serialization
        for model_name, hyperplane in self.hyperplanes.items():
            for key, value in hyperplane.items():
                if isinstance(value, np.ndarray):
                    results_copy.setdefault("hyperplanes", {}).setdefault(model_name, {})[key] = value.tolist()

        # Add metadata
        results_copy["metadata"] = {
            "classification_mode": self.classification_mode,
            "feature_dim": self.X.shape[1],
            "num_samples": self.X.shape[0],
            "timestamp": datetime.now().isoformat(),
            "original_metadata": self.metadata,
        }

        # Save results as JSON
        with open(output_path / "classification_results.json", "w") as f:
            json.dump(results_copy, f, indent=2)

        # Save models
        for name, clf in self.classifiers.items():
            joblib.dump(clf, output_path / f"{name}_model.joblib")

        logger.info(f"Results and models saved to {output_path}")
