import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, permutation_test_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC

from neuron_analyzer.load_util import JsonProcessor

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def clone_from_sklearn_estimator(estimator):
    """Clone a scikit-learn estimator for the bootstrap confidence intervals method."""
    try:
        return clone(estimator)
    except:
        # If clone doesn't work, try to initialize with the same parameters
        if isinstance(estimator, SVC):
            return SVC(
                C=estimator.C,
                kernel=estimator.kernel,
                degree=estimator.degree,
                gamma=estimator.gamma,
                coef0=estimator.coef0,
                random_state=np.random.randint(1000),
            )
        if isinstance(estimator, LinearSVC):
            return LinearSVC(
                C=estimator.C,
                penalty=estimator.penalty,
                loss=estimator.loss,
                dual=estimator.dual,
                random_state=np.random.randint(1000),
                max_iter=10000,
            )
        raise ValueError(f"Cannot clone estimator of type {type(estimator)}")


class NeuronClassifier:
    """Class for classifying neurons based on their feature vectors."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        neuron_indices: list[str],
        model_path: Path,
        eval_path: Path,
        metadata: dict | None = None,
        class_num: int = 2,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """Initialize the NeuronClassifier."""
        self.X = X
        self.y_original = y
        self.neuron_indices = neuron_indices
        self.metadata = metadata or {}
        self.class_num = class_num
        self.random_state = random_state
        self.test_size = test_size
        self.model_path = model_path
        self.eval_path = eval_path
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.eval_path.mkdir(parents=True, exist_ok=True)
        # Transform labels if using two-class mode
        if class_num == 2:
            # Convert to binary classification: 0 for common, 1 for special (boost or suppress)
            self.y = np.array([0 if label == 0 else 1 for label in y])
        else:
            self.y = y

        # Classifier containers
        self.classifiers = {}
        self.results = {}
        self.feature_importance = {}
        self.hyperplanes = {}
        # Record class distribution
        self.class_distribution = {str(c): int(np.sum(self.y == c)) for c in np.unique(self.y)}
        self.class_weights = {
            str(c): len(self.y) / (len(np.unique(self.y)) * np.sum(self.y == c)) for c in np.unique(self.y)
        }
        # initialize data split
        self.X_train, self.X_test, self.y_train, self.y_test = self.prepare_data()

    def prepare_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and testing sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state, stratify=self.y
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_svm(self, kernel: str = "linear", C: float = 1.0, gamma: str = "scale") -> dict:
        """Train an SVM classifier and evaluate its performance."""
        # Initialize and train the SVM classifier
        clf = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=self.random_state)
        clf.fit(self.X_train, self.y_train)
        # Make predictions
        y_pred = clf.predict(self.X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average="weighted", zero_division=0)
        report = classification_report(self.y_test, y_pred, output_dict=True, zero_division=0)

        # Calculate silhouette score if there are enough samples
        silhouette = None
        if len(np.unique(self.y_test)) > 1:
            try:
                silhouette = silhouette_score(self.X_test, y_pred)
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
            "confusion_matrix": confusion_matrix(self.y_test, y_pred).tolist(),
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
        class_weight: str = "balanced",  # use weighted loss for class imbalance
    ) -> dict:
        """Train a LinearSVC classifier (optimized for linear kernel)."""
        # Initialize and train the LinearSVC classifier
        clf = LinearSVC(
            C=C,
            penalty=penalty,
            loss=loss,
            dual=dual,
            class_weight=class_weight,
            random_state=self.random_state,
            max_iter=10000,  # Increased to ensure convergence
        )
        clf.fit(self.X_train, self.y_train)

        # Make predictions
        y_pred = clf.predict(self.X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average="weighted", zero_division=0)
        report = classification_report(self.y_test, y_pred, output_dict=True, zero_division=0)

        # Calculate silhouette score if possible
        silhouette = None
        if len(np.unique(self.y_test)) > 1:
            try:
                silhouette = silhouette_score(self.X_test, y_pred)
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
            "confusion_matrix": confusion_matrix(self.y_test, y_pred).tolist(),
            "silhouette_score": silhouette,
            "model_params": {"C": C, "penalty": penalty, "loss": loss, "dual": dual},
        }

        self.results[model_name] = results
        return results

    def train_comparison_classifiers(self) -> dict:
        """Train additional classifiers to verify the hyperplane."""
        # Initialize classifiers
        classifiers = {
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            "mlp": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=self.random_state),
        }

        comparison_results = {}

        # Train and evaluate each classifier
        for name, clf in classifiers.items():
            clf.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_test)

            # Store the model
            self.classifiers[name] = clf

            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average="weighted", zero_division=0)
            report = classification_report(self.y_test, y_pred, output_dict=True, zero_division=0)

            # Calculate silhouette score if possible
            silhouette = None
            if len(np.unique(self.y_test)) > 1:
                try:
                    silhouette = silhouette_score(self.X_test, y_pred)
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
                "confusion_matrix": confusion_matrix(self.y_test, y_pred).tolist(),
                "silhouette_score": silhouette,
            }

            self.results[name] = results
            comparison_results[name] = results

        return comparison_results

    def cross_validate_svm(
        self,
        kernel: str = "linear",
        C: float = 1.0,
        class_weight: str = "balanced",  # Added parameter
        n_splits: int = 5,
    ) -> dict:
        """Perform cross-validation for SVM classifier."""
        # Initialize the classifier
        clf = SVC(
            kernel=kernel,
            C=C,
            class_weight=class_weight,  # Added parameter
            random_state=self.random_state,
        )

        # Initialize k-fold cross-validation
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        # Lists to store results
        accuracies = []
        f1_scores = []

        # Initialize per-class F1 scores
        class_f1_scores = {str(cls): [] for cls in np.unique(self.y)}

        # Perform cross-validation
        for train_idx, test_idx in kfold.split(self.X, self.y):
            self.X_train, self.X_test = self.X[train_idx], self.X[test_idx]
            self.y_train, self.y_test = self.y[train_idx], self.y[test_idx]

            # Train the model
            clf.fit(self.X_train, self.y_train)

            # Make predictions
            y_pred = clf.predict(self.X_test)

            # Calculate metrics
            accuracies.append(accuracy_score(self.y_test, y_pred))
            f1_scores.append(f1_score(self.y_test, y_pred, average="weighted", zero_division=0))

            # Calculate per-class F1 scores
            if len(np.unique(self.y)) <= 2:
                # Binary classification
                for i, cls in enumerate(np.unique(self.y)):
                    cls_f1 = f1_score(self.y_test, y_pred, average="binary") if i == 1 else None

                    if cls_f1 is not None:
                        class_f1_scores[str(cls)].append(cls_f1)
            else:
                # Multi-class classification
                f1s = f1_score(self.y_test, y_pred, average=None)
                for i, cls in enumerate(np.unique(self.y_test)):
                    class_f1_scores[str(cls)].append(f1s[i])

        # Calculate mean and std of metrics
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)

        # Calculate per-class F1 statistics
        class_f1_stats = {}
        for cls, scores in class_f1_scores.items():
            if scores:  # Check if we have scores for this class
                class_f1_stats[cls] = {"mean": np.mean(scores), "std": np.std(scores), "values": scores}

        # Store cross-validation results
        cv_results = {
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy,
            "mean_f1": mean_f1,
            "std_f1": std_f1,
            "fold_accuracies": accuracies,
            "fold_f1_scores": f1_scores,
            "class_f1_scores": class_f1_stats,
            "class_distribution": {str(c): int(np.sum(self.y == c)) for c in np.unique(self.y)},
        }

        model_name = f"cv_svm_{kernel}"
        self.results[model_name] = cv_results

        return cv_results

    def perform_permutation_test(self, classifier_type: str = "linear_svc", n_permutations: int = 100) -> dict:
        """Perform permutation test to validate statistical significance of hyperplanes."""
        # Initialize the classifier
        if classifier_type == "linear_svc":
            clf = LinearSVC(random_state=self.random_state, max_iter=10000)
        else:
            clf = SVC(kernel="linear", random_state=self.random_state)

        # Perform permutation test
        score, perm_scores, pvalue = permutation_test_score(
            clf,
            self.X_train,
            self.y_train,
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

        # Calculate overall statistics
        margin_stats = {
            "mean_distance": np.mean(distances),
            "median_distance": np.median(distances),
            "std_distance": np.std(distances),
            "min_distance": np.min(distances),
            "max_distance": np.max(distances),
        }

        # Create DataFrame with neuron indices, classes, and distances
        margin_df = pd.DataFrame(
            {
                "neuron_id": self.neuron_indices,
                "class": self.y_original,  # Original class labels
                "distance": distances,
            }
        )

        # Group by class and calculate statistics
        class_margins = (
            margin_df.groupby("class")["distance"]
            .agg(
                [
                    "count",  # Count samples in each class
                    "mean",
                    "median",
                    "std",
                    "min",
                    "max",
                ]
            )
            .to_dict()
        )

        # Add class proportions
        class_proportions = {}
        for cls in class_margins["count"]:
            class_proportions[str(int(cls))] = class_margins["count"][cls] / len(margin_df)

        # Add normalized margin - relative to class averages
        class_means = margin_df.groupby("class")["distance"].mean().to_dict()
        margin_df["normalized_distance"] = margin_df.apply(
            lambda row: row["distance"] / class_means[row["class"]], axis=1
        )

        # Calculate quartiles for each class
        quartiles = margin_df.groupby("class")["distance"].quantile([0.25, 0.50, 0.75]).unstack().to_dict()

        # Find outliers for each class (1.5 * IQR rule)
        class_outliers = {}
        for cls in np.unique(margin_df["class"]):
            cls_data = margin_df[margin_df["class"] == cls]
            q1 = quartiles[0.25][cls]
            q3 = quartiles[0.75][cls]
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = cls_data[(cls_data["distance"] < lower_bound) | (cls_data["distance"] > upper_bound)]
            class_outliers[str(int(cls))] = {
                "count": len(outliers),
                "percentage": len(outliers) / len(cls_data) * 100,
                "neuron_ids": outliers["neuron_id"].tolist()
                if len(outliers) < 50
                else outliers["neuron_id"].tolist()[:50],
            }

        # Enhance the class margins with additional statistics
        enhanced_class_margins = {}
        for cls in np.unique(margin_df["class"]):
            cls_str = str(int(cls))
            enhanced_class_margins[cls_str] = {
                "count": class_margins["count"][cls],
                "proportion": class_proportions[cls_str],
                "mean": class_margins["mean"][cls],
                "median": class_margins["median"][cls],
                "std": class_margins["std"][cls],
                "min": class_margins["min"][cls],
                "max": class_margins["max"][cls],
                "quartiles": {"q1": quartiles[0.25][cls], "q2": quartiles[0.50][cls], "q3": quartiles[0.75][cls]},
                "outliers": class_outliers[cls_str],
            }

        # Store margin statistics in results
        margin_key = f"margin_stats_{clf_name}"
        self.results[margin_key] = {
            "overall": margin_stats,
            "class_margins": enhanced_class_margins,
            "class_distribution": {str(c): int(np.sum(self.y_original == c)) for c in np.unique(self.y_original)},
        }

        # Additional analysis for misclassified points and their margins
        if hasattr(clf, "predict"):
            predictions = clf.predict(self.X)
            correct_mask = predictions == self.y
            incorrect_mask = ~correct_mask

            # Margin statistics for correctly and incorrectly classified points
            classification_margins = {
                "correct": {
                    "count": np.sum(correct_mask),
                    "mean_distance": np.mean(distances[correct_mask]) if np.any(correct_mask) else 0,
                    "median_distance": np.median(distances[correct_mask]) if np.any(correct_mask) else 0,
                    "std_distance": np.std(distances[correct_mask]) if np.any(correct_mask) else 0,
                },
                "incorrect": {
                    "count": np.sum(incorrect_mask),
                    "mean_distance": np.mean(distances[incorrect_mask]) if np.any(incorrect_mask) else 0,
                    "median_distance": np.median(distances[incorrect_mask]) if np.any(incorrect_mask) else 0,
                    "std_distance": np.std(distances[incorrect_mask]) if np.any(incorrect_mask) else 0,
                },
            }

            self.results[margin_key]["classification_margins"] = classification_margins

        return self.results[margin_key]

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

        # Store per-class F1 scores
        class_f1_scores = {str(cls): [] for cls in np.unique(self.y)}

        # Perform bootstrap sampling
        for _ in range(n_bootstraps):
            # Create stratified bootstrap samples
            bootstrap_indices = []
            for cls in np.unique(self.y):
                # Get indices for this class
                cls_indices = np.where(self.y == cls)[0]

                # Sample with replacement from this class
                cls_bootstrap = np.random.choice(cls_indices, size=len(cls_indices), replace=True)

                # Add to our bootstrap indices
                bootstrap_indices.extend(cls_bootstrap)

            # Shuffle the indices
            np.random.shuffle(bootstrap_indices)

            # Get bootstrap sample
            X_bootstrap = self.X[bootstrap_indices]
            y_bootstrap = self.y[bootstrap_indices]

            # Split into train and test sets
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_bootstrap,
                y_bootstrap,
                test_size=self.test_size,
                random_state=np.random.randint(1000),
                stratify=y_bootstrap,  # Ensure stratified split
            )

            # Clone and train the classifier
            clf_bootstrap = clone_from_sklearn_estimator(clf)

            # Set class_weight='balanced' if applicable
            if hasattr(clf_bootstrap, "class_weight"):
                clf_bootstrap.class_weight = "balanced"

            clf_bootstrap.fit(self.X_train, self.y_train)

            # Make predictions
            y_pred = clf_bootstrap.predict(self.X_test)

            # Calculate metrics
            accuracies.append(accuracy_score(self.y_test, y_pred))
            f1_scores.append(f1_score(self.y_test, y_pred, average="weighted"))

            # Calculate per-class F1 scores
            if len(np.unique(self.y_test)) <= 2:
                # Binary classification
                for cls in np.unique(self.y_test):
                    if cls == 1:  # positive class
                        cls_f1 = f1_score(self.y_test, y_pred, average="binary")
                        class_f1_scores[str(cls)].append(cls_f1)
            else:
                # Multi-class classification
                f1s = f1_score(self.y_test, y_pred, average=None)
                for i, cls in enumerate(np.unique(self.y_test)):
                    class_f1_scores[str(cls)].append(f1s[i])

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

        # For per-class F1 scores
        class_f1_ci = {}
        for cls, scores in class_f1_scores.items():
            if scores:  # Check if we have scores for this class
                class_f1_ci[cls] = {
                    "lower": np.percentile(scores, 100 * alpha),
                    "mean": np.mean(scores),
                    "upper": np.percentile(scores, 100 * (1 - alpha)),
                }

        # Store bootstrap results
        bootstrap_results = {
            "accuracy_ci": {"lower": acc_lower, "mean": acc_mean, "upper": acc_upper},
            "f1_score_ci": {"lower": f1_lower, "mean": f1_mean, "upper": f1_upper},
            "class_f1_ci": class_f1_ci,
            "n_bootstraps": n_bootstraps,
            "confidence_level": confidence_level,
            "class_distribution": {str(c): int(np.sum(self.y == c)) for c in np.unique(self.y)},
        }

        bootstrap_key = f"bootstrap_{clf_name}"
        self.results[bootstrap_key] = bootstrap_results

        return bootstrap_results

    def analyze_misclassifications(self, clf_name: str = "linear_svc") -> dict:
        """Analyze misclassified samples to identify potential subclusters or outliers."""
        if clf_name not in self.classifiers:
            raise ValueError(f"Classifier '{clf_name}' not found. Train it first.")

        clf = self.classifiers[clf_name]

        # Make predictions
        y_pred = clf.predict(self.X_test)

        # Find misclassified samples
        misclassified_indices = np.where(self.y_test != y_pred)[0]
        correctly_classified_indices = np.where(self.y_test == y_pred)[0]

        # Get the original indices
        indices_in_test = np.arange(len(self.X))[len(self.X_train) :]
        misclassified_original_indices = indices_in_test[misclassified_indices]

        # Get the corresponding neuron indices
        misclassified_neuron_indices = [self.neuron_indices[i] for i in misclassified_original_indices]

        # Analyze misclassifications by class
        true_vs_pred = {}
        for i in misclassified_indices:
            true_class = int(self.y_test[i])
            pred_class = int(y_pred[i])
            key = f"{true_class}_{pred_class}"
            if key not in true_vs_pred:
                true_vs_pred[key] = 0
            true_vs_pred[key] += 1

        # Create misclassification summary
        misclass_analysis = {
            "total_test_samples": len(self.y_test),
            "misclassified_count": len(misclassified_indices),
            "misclassification_rate": len(misclassified_indices) / len(self.y_test),
            "class_confusion": true_vs_pred,
            "misclassified_neuron_indices": misclassified_neuron_indices,
        }

        misclass_key = f"misclass_analysis_{clf_name}"
        self.results[misclass_key] = misclass_analysis

        return misclass_analysis

    def run_pipeline(self) -> dict:
        """Run all classification and analysis methods."""
        # Train classifiers
        logger.info("Training Linear SVC...")
        self.train_linear_svc()

        logger.info("Training SVM with linear kernel...")
        self.train_svm(kernel="linear")

        logger.info("Training comparison classifiers...")
        self.train_comparison_classifiers()

        logger.info("Performing cross-validation...")
        self.cross_validate_svm(kernel="linear", n_splits=5)

        logger.info("Performing permutation test...")
        self.perform_permutation_test(n_permutations=100)  # Use a smaller number for speed

        logger.info("Calculating margin statistics...")
        self.calculate_margin_statistics(clf_name="linear_svc")

        logger.info("Analyzing misclassifications...")
        self.analyze_misclassifications(clf_name="linear_svc")

        # Compile summary of key results
        summary = {
            "class_num": self.class_num,
            "sample_counts": {
                "total": len(self.y),
                "class_distribution": {str(c): int(np.sum(self.y == c)) for c in np.unique(self.y)},
            },
            "linear_svc": {
                "accuracy": self.results["linear_svc"]["accuracy"],
                "f1_score": self.results["linear_svc"]["f1_score"],
                "class_wise_f1": self.results["linear_svc"]["classification_report"]["weighted avg"],
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
        self._save_results()
        return summary

    def _save_results(self) -> None:
        """Save all results and models to disk."""
        # Save results JSON
        results_copy = {k: v for k, v in self.results.items()}

        # Convert numpy arrays in hyperplanes to lists for JSON serialization
        for model_name, hyperplane in self.hyperplanes.items():
            for key, value in hyperplane.items():
                if isinstance(value, np.ndarray):
                    results_copy.setdefault("hyperplanes", {}).setdefault(model_name, {})[key] = value.tolist()

        # Add metadata
        results_copy["metadata"] = {
            "class_num": self.class_num,
            "feature_dim": self.X.shape[1],
            "num_samples": self.X.shape[0],
            "timestamp": datetime.now().isoformat(),
            "original_metadata": self.metadata,
        }

        # Save models
        for name, clf in self.classifiers.items():
            joblib.dump(clf, self.model_path / f"{name}.joblib")
        logger.info(f"Models saved to {self.model_path}")

        # Save results as JSON
        JsonProcessor.save_json(results_copy, self.eval_path / "classification_results.json")
        logger.info(f"Results saved to {self.eval_path}")
