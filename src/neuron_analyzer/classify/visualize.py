import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold, permutation_test_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import json


class NeuronClassifier:
    """Class for classifying neurons based on their feature vectors.
    
    This class implements multiple classification approaches to test whether
    special neurons (boosters/suppressors) can be separated from random neurons
    in activation space using hyperplanes.
    """
    
    def __init__(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        neuron_indices: List[str],
        metadata: Optional[Dict] = None,
        classification_mode: str = "three_class",
        random_state: int = 42
    ):
        """Initialize the NeuronClassifier.
        
        Args:
            X: Feature vectors for each neuron (n_samples, n_features)
            y: Labels for each neuron (0: Common, 1: Boost, 2: Suppress)
            neuron_indices: List of neuron identifiers
            metadata: Additional metadata about the neurons
            classification_mode: Either "three_class" (Common vs Boost vs Suppress) or
                               "two_class" (Common vs Special)
            random_state: Random seed for reproducibility
        """
        self.X = X
        self.y_original = y
        self.neuron_indices = neuron_indices
        self.metadata = metadata or {}
        self.classification_mode = classification_mode
        self.random_state = random_state
        
        # Transform labels if using two-class mode
        if classification_mode == "two_class":
            # Convert to binary classification: 0 for common, 1 for special (boost or suppress)
            self.y = np.array([0 if label == 0 else 1 for label in y])
        else:
            self.y = y
        
        # Classifier containers
        self.classifiers = {}
        self.results = {}
        self.feature_importance = {}
        self.hyperplanes = {}
        
    def prepare_data(self, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and testing sets.
        
        Args:
            test_size: Proportion of the dataset to include in the test split
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, 
            random_state=self.random_state, stratify=self.y
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_svm(
        self, 
        kernel: str = 'linear', 
        C: float = 1.0,
        gamma: str = 'scale',
        test_size: float = 0.2
    ) -> Dict:
        """Train an SVM classifier and evaluate its performance.
        
        Args:
            kernel: Kernel type to be used in the SVM
            C: Regularization parameter
            gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
            test_size: Proportion of the dataset to include in the test split
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(test_size)
        
        # Initialize and train the SVM classifier
        clf = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=self.random_state)
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_test)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Calculate silhouette score if there are enough samples
        silhouette = None
        if len(np.unique(y_test)) > 1:
            try:
                silhouette = silhouette_score(X_test, y_pred)
            except Exception as e:
                silhouette = None
        
        # Store model and results
        model_name = f"svm_{kernel}"
        self.classifiers[model_name] = clf
        
        # Extract hyperplane for linear SVM
        if kernel == 'linear':
            self.hyperplanes[model_name] = {
                'w': clf.coef_[0] if len(clf.coef_) == 1 else clf.coef_,
                'b': clf.intercept_[0] if len(clf.intercept_) == 1 else clf.intercept_,
                'support_vectors': clf.support_vectors_
            }
        
        # Store results
        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'silhouette_score': silhouette,
            'model_params': {
                'kernel': kernel,
                'C': C,
                'gamma': gamma
            }
        }
        
        self.results[model_name] = results
        return results
    
    def train_linear_svc(
        self, 
        C: float = 1.0,
        penalty: str = 'l2',
        loss: str = 'squared_hinge',
        dual: bool = True,
        test_size: float = 0.2
    ) -> Dict:
        """Train a LinearSVC classifier (optimized for linear kernel).
        
        Args:
            C: Regularization parameter
            penalty: Norm used in the penalization
            loss: Loss function
            dual: Dual or primal formulation
            test_size: Proportion of the dataset to include in the test split
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(test_size)
        
        # Initialize and train the LinearSVC classifier
        clf = LinearSVC(
            C=C, 
            penalty=penalty, 
            loss=loss, 
            dual=dual, 
            random_state=self.random_state,
            max_iter=10000  # Increased to ensure convergence
        )
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_test)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Calculate silhouette score if possible
        silhouette = None
        if len(np.unique(y_test)) > 1:
            try:
                silhouette = silhouette_score(X_test, y_pred)
            except Exception as e:
                silhouette = None
        
        # Store model and results
        model_name = "linear_svc"
        self.classifiers[model_name] = clf
        
        # Extract hyperplane
        self.hyperplanes[model_name] = {
            'w': clf.coef_[0] if len(clf.coef_) == 1 else clf.coef_,
            'b': clf.intercept_[0] if len(clf.intercept_) == 1 else clf.intercept_
        }
        
        # Store results
        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'silhouette_score': silhouette,
            'model_params': {
                'C': C,
                'penalty': penalty,
                'loss': loss,
                'dual': dual
            }
        }
        
        self.results[model_name] = results
        return results
    
    def train_comparison_classifiers(self, test_size: float = 0.2) -> Dict:
        """Train additional classifiers for comparison.
        
        Args:
            test_size: Proportion of the dataset to include in the test split
            
        Returns:
            Dictionary containing evaluation metrics for all classifiers
        """
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(test_size)
        
        # Initialize classifiers
        classifiers = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(100, 50), 
                max_iter=1000,
                random_state=self.random_state
            )
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
            f1 = f1_score(y_test, y_pred, average='weighted')
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Calculate silhouette score if possible
            silhouette = None
            if len(np.unique(y_test)) > 1:
                try:
                    silhouette = silhouette_score(X_test, y_pred)
                except Exception as e:
                    silhouette = None
            
            # Calculate feature importance for Random Forest
            if name == 'random_forest':
                self.feature_importance[name] = clf.feature_importances_
            
            # Store results
            results = {
                'accuracy': accuracy,
                'f1_score': f1,
                'classification_report': report,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'silhouette_score': silhouette
            }
            
            self.results[name] = results
            comparison_results[name] = results
        
        return comparison_results
    
    def cross_validate_svm(
        self, 
        kernel: str = 'linear',
        C: float = 1.0,
        n_splits: int = 5
    ) -> Dict:
        """Perform cross-validation for SVM classifier.
        
        Args:
            kernel: Kernel type to be used in the SVM
            C: Regularization parameter
            n_splits: Number of splits for k-fold cross-validation
            
        Returns:
            Dictionary containing cross-validation results
        """
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
            f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
        
        # Calculate mean and std of metrics
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        
        # Store cross-validation results
        cv_results = {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'mean_f1': mean_f1,
            'std_f1': std_f1,
            'fold_accuracies': accuracies,
            'fold_f1_scores': f1_scores,
        }
        
        model_name = f"cv_svm_{kernel}"
        self.results[model_name] = cv_results
        
        return cv_results
    
    def perform_permutation_test(
        self, 
        classifier_type: str = 'linear_svc',
        n_permutations: int = 1000,
        test_size: float = 0.2
    ) -> Dict:
        """Perform permutation test to validate statistical significance.
        
        Args:
            classifier_type: Type of classifier to use ('linear_svc' or 'svm')
            n_permutations: Number of permutations for the test
            test_size: Proportion of the dataset to include in the test split
            
        Returns:
            Dictionary containing permutation test results
        """
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(test_size)
        
        # Initialize the classifier
        if classifier_type == 'linear_svc':
            clf = LinearSVC(random_state=self.random_state, max_iter=10000)
        else:
            clf = SVC(kernel='linear', random_state=self.random_state)
        
        # Perform permutation test
        score, perm_scores, pvalue = permutation_test_score(
            clf, X_train, y_train, scoring='accuracy',
            cv=5, n_permutations=n_permutations, random_state=self.random_state
        )
        
        # Store permutation test results
        perm_test_results = {
            'score': score,
            'perm_scores_mean': np.mean(perm_scores),
            'perm_scores_std': np.std(perm_scores),
            'p_value': pvalue,
            'n_permutations': n_permutations
        }
        
        model_name = f"perm_test_{classifier_type}"
        self.results[model_name] = perm_test_results
        
        return perm_test_results
    
    def calculate_margin_statistics(self, clf_name: str = 'linear_svc') -> Dict:
        """Calculate statistics about the margin distribution.
        
        Args:
            clf_name: Name of the classifier to use
            
        Returns:
            Dictionary containing margin statistics
        """
        if clf_name not in self.classifiers:
            raise ValueError(f"Classifier '{clf_name}' not found. Train it first.")
        
        clf = self.classifiers[clf_name]
        
        # For linear SVM, calculate distance to hyperplane
        distances = []
        if clf_name == 'linear_svc' or (clf_name == 'svm_linear'):
            # Get w and b from the hyperplane
            w = self.hyperplanes[clf_name]['w']
            b = self.hyperplanes[clf_name]['b']
            
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
        else:
            # For non-linear SVM with decision_function method
            if hasattr(clf, 'decision_function'):
                # Get raw decision function values
                decision_values = clf.decision_function(self.X)
                
                # For multi-class, take the minimum absolute distance to any hyperplane
                if len(decision_values.shape) > 1:
                    distances = np.min(np.abs(decision_values), axis=1)
                else:
                    distances = np.abs(decision_values)
            else:
                # For other classifiers, use probability as a proxy for confidence
                if hasattr(clf, 'predict_proba'):
                    proba = clf.predict_proba(self.X)
                    distances = np.max(proba, axis=1)  # Use max probability as confidence
        
        # Calculate statistics
        margin_stats = {
            'mean_distance': np.mean(distances),
            'median_distance': np.median(distances),
            'std_distance': np.std(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
        }
        
        # Store margin statistics in results
        margin_key = f"margin_stats_{clf_name}"
        self.results[margin_key] = margin_stats
        
        # Create DataFrame with neuron indices, classes, and distances
        margin_df = pd.DataFrame({
            'neuron_id': self.neuron_indices,
            'class': self.y_original,  # Original class labels
            'distance': distances
        })
        
        # Group by class and calculate statistics
        class_margins = margin_df.groupby('class')['distance'].agg([
            'mean', 'median', 'std', 'min', 'max'
        ]).to_dict()
        
        # Add class-specific margin statistics
        self.results[margin_key]['class_margins'] = class_margins
        
        return margin_stats
    
    def bootstrap_confidence_intervals(
        self, 
        clf_name: str = 'linear_svc',
        n_bootstraps: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict:
        """Calculate bootstrap confidence intervals for classifier metrics.
        
        Args:
            clf_name: Name of the classifier to use
            n_bootstraps: Number of bootstrap samples
            confidence_level: Confidence level for the intervals
            
        Returns:
            Dictionary containing bootstrap confidence intervals
        """
        if clf_name not in self.classifiers:
            raise ValueError(f"Classifier '{clf_name}' not found. Train it first.")
        
        clf = self.classifiers[clf_name]
        
        # Lists to store bootstrap metrics
        accuracies = []
        f1_scores = []
        
        # Perform bootstrap sampling
        for _ in range(n_bootstraps):
            # Sample with replacement
            indices = np.random.choice(
                range(len(self.X)), 
                size=len(self.X), 
                replace=True
            )
            X_bootstrap = self.X[indices]
            y_bootstrap = self.y[indices]
            
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X_bootstrap, y_bootstrap, 
                test_size=0.2, random_state=np.random.randint(1000)
            )
            
            # Clone and train the classifier
            clf_bootstrap = clone_from_sklearn_estimator(clf)
            clf_bootstrap.fit(X_train, y_train)
            
            # Make predictions
            y_pred = clf_bootstrap.predict(X_test)
            
            # Calculate metrics
            accuracies.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
        
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
            'accuracy_ci': {
                'lower': acc_lower,
                'mean': acc_mean,
                'upper': acc_upper
            },
            'f1_score_ci': {
                'lower': f1_lower,
                'mean': f1_mean,
                'upper': f1_upper
            },
            'n_bootstraps': n_bootstraps,
            'confidence_level': confidence_level
        }
        
        bootstrap_key = f"bootstrap_{clf_name}"
        self.results[bootstrap_key] = bootstrap_results
        
        return bootstrap_results
    
    def analyze_misclassifications(self, clf_name: str = 'linear_svc', test_size: float = 0.2) -> Dict:
        """Analyze misclassified samples to identify potential subclusters or outliers.
        
        Args:
            clf_name: Name of the classifier to use
            test_size: Proportion of the dataset to include in the test split
            
        Returns:
            Dictionary containing misclassification analysis
        """
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
        indices_in_test = np.arange(len(self.X))[len(X_train):]
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
            'total_test_samples': len(y_test),
            'misclassified_count': len(misclassified_indices),
            'misclassification_rate': len(misclassified_indices) / len(y_test),
            'class_confusion': true_vs_pred,
            'misclassified_neuron_indices': misclassified_neuron_indices
        }
        
        misclass_key = f"misclass_analysis_{clf_name}"
        self.results[misclass_key] = misclass_analysis
        
        return misclass_analysis
    
    def run_all_analyses(self, test_size: float = 0.2) -> Dict:
        """Run all classification and analysis methods.
        
        Args:
            test_size: Proportion of the dataset to include in the test split
            
        Returns:
            Dictionary containing all results
        """
        # Train classifiers
        print("Training Linear SVC...")
        self.train_linear_svc(test_size=test_size)
        
        print("Training SVM with linear kernel...")
        self.train_svm(kernel='linear', test_size=test_size)
        
        print("Training comparison classifiers...")
        self.train_comparison_classifiers(test_size=test_size)
        
        print("Performing cross-validation...")
        self.cross_validate_svm(kernel='linear', n_splits=5)
        
        print("Performing permutation test...")
        self.perform_permutation_test(n_permutations=100)  # Use a smaller number for speed
        
        print("Calculating margin statistics...")
        self.calculate_margin_statistics(clf_name='linear_svc')
        
        print("Analyzing misclassifications...")
        self.analyze_misclassifications(clf_name='linear_svc', test_size=test_size)
        
        # Compile summary of key results
        summary = {
            'classification_mode': self.classification_mode,
            'sample_counts': {
                'total': len(self.y),
                'class_distribution': {
                    str(c): int(np.sum(self.y == c)) for c in np.unique(self.y)
                }
            },
            'linear_svc': {
                'accuracy': self.results['linear_svc']['accuracy'],
                'f1_score': self.results['linear_svc']['f1_score'],
                'silhouette_score': self.results['linear_svc']['silhouette_score']
            },
            'svm_linear': {
                'accuracy': self.results['svm_linear']['accuracy'],
                'f1_score': self.results['svm_linear']['f1_score'],
                'silhouette_score': self.results['svm_linear']['silhouette_score']
            },
            'cross_validation': {
                'mean_accuracy': self.results['cv_svm_linear']['mean_accuracy'],
                'std_accuracy': self.results['cv_svm_linear']['std_accuracy']
            },
            'permutation_test': {
                'p_value': self.results['perm_test_linear_svc']['p_value']
            },
            'margin_statistics': {
                'mean_distance': self.results['margin_stats_linear_svc']['mean_distance'],
                'class_margins': self.results['margin_stats_linear_svc']['class_margins']
            },
            'misclassification_rate': self.results['misclass_analysis_linear_svc']['misclassification_rate']
        }
        
        self.results['summary'] = summary
        return summary
    
    def save_results(self, output_path: Path) -> None:
        """Save all results and models to disk.
        
        Args:
            output_path: Directory to save results to
        """
        # Create output directory
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save results JSON
        results_copy = {k: v for k, v in self.results.items()}
        
        # Convert numpy arrays in hyperplanes to lists for JSON serialization
        for model_name, hyperplane in self.hyperplanes.items():
            for key, value in hyperplane.items():
                if isinstance(value, np.ndarray):
                    results_copy.setdefault('hyperplanes', {}).setdefault(model_name, {})[key] = value.tolist()
        
        # Add metadata
        results_copy['metadata'] = {
            'classification_mode': self.classification_mode,
            'feature_dim': self.X.shape[1],
            'num_samples': self.X.shape[0],
            'timestamp': datetime.now().isoformat(),
            'original_metadata': self.metadata
        }
        
        # Save results as JSON
        with open(output_path / 'classification_results.json', 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        # Save models
        for name, clf in self.classifiers.items():
            joblib.dump(clf, output_path / f'{name}_model.joblib')
        
        print(f"Results and models saved to {output_path}")


def visualize_pca_projection(X, y, n_components=2, output_path=None):
    """Visualize the data in PCA space.
    
    Args:
        X: Feature matrix
        y: Labels
        n_components: Number of PCA components
        output_path: Path to save the figure
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Create DataFrame for visualization
    df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
    df['class'] = y
    
    # Map classes to descriptive names
    class_names = {0: "Common", 1: "Boost", 2: "Suppress"}
    df['class_name'] = df['class'].map(lambda x: class_names.get(x, f"Class {x}"))
    
    # Create the visualization
    plt.figure(figsize=(10, 8))
    
    if n_components == 2:
        # 2D scatter plot
        sns.scatterplot(x="PC1", y="PC2", hue="class_name", data=df, palette="viridis",
                       alpha=0.7, s=100, edgecolor='k', linewidth=0.5)
        plt.title('PCA Projection of Neuron Activation Patterns')
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    elif n_components == 3:
        # 3D scatter plot
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for cls in df['class'].unique():
            cls_data = df[df['class'] == cls]
            cls_name = class_names.get(cls, f"Class {cls}")
            ax.scatter(cls_data['PC1'], cls_data['PC2'], cls_data['PC3'], 
                      label=cls_name, alpha=0.7, s=100, edgecolor='k', linewidth=0.5)
        
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)")
        ax.set_title('3D PCA Projection of Neuron Activation Patterns')
    
    plt.legend(title="Neuron Class")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
    
    # Return the PCA object and transformed data
    return pca, X_pca


def visualize_tsne_projection(X, y, perplexity=30, n_components=2, output_path=None):
    """Visualize the data in t-SNE space.
    
    Args:
        X: Feature matrix
        y: Labels
        perplexity: t-SNE perplexity parameter
        n_components: Number of t-SNE components
        output_path: Path to save the figure
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Apply t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # Create DataFrame for visualization
    df = pd.DataFrame(X_tsne, columns=[f"TSNE{i+1}" for i in range(n_components)])
    df['class'] = y
    
    # Map classes to descriptive names
    class_names = {0: "Common", 1: "Boost", 2: "Suppress"}
    df['class_name'] = df['class'].map(lambda x: class_names.get(x, f"Class {x}"))
    
    # Create the visualization
    plt.figure(figsize=(10, 8))
    
    if n_components == 2:
        # 2D scatter plot
        sns.scatterplot(x="TSNE1", y="TSNE2", hue="class_name", data=df, 
                       palette="viridis", alpha=0.7, s=100, edgecolor='k', linewidth=0.5)
        plt.title('t-SNE Projection of Neuron Activation Patterns')
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
    elif n_components == 3:
        # 3D scatter plot
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for cls in df['class'].unique():
            cls_data = df[df['class'] == cls]
            cls_name = class_names.get(cls, f"Class {cls}")
            ax.scatter(cls_data['TSNE1'], cls_data['TSNE2'], cls_data['TSNE3'], 
                      label=cls_name, alpha=0.7, s=100, edge
    # Example usage with the LabelAnnotator class from the previous implementation
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Run neuron hyperplane separability test")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--threshold_mode", type=str, default="delta_loss", help="Threshold mode")
    parser.add_argument("--class_mode", type=str, default="three_class", 
                      choices=["three_class", "two_class"], help="Classification mode")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    args = parser.parse_args()
    
    # Convert paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load and prepare the dataset
    threshold = get_threshold(data_dir / "thresholds.json", args.threshold_mode)
    print(f"Using threshold: {threshold}")
    
    label_annotator = LabelAnnotator(
        resume=False,
        class_mode=args.class_mode,
        threshold=threshold,
        threshold_mode=args.threshold_mode,
        data_dir=data_dir
    )
    
    # Run the pipeline to get dataset
    X, y, neuron_indices, metadata = label_annotator.run_pipeline(normalize=True)
    print(f"Dataset prepared: X shape={X.shape}, y shape={y.shape}")
    
    # Step 2: Create and run the classifier
    classifier = NeuronClassifier(
        X=X,
        y=y,
        neuron_indices=neuron_indices,
        metadata=metadata,
        classification_mode=args.class_mode
    )
    
    # Run all analyses
    summary = classifier.run_all_analyses(test_size=0.2)
    print("Classification complete. Summary:")
    for key, value in summary.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value}")
    
    # Save results
    classifier.save_results(output_dir / "classifier_results")
    
    # Step 3: Test the hyperplane hypothesis
    hypothesis_tester = NeuronHypothesisTester(
        classifier_results=classifier.results,
        output_dir=output_dir / "hypothesis_test"
    )
    
    # Generate and save the report
    report = hypothesis_tester.generate_report()
    print("\nHypothesis Test Report Summary:")
    print(f"Evidence Level: {hypothesis_tester.summarize_hyperplane_separation()['evidence_level']}")
    
    # Generate visualizations if X has more than 1 feature
    if X.shape[1] > 1:
        # Visualize hyperplane in 2D (using first two features)
        visualize_hyperplane_2d(
            classifier.classifiers['linear_svc'], 
            X, y, 
            feature_indices=[0, 1],
            output_path=output_dir / "visualizations/hyperplane_2d.png"
        )
        
        # Visualize margins by class
        visualize_margins_by_class(
            classifier.classifiers['linear_svc'],
            X, y,
            output_path=output_dir / "visualizations/margins_by_class.png"
        )
        
        # Visualize feature importance
        visualize_feature_importance(
            classifier.classifiers['linear_svc'],
            output_path=output_dir / "visualizations/feature_importance.png"
        )
    
    print(f"\nAll results saved to {output_dir}")
    return classifier, hypothesis_tester


def clone_from_sklearn_estimator(estimator):
    """Clone a scikit-learn estimator.
    
    This is a helper function for the bootstrap confidence intervals method.
    """
    from sklearn.base import clone
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
                random_state=np.random.randint(1000)
            )
        elif isinstance(estimator, LinearSVC):
            return LinearSVC(
                C=estimator.C,
                penalty=estimator.penalty,
                loss=estimator.loss,
                dual=estimator.dual,
                random_state=np.random.randint(1000),
                max_iter=10000
            )
        else:
            raise ValueError(f"Cannot clone estimator of type {type(estimator)}")


class NeuronHypothesisTester:
    """Class for testing the hyperplane separability hypothesis on neuron data.
    
    This class provides methods to test whether special neurons (boosters/suppressors) can be
    separated from random neurons in activation space via hyperplanes, focusing on visualizations
    and statistical validation of the separation quality.
    """
    
    def __init__(self, classifier_results, output_dir: Path):
        """Initialize the hypothesis tester.
        
        Args:
            classifier_results: Results dictionary from NeuronClassifier
            output_dir: Directory to save results and visualizations
        """
        self.results = classifier_results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def summarize_hyperplane_separation(self) -> Dict:
        """Summarize the evidence for hyperplane separation of neuron groups.
        
        Returns:
            Dictionary containing summary of separation evidence
        """
        # Check if we have all the necessary results
        required_keys = ['linear_svc', 'cv_svm_linear', 'perm_test_linear_svc', 
                         'margin_stats_linear_svc', 'misclass_analysis_linear_svc']
        
        for key in required_keys:
            if key not in self.results:
                raise ValueError(f"Missing required result '{key}'. Run the full analysis first.")
        
        # Extract key metrics
        accuracy = self.results['linear_svc']['accuracy']
        f1_score = self.results['linear_svc']['f1_score']
        cv_accuracy = self.results['cv_svm_linear']['mean_accuracy']
        cv_std = self.results['cv_svm_linear']['std_accuracy']
        perm_pvalue = self.results['perm_test_linear_svc']['p_value']
        
        # Margin statistics
        margin_stats = self.results['margin_stats_linear_svc']
        mean_margin = margin_stats['mean_distance']
        class_margins = margin_stats.get('class_margins', {})
        
        # Misclassification analysis
        misclass_rate = self.results['misclass_analysis_linear_svc']['misclassification_rate']
        
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
                "cross_validation": {
                    "mean_accuracy": cv_accuracy,
                    "std_accuracy": cv_std
                },
                "permutation_test_pvalue": perm_pvalue,
                "misclassification_rate": misclass_rate,
                "margin_statistics": {
                    "mean_margin": mean_margin,
                    "class_specific_margins": class_margins
                }
            },
            "conclusion": self._generate_conclusion(evidence_level, accuracy, perm_pvalue, mean_margin)
        }
        
        # Save to file
        with open(self.output_dir / "hypothesis_test_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def _generate_conclusion(self, evidence_level, accuracy, pvalue, margin):
        """Generate a conclusion based on the evidence level and key metrics."""
        if evidence_level == "Strong":
            return (
                f"With an accuracy of {accuracy:.2f} and p-value of {pvalue:.4f}, "
                f"we find strong evidence supporting the hyperplane separability hypothesis. "
                f"The mean margin of {margin:.4f} suggests a clear separation between neuron groups."
            )
        elif evidence_level == "Moderate":
            return (
                f"With an accuracy of {accuracy:.2f} and p-value of {pvalue:.4f}, "
                f"we find moderate evidence supporting the hyperplane separability hypothesis. "
                f"The mean margin of {margin:.4f} suggests some separation between neuron groups, "
                f"but there may be overlap regions or outliers."
            )
        elif evidence_level == "Weak":
            return (
                f"With an accuracy of {accuracy:.2f} and p-value of {pvalue:.4f}, "
                f"we find weak evidence supporting the hyperplane separability hypothesis. "
                f"The mean margin of {margin:.4f} suggests limited separation between neuron groups, "
                f"with substantial overlap or complex boundaries."
            )
        else:
            return (
                f"With an accuracy of {accuracy:.2f} and p-value of {pvalue:.4f}, "
                f"we find insufficient evidence supporting the hyperplane separability hypothesis. "
                f"The results suggest that neuron groups may not be linearly separable in activation space."
            )
    
    def generate_report(self) -> str:
        """Generate a full report on the hyperplane separability hypothesis test.
        
        Returns:
            Markdown-formatted report string
        """
        # Get summary
        summary = self.summarize_hyperplane_separation()
        
        # Create markdown report
        report = f"""# Neuron Hyperplane Separability Analysis Report

## Summary

**Evidence Level:** {summary['evidence_level']}

{summary['interpretation']}

## Key Findings

- **Accuracy:** {summary['key_metrics']['accuracy']:.4f}
- **F1 Score:** {summary['key_metrics']['f1_score']:.4f}
- **Cross-Validation:** Mean accuracy {summary['key_metrics']['cross_validation']['mean_accuracy']:.4f} ± {summary['key_metrics']['cross_validation']['std_accuracy']:.4f}
- **Statistical Significance:** p-value = {summary['key_metrics']['permutation_test_pvalue']:.4f}
- **Misclassification Rate:** {summary['key_metrics']['misclassification_rate']:.4f}

## Margin Analysis

The mean margin distance is {summary['key_metrics']['margin_statistics']['mean_margin']:.4f}, indicating the average distance of samples from the separating hyperplane.

### Class-Specific Margins

"""
        
        # Add class margins if available
        class_margins = summary['key_metrics']['margin_statistics'].get('class_specific_margins', {})
        if class_margins:
            for cls, margins in class_margins.items():
                class_name = {0: "Common", 1: "Boost", 2: "Suppress"}.get(int(cls), f"Class {cls}")
                report += f"- **{class_name}:** Mean margin = {margins.get('mean', 'N/A'):.4f}, Median = {margins.get('median', 'N/A'):.4f}\n"
        
        # Add conclusion
        report += f"""
## Conclusion

{summary['conclusion']}

## Methodological Details

This analysis used a linear SVM classifier to test whether special neurons (boosters/suppressors) can be separated from random neurons in activation space. The classifier was trained and evaluated using cross-validation, and statistical significance was assessed using permutation tests. Margin analysis was performed to quantify the quality of separation.

"""
        
        # Save report to file
        with open(self.output_dir / "hyperplane_hypothesis_report.md", "w") as f:
            f.write(report)
        
        return report


def visualize_margins_by_class(classifier, X, y, output_path=None):
    """Visualize the distribution of margins by class.
    
    Args:
        classifier: Trained classifier object (SVC or LinearSVC)
        X: Feature matrix
        y: Labels
        output_path: Path to save the figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Calculate decision function values (distance to hyperplane)
    if hasattr(classifier, 'decision_function'):
        # Get raw decision function values
        decision_values = classifier.decision_function(X)
        
        # For multi-class problems
        if len(decision_values.shape) > 1:
            # For multi-class, use the maximum absolute value as the margin
            distances = np.max(np.abs(decision_values), axis=1)
        else:
            distances = np.abs(decision_values)
    else:
        # For other classifiers without decision_function
        if hasattr(classifier, 'predict_proba'):
            proba = classifier.predict_proba(X)
            distances = np.max(proba, axis=1)  # Use max probability as confidence measure
        else:
            raise ValueError("Classifier does not support decision_function or predict_proba")
    
    # Create a DataFrame for easier visualization
    df = pd.DataFrame({
        'margin': distances,
        'class': y
    })
    
    # Map numeric classes to descriptive names
    class_names = {0: "Common", 1: "Boost", 2: "Suppress"}
    df['class_name'] = df['class'].map(lambda x: class_names.get(x, f"Class {x}"))
    
    # Create the visualization
    plt.figure(figsize=(12, 6))
    
    # Box plot
    plt.subplot(1, 2, 1)
    sns.boxplot(x='class_name', y='margin', data=df)
    plt.title('Margin Distribution by Class (Box Plot)')
    plt.xlabel('Neuron Class')
    plt.ylabel('Distance to Hyperplane')
    
    # Violin plot for more detailed distribution
    plt.subplot(1, 2, 2)
    sns.violinplot(x='class_name', y='margin', data=df, inner='quartile')
    plt.title('Margin Distribution by Class (Violin Plot)')
    plt.xlabel('Neuron Class')
    plt.ylabel('Distance to Hyperplane')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
        
    return df


def visualize_feature_importance(classifier, feature_names=None, top_n=10, output_path=None):
    """Visualize feature importance for the classifier.
    
    Args:
        classifier: Trained classifier object
        feature_names: List of feature names
        top_n: Number of top features to show
        output_path: Path to save the figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get feature importance
    if hasattr(classifier, 'coef_'):
        # For linear models
        importance = np.abs(classifier.coef_)
        if len(importance.shape) > 1 and importance.shape[0] > 1:
            # For multiclass, use the average importance across classes
            importance = np.mean(np.abs(importance), axis=0)
        else:
            importance = importance.ravel()
    elif hasattr(classifier, 'feature_importances_'):
        # For tree-based models
        importance = classifier.feature_importances_
    else:
        raise ValueError("Classifier does not provide feature importance")
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importance))]
    
    # Create DataFrame for visualization
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    df = df.sort_values('Importance', ascending=False)
    
    # Limit to top N features
    df = df.head(top_n)
    
    # Create the visualization
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=df)
    plt.title(f'Top {top_n} Features by Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
        
    return df


def visualize_hyperplane_2d(classifier, X, y, feature_indices=[0, 1], output_path=None):
    """Visualize the hyperplane for a 2D projection of the data.
    
    Args:
        classifier: Trained classifier object (SVC or LinearSVC)
        X: Feature matrix
        y: Labels
        feature_indices: Indices of the two features to visualize
        output_path: Path to save the figure
    """
    import matplotlib.pyplot as plt
    
    # Extract the two features
    X_2d = X[:, feature_indices]
    
    # Create a mesh grid
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Create a feature array for the mesh grid points
    mesh_features = []
    for i in range(len(xx.ravel())):
        # Initialize full feature vector with zeros
        full_feature = np.zeros(X.shape[1])
        # Set the two specific features
        full_feature[feature_indices[0]] = xx.ravel()[i]
        full_feature[feature_indices[1]] = yy.ravel()[i]
        mesh_features.append(full_feature)
    
    mesh_features = np.array(mesh_features)
    
    # Predict on the mesh grid
    Z = classifier.predict(mesh_features)
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    
    # Plot the data points
    unique_classes = np.unique(y)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(unique_classes)))
    class_names = {0: "Common", 1: "Boost", 2: "Suppress"}
    
    for i, cls in enumerate(unique_classes):
        plt.scatter(X_2d[y == cls, 0], X_2d[y == cls, 1], 
                   c=[colors[i]], label=class_names.get(cls, f"Class {cls}"),
                   edgecolors='k', alpha=0.8)
    
    # Plot support vectors if the classifier is SVC
    if hasattr(classifier, 'support_vectors_'):
        # Extract the support vectors for the 2D features
        support_vectors = classifier.support_vectors_[:, feature_indices]
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100,
                    linewidth=1, facecolors='none', edgecolors='k',
                    label='Support Vectors')
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
    # Get feature names or indices for axis labels
    feature_names = [f"Feature {idx}" for idx in feature_indices]
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    
    plt.title("Decision Boundary with Neuron Classes")
    plt.legend()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()