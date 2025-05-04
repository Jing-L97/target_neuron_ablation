import pickle
import typing as t
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm


class OptimizedSVMHyperplaneReflection:
    """Optimized implementation of the hyperplane reflection for analyzing neural network layers.

    This class performs reflection of neuron activations across an SVM decision boundary
    and computes the effect on token probabilities using linear transformations.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        svm_checkpoint_path: str | Path,
        output_projection_weights: torch.Tensor,
        use_pca: bool = False,
        n_components: int = 10,
    ):
        """Initialize the reflection analysis tool.

        Args:
            model: The language model to analyze
            svm_checkpoint_path: Path to the saved SVM model (.blob file)
            output_projection_weights: Weight matrix mapping last MLP layer to logits
                                     (shape: hidden_size × vocab_size)
            use_pca: Whether to use PCA for dimensionality reduction
            n_components: Number of PCA components if use_pca=True

        """
        self.model = model
        self.svm_checkpoint_path = Path(svm_checkpoint_path)
        self.output_weights = output_projection_weights
        self.use_pca = use_pca
        self.n_components = n_components
        self.pca = None

        # Load SVM model and extract hyperplane parameters
        self.svm_model = self._load_svm_model()
        self._extract_hyperplane_params()

        print(f"Output projection weight shape: {self.output_weights.shape}")

    def _load_svm_model(self) -> dict[str, np.ndarray] | t.Any:
        """Load the trained SVM model from checkpoint."""
        with open(self.svm_checkpoint_path, "rb") as f:
            return pickle.load(f)

    def _extract_hyperplane_params(self) -> None:
        """Extract hyperplane parameters (normal vector and intercept) from SVM model."""
        # Handle different SVM model formats
        if hasattr(self.svm_model, "coef_"):
            self.normal_vector = self.svm_model.coef_[0]  # w
            self.intercept = self.svm_model.intercept_[0]  # b
        else:
            # Custom format where hyperplane info is stored directly
            self.normal_vector = self.svm_model["w"]
            self.intercept = self.svm_model["b"]

        # Normalize the normal vector for stability
        self.normal_unit = self.normal_vector / np.linalg.norm(self.normal_vector)

        # Find a point on the hyperplane (using -b/||w||² * w)
        self.hyperplane_point = -self.intercept * self.normal_unit

        print(f"Hyperplane normal vector shape: {self.normal_vector.shape}")
        print(f"Hyperplane intercept: {self.intercept}")

    def reflect_across_hyperplane(self, activation: np.ndarray) -> np.ndarray:
        """Reflect an activation vector across the hyperplane.

        The reflection formula is: x' = x - 2 * ((x - p) · n̂) * n̂
        where x is the point to reflect, p is a point on the hyperplane,
        and n̂ is the unit normal vector.

        Args:
            activation: Vector to reflect

        Returns:
            Reflected vector on the opposite side of the hyperplane

        """
        # Compute signed distance to hyperplane
        dist_to_plane = np.dot(activation - self.hyperplane_point, self.normal_unit)

        # Apply reflection formula
        reflected = activation - 2 * dist_to_plane * self.normal_unit

        # Verify the reflection is correct
        self._verify_reflection(activation, reflected)

        return reflected

    def _verify_reflection(self, original: np.ndarray, reflected: np.ndarray) -> None:
        """Verify that the reflection is correct by checking:
        1. The reflected point is on the opposite side of the hyperplane
        2. The distance from both points to the hyperplane is equal
        """
        # Calculate distances
        orig_dist = np.dot(original - self.hyperplane_point, self.normal_unit)
        refl_dist = np.dot(reflected - self.hyperplane_point, self.normal_unit)

        # Check if signs are opposite (on different sides)
        assert np.sign(orig_dist) != np.sign(refl_dist), "Points are not on opposite sides"

        # Check if distances are equal
        assert np.isclose(abs(orig_dist), abs(refl_dist)), "Distances are not equal"

    def classify_neuron(self, activation: np.ndarray) -> str:
        """Classify a neuron based on SVM decision function."""
        decision_value = np.dot(activation, self.normal_vector) + self.intercept
        return "special" if decision_value > 0 else "common"

    def compute_token_probabilities_linear(
        self,
        original_activation: np.ndarray,
        reflected_activation: np.ndarray,
        neuron_idx: int,
        token_ids: list[int],
        original_logits: torch.Tensor,
    ) -> tuple[dict[int, float], dict[int, float]]:
        """Efficiently compute token probabilities before and after reflection.

        This method uses linear transformation properties to compute the change
        in logits due to the activation change.
        """
        # Convert to tensors
        orig_tensor = torch.tensor(original_activation, device=original_logits.device, dtype=original_logits.dtype)
        refl_tensor = torch.tensor(reflected_activation, device=original_logits.device, dtype=original_logits.dtype)

        # Extract weights for this neuron
        neuron_weights = self.output_weights[neuron_idx, :].to(original_logits.device)

        # Compute logit change
        activation_change = refl_tensor - orig_tensor
        logit_change = activation_change * neuron_weights
        reflected_logits = original_logits + logit_change

        # Apply softmax
        original_probs = torch.nn.functional.softmax(original_logits, dim=-1)
        reflected_probs = torch.nn.functional.softmax(reflected_logits, dim=-1)

        # Extract probabilities for specified tokens
        orig_probs_dict = {token_id: original_probs[token_id].item() for token_id in token_ids}
        refl_probs_dict = {token_id: reflected_probs[token_id].item() for token_id in token_ids}

        return orig_probs_dict, refl_probs_dict

    def fit_pca(self, activations: np.ndarray) -> None:
        """Fit PCA on activation data for dimensionality reduction."""
        if self.use_pca:
            self.pca = PCA(n_components=self.n_components)
            self.pca.fit(activations)
            variance_explained = np.sum(self.pca.explained_variance_ratio_)
            print(f"Cumulative explained variance: {variance_explained:.4f}")

    def run_reflection_analysis(
        self,
        neurons: list[int],
        token_ids: list[int],
        context_inputs: list[torch.Tensor],
        neuron_vectors: np.ndarray,
        neuron_types: list[str] | None = None,
    ) -> dict[str, t.Any]:
        """Run comprehensive reflection analysis."""
        results = {
            "original_probs": [],
            "reflected_probs": [],
            "prob_changes": [],
            "original_activations": [],
            "reflected_activations": [],
            "distances_to_hyperplane": [],
            "neurons": neurons,
            "token_ids": token_ids,
            "neuron_types": neuron_types or [],
        }

        # Fit PCA if enabled
        if self.use_pca:
            self.fit_pca(neuron_vectors)

        # Compute original logits
        original_logits_list = []
        for context in tqdm(context_inputs, desc="Computing original logits"):
            with torch.no_grad():
                logits = self.model(context)
                original_logits_list.append(logits[0, -1, :].clone())

        # Process each neuron
        for i, neuron_idx in enumerate(tqdm(neurons, desc="Processing neurons")):
            activation = neuron_vectors[i]

            # Classify neuron if types not provided
            if neuron_types is None:
                neuron_type = self.classify_neuron(activation)
                results["neuron_types"].append(neuron_type)

            # Reflect activation
            reflected = self.reflect_across_hyperplane(activation)

            # Calculate distance to hyperplane
            dist = np.dot(activation - self.hyperplane_point, self.normal_unit)
            results["distances_to_hyperplane"].append(dist)

            # Store activations
            if self.use_pca:
                orig_pca = self.pca.transform(activation.reshape(1, -1))[0]
                refl_pca = self.pca.transform(reflected.reshape(1, -1))[0]
                results["original_activations"].append({"full": activation, "pca": orig_pca})
                results["reflected_activations"].append({"full": reflected, "pca": refl_pca})
            else:
                results["original_activations"].append(activation)
                results["reflected_activations"].append(reflected)

            # Process each context
            neuron_orig_probs = []
            neuron_refl_probs = []
            neuron_prob_changes = []

            for j, context in enumerate(context_inputs):
                orig_logits = original_logits_list[j]

                # Compute probabilities
                orig_probs, refl_probs = self.compute_token_probabilities_linear(
                    activation, reflected, neuron_idx, token_ids, orig_logits
                )

                # Calculate changes
                prob_changes = {t_id: refl_probs[t_id] - orig_probs[t_id] for t_id in token_ids}

                neuron_orig_probs.append(orig_probs)
                neuron_refl_probs.append(refl_probs)
                neuron_prob_changes.append(prob_changes)

            results["original_probs"].append(neuron_orig_probs)
            results["reflected_probs"].append(neuron_refl_probs)
            results["prob_changes"].append(neuron_prob_changes)

        return results
