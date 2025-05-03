import pickle

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm


class OptimizedSVMHyperplaneReflection:
    """Optimized implementation of the hyperplane reflection method for the last MLP layer.
    This class performs reflection of neuron activations across the SVM decision boundary
    and efficiently computes the effect on token probabilities using linear transformation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        svm_checkpoint_path: str,
        output_projection_weights: torch.Tensor,
        use_pca: bool = False,
        n_components: int = 10,
    ):
        """Initialize the reflection analysis tool with an SVM model.

        Args:
            model: The language model to analyze
            svm_checkpoint_path: Path to the saved SVM model checkpoint (.blob file)
            output_projection_weights: Weight matrix that maps from last MLP layer to logits
                                      (shape: hidden_size × vocab_size)
            use_pca: Whether to use PCA for analysis and visualization
            n_components: Number of PCA components to use if use_pca=True

        """
        self.model = model
        self.svm_model = self._load_svm_model(svm_checkpoint_path)
        self.output_weights = output_projection_weights
        self.use_pca = use_pca
        self.n_components = n_components
        self.pca = None

        # Extract hyperplane parameters from SVM
        self._extract_hyperplane_params()

        print(f"Output projection weight shape: {self.output_weights.shape}")

    def _load_svm_model(self, checkpoint_path: str):
        """Load the trained SVM model from checkpoint.

        Args:
            checkpoint_path: Path to the saved model

        Returns:
            Loaded SVM model

        """
        with open(checkpoint_path, "rb") as f:
            svm_model = pickle.load(f)
        return svm_model

    def _extract_hyperplane_params(self):
        """Extract hyperplane parameters (normal vector and intercept) from SVM model."""
        # For binary classification, coef_ is a single row
        if hasattr(self.svm_model, "coef_"):
            self.normal_vector = self.svm_model.coef_[0]  # w
            self.intercept = self.svm_model.intercept_[0]  # b
        else:
            # If we loaded from your custom format where hyperplane info is stored directly
            self.normal_vector = self.svm_model["w"]
            self.intercept = self.svm_model["b"]

        # Normalize the normal vector
        self.normal_unit = self.normal_vector / np.linalg.norm(self.normal_vector)

        # Calculate a point on the hyperplane
        # We can use: p = -b * n̂ / ||n̂||² = -b * n̂ (since n̂ is unit vector)
        self.hyperplane_point = -self.intercept * self.normal_unit

        print(f"Hyperplane normal vector shape: {self.normal_vector.shape}")
        print(f"Hyperplane point shape: {self.hyperplane_point.shape}")

    def fit_pca(self, activations: np.ndarray):
        """Fit PCA on activation data.

        Args:
            activations: Array of shape (n_samples, n_dimensions) containing neuron activations

        """
        if self.use_pca:
            self.pca = PCA(n_components=self.n_components)
            self.pca.fit(activations)
            print(f"Explained variance ratio: {self.pca.explained_variance_ratio_}")
            print(f"Cumulative explained variance: {np.sum(self.pca.explained_variance_ratio_):.4f}")

    def reflect_across_hyperplane(self, activation: np.ndarray) -> np.ndarray:
        """Reflect an activation vector across the hyperplane.

        Args:
            activation: Vector to reflect

        Returns:
            Reflected vector on the opposite side of the hyperplane

        """
        # Calculate signed distance to hyperplane (a - p) · n̂
        dist_to_plane = np.dot(activation - self.hyperplane_point, self.normal_unit)

        # Reflection formula: a' = a - 2 * (dist_to_plane) * n̂
        reflected = activation - 2 * dist_to_plane * self.normal_unit

        return reflected

    def classify_neuron(self, activation: np.ndarray) -> str:
        """Classify a neuron as special or common based on SVM decision function.

        Args:
            activation: Neuron activation vector

        Returns:
            'special' or 'common' based on which side of the hyperplane the neuron is on

        """
        # Apply SVM decision function: w·x + b
        decision_value = np.dot(activation, self.normal_vector) + self.intercept

        # Determine class based on sign of decision value
        if decision_value > 0:
            return "special"
        return "common"

    def compute_token_probabilities_linear(
        self,
        original_activation: np.ndarray,
        reflected_activation: np.ndarray,
        neuron_idx: int,
        token_ids: list[int],
        original_logits: torch.Tensor,
    ) -> tuple[dict[int, float], dict[int, float]]:
        """Efficiently compute token probabilities before and after reflection"""
        # Convert activations to tensors matching original_logits
        orig_act_tensor = torch.tensor(original_activation, device=original_logits.device, dtype=original_logits.dtype)

        refl_act_tensor = torch.tensor(reflected_activation, device=original_logits.device, dtype=original_logits.dtype)

        # Extract weights connecting this neuron to each token
        neuron_weights = self.output_weights[neuron_idx, :].to(original_logits.device)

        # Compute original and reflected logits
        # Original logits are already provided, but we need to ensure
        # they properly reflect the original activation
        computed_logit_contribution = orig_act_tensor * neuron_weights

        # Compute new logits with reflected activation
        activation_change = refl_act_tensor - orig_act_tensor
        logit_change = activation_change * neuron_weights
        reflected_logits = original_logits + logit_change

        # Apply softmax to get probabilities
        original_probs = torch.nn.functional.softmax(original_logits, dim=-1)
        reflected_probs = torch.nn.functional.softmax(reflected_logits, dim=-1)

        # Extract probabilities for tokens of interest
        original_probs_dict = {token_id: original_probs[token_id].item() for token_id in token_ids}
        reflected_probs_dict = {token_id: reflected_probs[token_id].item() for token_id in token_ids}

        return original_probs_dict, reflected_probs_dict

    def run_reflection_analysis(
        self,
        neurons: list[int],  # List of neuron indices in the last MLP layer
        token_ids: list[int],  # List of rare token IDs to track
        context_inputs: list[torch.Tensor],  # List of context inputs to test with
        neuron_vectors: np.ndarray | None = None,  # Pre-computed neuron activation vectors
        neuron_types: list[str] | None = None,  # Optional pre-assigned neuron types
    ) -> dict:
        """Run the reflection analysis on specified neurons in the last MLP layer."""
        results = {
            "original_probs": [],
            "reflected_probs": [],
            "prob_changes": [],
            "original_activations": [],
            "reflected_activations": [],
            "distances_to_hyperplane": [],
            "neurons": neurons,
            "token_ids": token_ids,
            "neuron_types": neuron_types if neuron_types else [],
        }

        # If neuron types not provided, initialize empty list to fill
        if neuron_types is None:
            results["neuron_types"] = []

        # If using pre-computed neuron vectors
        if neuron_vectors is not None:
            print("Using provided neuron activation vectors")

            # If using PCA, fit it on the neuron vectors
            if self.use_pca:
                self.fit_pca(neuron_vectors)

            # Compute original logits for each context once
            original_logits_by_context = []
            for context in tqdm(context_inputs, desc="Computing original logits"):
                with torch.no_grad():
                    logits = self.model(context)
                    original_logits_by_context.append(logits[0, -1, :].clone())

            # Process each neuron
            for i, neuron_idx in enumerate(tqdm(neurons, desc="Processing neurons")):
                # Get neuron vector
                activation_vector = neuron_vectors[i]

                # Determine neuron type if not provided
                if neuron_types is None:
                    neuron_type = self.classify_neuron(activation_vector)
                    results["neuron_types"].append(neuron_type)

                # Calculate reflection of the activation vector
                reflected_vector = self.reflect_across_hyperplane(activation_vector)

                # Calculate distance to hyperplane
                dist_to_plane = np.dot(activation_vector - self.hyperplane_point, self.normal_unit)

                # Store activation vectors
                if self.use_pca:
                    original_pca = self.pca.transform(activation_vector.reshape(1, -1))[0]
                    reflected_pca = self.pca.transform(reflected_vector.reshape(1, -1))[0]
                    results["original_activations"].append({"full": activation_vector, "pca": original_pca})
                    results["reflected_activations"].append({"full": reflected_vector, "pca": reflected_pca})
                else:
                    results["original_activations"].append(activation_vector)
                    results["reflected_activations"].append(reflected_vector)

                results["distances_to_hyperplane"].append(dist_to_plane)

                # Process each context for this neuron
                neuron_original_probs = []
                neuron_reflected_probs = []
                neuron_prob_changes = []

                for context_idx, context in enumerate(context_inputs):
                    # Get pre-computed original logits
                    original_logits = original_logits_by_context[context_idx]

                    # Compute token probabilities using linear transformation
                    orig_probs, refl_probs = self.compute_token_probabilities_linear(
                        activation_vector, reflected_vector, neuron_idx, token_ids, original_logits
                    )

                    # Compute probability changes
                    prob_changes = {t_id: refl_probs[t_id] - orig_probs[t_id] for t_id in token_ids}

                    neuron_original_probs.append(orig_probs)
                    neuron_reflected_probs.append(refl_probs)
                    neuron_prob_changes.append(prob_changes)

                # Store results for this neuron
                results["original_probs"].append(neuron_original_probs)
                results["reflected_probs"].append(neuron_reflected_probs)
                results["prob_changes"].append(neuron_prob_changes)

        # If we need to compute activations dynamically
        else:
            print("Computing neuron activations dynamically")

            # Collect activations for all neurons and contexts if using PCA
            if self.use_pca:
                print("Collecting activations for PCA...")
                all_activations = []
                for context in tqdm(context_inputs, desc="Collecting activations for PCA"):
                    with torch.no_grad():
                        _ = self.model(context)
                        # This assumes you have a method to get activations for all neurons at once
                        layer_activations = self.model.get_last_mlp_activations().cpu().numpy()
                        for neuron_idx in neurons:
                            all_activations.append(layer_activations[neuron_idx])

                # Fit PCA on all collected activations
                self.fit_pca(np.array(all_activations))

            # Process each neuron
            for i, neuron_idx in enumerate(tqdm(neurons, desc="Processing neurons")):
                neuron_original_probs = []
                neuron_reflected_probs = []
                neuron_prob_changes = []
                neuron_original_activations = []
                neuron_reflected_activations = []
                neuron_distances = []

                # Process each context for this neuron
                for context_idx, context in enumerate(context_inputs):
                    # Run forward pass to get activations and logits
                    with torch.no_grad():
                        # Get original logits
                        logits = self.model(context)
                        original_logits = logits[0, -1, :].clone()

                        # Get activation for this neuron
                        # This assumes you have a method to get the activation for a specific neuron
                        original_activation = self.model.get_last_mlp_activation(neuron_idx).cpu().numpy()

                    # Determine neuron type if not provided and this is the first context
                    if neuron_types is None and context_idx == 0:
                        neuron_type = self.classify_neuron(original_activation)
                        results["neuron_types"].append(neuron_type)

                    # Compute reflection
                    reflected_activation = self.reflect_across_hyperplane(original_activation)

                    # Calculate distance to hyperplane
                    dist_to_plane = np.dot(original_activation - self.hyperplane_point, self.normal_unit)

                    # Store activation vectors and distance
                    if self.use_pca:
                        original_pca = self.pca.transform(original_activation.reshape(1, -1))[0]
                        reflected_pca = self.pca.transform(reflected_activation.reshape(1, -1))[0]
                        neuron_original_activations.append({"full": original_activation, "pca": original_pca})
                        neuron_reflected_activations.append({"full": reflected_activation, "pca": reflected_pca})
                    else:
                        neuron_original_activations.append(original_activation)
                        neuron_reflected_activations.append(reflected_activation)

                    neuron_distances.append(dist_to_plane)

                    # Compute token probabilities using linear transformation
                    orig_probs, refl_probs = self.compute_token_probabilities_linear(
                        original_activation, reflected_activation, neuron_idx, token_ids, original_logits
                    )

                    # Compute probability changes
                    prob_changes = {t_id: refl_probs[t_id] - orig_probs[t_id] for t_id in token_ids}

                    neuron_original_probs.append(orig_probs)
                    neuron_reflected_probs.append(refl_probs)
                    neuron_prob_changes.append(prob_changes)

                # Store results for this neuron
                results["original_probs"].append(neuron_original_probs)
                results["reflected_probs"].append(neuron_reflected_probs)
                results["prob_changes"].append(neuron_prob_changes)
                results["original_activations"].append(neuron_original_activations)
                results["reflected_activations"].append(neuron_reflected_activations)
                results["distances_to_hyperplane"].append(neuron_distances)

        return results
