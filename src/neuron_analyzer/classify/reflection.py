import logging
import pickle
import typing as t
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SVMHyperplaneReflector:
    """Performs reflection of neuron activations across an SVM decision boundary.

    This class uses PyTorch hooks to modify activations during the forward pass,
    allowing analysis of how crossing the SVM hyperplane affects model behavior.
    """

    def __init__(
        self,
        svm_checkpoint_path: str | Path,
        layer_name: str | None = None,
        layer_num: int = -1,
        model_name: str = "pythia-410m",
        use_pca: bool = False,
        n_components: int = 10,
    ):
        """Initialize the reflector with model and configuration.

        Args:
            model: The neural network model to analyze
            svm_checkpoint_path: Path to the SVM model file
            layer_name: Full name of the layer (overrides layer_num if provided)
            layer_num: Layer number (used if layer_name not provided)
            model_name: Name of the model architecture
            use_pca: Whether to use PCA for dimensionality reduction
            n_components: Number of PCA components if use_pca=True

        """
        self.model = model
        self.device = next(model.parameters()).device
        self.svm_checkpoint_path = Path(svm_checkpoint_path)
        self.use_pca = use_pca
        self.n_components = n_components
        self.model_name = model_name

        # Set layer name based on model architecture or use provided name
        if layer_name is None:
            # Default layer naming based on model architecture
            if "pythia" in model_name.lower() or "gpt-neox" in model_name.lower():
                self.layer_name = f"gpt_neox.layers.{layer_num}.mlp.dense_h_to_4h"
            elif "llama" in model_name.lower():
                self.layer_name = f"model.layers.{layer_num}.mlp.up_proj"
            elif "gpt2" in model_name.lower():
                self.layer_name = f"transformer.h.{layer_num}.mlp.c_fc"
            else:
                # Generic fallback
                self.layer_name = f"model.layers.{layer_num}.mlp"
        else:
            self.layer_name = layer_name

        # Setup hook and state variables
        self.hook_handle = None
        self.reflection_enabled = False
        self.pca = None

        # Target neurons and their activations
        self.target_neurons: list[int] = []
        self.original_activations: dict[int, np.ndarray] = {}
        self.reflected_activations: dict[int, np.ndarray] = {}

        # Hyperplane parameters
        self.normal_vector: np.ndarray | None = None
        self.normal_unit: np.ndarray | None = None
        self.intercept: float | None = None
        self.hyperplane_point: np.ndarray | None = None

        # Load SVM model and set up hooks
        self._load_svm_model()
        self._setup_hooks()

    def _load_svm_model(self) -> None:
        """Load SVM model and extract hyperplane parameters."""
        if not self.svm_checkpoint_path.exists():
            raise FileNotFoundError(f"SVM model file not found: {self.svm_checkpoint_path}")

        # Load the SVM model
        with open(self.svm_checkpoint_path, "rb") as f:
            svm_model = pickle.load(f)

        # Extract hyperplane parameters
        if hasattr(svm_model, "coef_"):
            # scikit-learn SVM model
            self.normal_vector = svm_model.coef_[0]  # w
            self.intercept = svm_model.intercept_[0]  # b
        else:
            # Custom format where hyperplane info is stored directly
            if "w" not in svm_model or "b" not in svm_model:
                raise ValueError("SVM model does not contain expected hyperplane parameters (w and b)")
            self.normal_vector = svm_model["w"]
            self.intercept = svm_model["b"]

        # Normalize the normal vector
        norm = np.linalg.norm(self.normal_vector)
        if norm < 1e-10:
            raise ValueError("Normal vector norm is too small, suggesting an invalid hyperplane")
        self.normal_unit = self.normal_vector / norm

        # Find a point on the hyperplane
        self.hyperplane_point = -self.intercept * self.normal_unit

        logger.info(f"Loaded SVM model from {self.svm_checkpoint_path}")
        logger.info(f"Hyperplane normal vector shape: {self.normal_vector.shape}")
        logger.info(f"Hyperplane intercept: {self.intercept}")

    def _setup_hooks(self) -> None:
        """Set up forward hooks for neuron reflection."""
        try:
            target_layer = self._get_layer_module()

            def reflection_hook(module, input_tensor: tuple[torch.Tensor], output: torch.Tensor) -> torch.Tensor:
                """Forward hook to reflect specified neurons across the hyperplane."""
                if not self.reflection_enabled or not self.target_neurons:
                    return output

                # Clone the output to avoid modifying the original tensor in-place
                modified_output = output.clone()

                # Apply reflection to each target neuron
                for neuron_idx in self.target_neurons:
                    if neuron_idx in self.reflected_activations:
                        # Get the reflected activation for this neuron
                        reflected_value = self.reflected_activations[neuron_idx]

                        # Apply the reflected value based on output tensor shape
                        if len(modified_output.shape) == 3:
                            # [batch, seq_len, hidden_dim]
                            # Modify only the last token's activation for the target neuron
                            modified_output[:, -1, neuron_idx] = torch.tensor(
                                reflected_value, device=modified_output.device, dtype=modified_output.dtype
                            )
                        elif len(modified_output.shape) == 2:
                            # [batch, hidden_dim]
                            modified_output[:, neuron_idx] = torch.tensor(
                                reflected_value, device=modified_output.device, dtype=modified_output.dtype
                            )

                return modified_output

            # Register the hook
            self.hook_handle = target_layer.register_forward_hook(reflection_hook)
            logger.info(f"Registered reflection hook on layer: {self.layer_name}")

        except Exception as e:
            logger.error(f"Error setting up reflection hook: {e}")
            raise

    def reflect_across_hyperplane(self, activation: np.ndarray) -> np.ndarray:
        """Reflect an activation vector across the SVM hyperplane.

        Args:
            activation: Vector to reflect

        Returns:
            Reflected vector on the opposite side of the hyperplane

        """
        # Compute signed distance to hyperplane
        dist_to_plane = np.dot(activation - self.hyperplane_point, self.normal_unit)

        # Apply reflection formula: x' = x - 2 * ((x - p) · n̂) * n̂
        reflected = activation - 2 * dist_to_plane * self.normal_unit
        return reflected

    def classify_neuron(self, activation: np.ndarray) -> str:
        """Classify a neuron based on SVM decision function.

        Args:
            activation: Neuron activation vector

        Returns:
            Classification string: "special" if on positive side of hyperplane, "common" otherwise

        """
        decision_value = np.dot(activation, self.normal_vector) + self.intercept
        return "special" if decision_value > 0 else "common"

    def setup_reflection(self, neuron_idx: int, activation: np.ndarray) -> None:
        """Set up reflection for a specific neuron.

        Args:
            neuron_idx: Index of the neuron to reflect
            activation: Original activation of the neuron

        """
        # Store original activation
        self.original_activations[neuron_idx] = activation

        # Calculate reflected activation
        reflected = self.reflect_across_hyperplane(activation)

        # Store reflected activation
        self.reflected_activations[neuron_idx] = reflected

        # Add to target neurons list if not already there
        if neuron_idx not in self.target_neurons:
            self.target_neurons.append(neuron_idx)

        logger.info(f"Set up reflection for neuron {neuron_idx}")

    def setup_batch_reflection(self, neurons: list[int], activations: np.ndarray) -> None:
        """Set up reflection for multiple neurons.

        Args:
            neurons: List of neuron indices to reflect
            activations: Array of activation vectors for the neurons

        """
        if len(neurons) != len(activations):
            raise ValueError(
                f"Number of neurons ({len(neurons)}) must match number of activations ({len(activations)})"
            )

        # Reset current targets
        self.target_neurons = []
        self.original_activations = {}
        self.reflected_activations = {}

        # Set up reflection for each neuron
        for i, neuron_idx in enumerate(neurons):
            self.setup_reflection(neuron_idx, activations[i])

    def enable_reflection(self) -> None:
        """Enable neuron reflection."""
        self.reflection_enabled = True
        logger.info("Neuron reflection enabled")

    def disable_reflection(self) -> None:
        """Disable neuron reflection."""
        self.reflection_enabled = False
        logger.info("Neuron reflection disabled")

    def cleanup(self) -> None:
        """Remove hooks and clean up resources."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

        # Clear stored data
        self.target_neurons = []
        self.original_activations = {}
        self.reflected_activations = {}
        self.pca = None

        logger.info("Reflector cleaned up")

    def compute_token_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: list[int],
        neurons: list[int],
        activations: np.ndarray,
        original_losses: dict[int, float] = None,
    ) -> dict[str, t.Any]:
        """Compute token loss after reflection.

        Args:
            input_ids: Input token IDs
            target_ids: List of target token IDs to compute loss for
            neurons: List of neuron indices to reflect
            activations: Activation vectors for the neurons
            original_losses: Dictionary of original losses per target token (if already computed)

        Returns:
            Dictionary of loss results

        """
        results = {
            "neurons": neurons,
            "target_ids": target_ids,
            "original_loss": [],
            "reflected_loss": [],
            "loss_changes": [],
            "neuron_types": [],
        }

        # Classify neurons
        for i, neuron_idx in enumerate(neurons):
            neuron_type = self.classify_neuron(activations[i])
            results["neuron_types"].append(neuron_type)

        # If original losses not provided, compute them
        if original_losses is None:
            self.disable_reflection()
            with torch.no_grad():
                outputs = self.model(input_ids)
                # Get logits for the last token prediction
                if hasattr(outputs, "logits"):
                    original_logits = outputs.logits[:, -1, :]  # [batch, vocab_size]
                else:
                    original_logits = outputs[:, -1, :]  # [batch, vocab_size]

                # Calculate loss using model's loss function if available
                # Otherwise use CrossEntropyLoss
                original_losses = {}
                for target_id in target_ids:
                    # Create target tensor [batch_size]
                    target_tensor = torch.tensor([target_id] * input_ids.size(0), device=self.device)

                    # Calculate loss
                    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
                    losses = loss_fn(original_logits, target_tensor)

                    # Average across batch dimension if needed
                    original_losses[target_id] = losses.mean().item()

        # Process each neuron
        for i, neuron_idx in enumerate(tqdm(neurons, desc="Processing neurons")):
            # Store the original loss
            results["original_loss"].append(original_losses)

            # Set up reflection for this neuron
            self.setup_reflection(neuron_idx, activations[i])

            # Compute reflected loss
            self.enable_reflection()

            with torch.no_grad():
                outputs = self.model(input_ids)
                # Get logits for the last token prediction
                if hasattr(outputs, "logits"):
                    reflected_logits = outputs.logits[:, -1, :]  # [batch, vocab_size]
                else:
                    reflected_logits = outputs[:, -1, :]  # [batch, vocab_size]

                # Calculate loss for each target token
                reflected_losses = {}
                for target_id in target_ids:
                    # Create target tensor [batch_size]
                    target_tensor = torch.tensor([target_id] * input_ids.size(0), device=self.device)

                    # Calculate loss
                    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
                    losses = loss_fn(reflected_logits, target_tensor)

                    # Average across batch dimension if needed
                    reflected_losses[target_id] = losses.mean().item()

            # Compute loss changes
            loss_changes = {t_id: reflected_losses[t_id] - original_losses[t_id] for t_id in target_ids}

            # Store results
            results["reflected_loss"].append(reflected_losses)
            results["loss_changes"].append(loss_changes)

            # Disable reflection for next neuron
            self.disable_reflection()

        return results

    def run_reflection_analysis(
        self,
        input_dataset: list[torch.Tensor],
        target_ids: list[int],
        neurons: list[int],
        activations: np.ndarray,
        original_losses_by_context: list[dict[int, float]] = None,
    ) -> dict[str, t.Any]:
        """Run comprehensive reflection analysis on a dataset.

        Args:
            input_dataset: List of input tensors
            target_ids: Target token IDs to analyze loss changes for
            neurons: List of neuron indices to reflect
            activations: Activation vectors for the neurons
            original_losses_by_context: Pre-computed original losses for each context and target token

        Returns:
            Dictionary containing analysis results

        """
        if len(neurons) != len(activations):
            raise ValueError(f"Number of neurons ({len(neurons)}) must match activations ({len(activations)})")

        # Prepare results structure
        results = {
            "neurons": neurons,
            "target_ids": target_ids,
            "original_loss": [[] for _ in range(len(neurons))],
            "reflected_loss": [[] for _ in range(len(neurons))],
            "loss_changes": [[] for _ in range(len(neurons))],
            "original_activations": activations.tolist(),
            "reflected_activations": [],
            "distances_to_hyperplane": [],
            "neuron_types": [],
        }

        # Calculate and store reflected activations and neuron types
        for i, act in enumerate(activations):
            # Calculate distance to hyperplane
            dist = np.dot(act - self.hyperplane_point, self.normal_unit)
            results["distances_to_hyperplane"].append(float(dist))

            # Reflect activation and store
            reflected = self.reflect_across_hyperplane(act)
            results["reflected_activations"].append(reflected.tolist())

            # Classify neuron
            neuron_type = self.classify_neuron(act)
            results["neuron_types"].append(neuron_type)

        # Process each context input
        for context_idx, input_ids in enumerate(tqdm(input_dataset, desc="Processing contexts")):
            # Move input to correct device
            if hasattr(input_ids, "to"):
                input_ids = input_ids.to(self.device)

            # Get or compute original losses for all target tokens
            if original_losses_by_context is not None and context_idx < len(original_losses_by_context):
                # Use pre-computed losses
                original_losses_by_target = original_losses_by_context[context_idx]
            else:
                # Compute original losses
                self.disable_reflection()
                original_losses_by_target = {}

                with torch.no_grad():
                    outputs = self.model(input_ids)
                    # Get logits for the last token prediction
                    if hasattr(outputs, "logits"):
                        original_logits = outputs.logits[:, -1, :]  # [batch, vocab_size]
                    else:
                        original_logits = outputs[:, -1, :]  # [batch, vocab_size]

                    # Calculate loss for each target token
                    for target_id in target_ids:
                        # Create target tensor [batch_size]
                        target_tensor = torch.tensor([target_id] * input_ids.size(0), device=self.device)

                        # Calculate loss
                        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
                        losses = loss_fn(original_logits, target_tensor)

                        # Average across batch dimension if needed
                        original_losses_by_target[target_id] = losses.mean().item()

            # Process each neuron
            for neuron_idx, neuron in enumerate(
                tqdm(neurons, desc=f"Processing neurons for context {context_idx}", leave=False)
            ):
                # Store original loss
                results["original_loss"][neuron_idx].append(original_losses_by_target)

                # Get corresponding activation
                activation = activations[neuron_idx]

                # Set up reflection for this neuron
                self.setup_reflection(neuron, activation)

                # Compute reflected loss
                self.enable_reflection()

                with torch.no_grad():
                    outputs = self.model(input_ids)
                    # Get logits for the last token prediction
                    if hasattr(outputs, "logits"):
                        reflected_logits = outputs.logits[:, -1, :]  # [batch, vocab_size]
                    else:
                        reflected_logits = outputs[:, -1, :]  # [batch, vocab_size]

                    # Calculate reflected loss for each target token
                    reflected_losses = {}
                    for target_id in target_ids:
                        # Create target tensor [batch_size]
                        target_tensor = torch.tensor([target_id] * input_ids.size(0), device=self.device)

                        # Calculate loss
                        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
                        losses = loss_fn(reflected_logits, target_tensor)

                        # Average across batch dimension if needed
                        reflected_losses[target_id] = losses.mean().item()

                # Compute loss changes
                loss_changes = {t_id: reflected_losses[t_id] - original_losses_by_target[t_id] for t_id in target_ids}

                # Store results
                results["reflected_loss"][neuron_idx].append(reflected_losses)
                results["loss_changes"][neuron_idx].append(loss_changes)

                # Disable reflection for the next neuron
                self.disable_reflection()

        return results

    def save_results(self, results: dict[str, t.Any], output_path: str | Path) -> None:
        """Save analysis results to a file.

        Args:
            results: Analysis results dictionary
            output_path: Path to save results to

        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            pickle.dump(results, f)

        logger.info(f"Analysis results saved to {output_path}")

    def load_results(self, input_path: str | Path) -> dict[str, t.Any]:
        """Load analysis results from a file.

        Args:
            input_path: Path to load results from

        Returns:
            Dictionary of analysis results

        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Results file not found: {input_path}")

        with open(input_path, "rb") as f:
            results = pickle.load(f)

        logger.info(f"Analysis results loaded from {input_path}")
        return results

    def analyze_impact(self, results: dict[str, t.Any], threshold: float = 0.1) -> dict[str, t.Any]:
        """Analyze functional impact of reflection across hyperplane.

        Args:
            results: Analysis results dictionary
            threshold: Threshold for significant loss change

        Returns:
            Dictionary with impact metrics

        """
        impact_metrics = {
            "significant_changes": 0,
            "consistent_with_hypothesis": 0,
            "average_loss_delta": 0.0,
            "max_loss_delta": 0.0,
            "neuron_level_impacts": [],
        }

        total_samples = 0
        total_delta = 0.0
        max_delta = 0.0

        for neuron_idx, loss_changes in enumerate(results["loss_changes"]):
            neuron_type = results["neuron_types"][neuron_idx]
            neuron_impact = {
                "neuron_idx": results["neurons"][neuron_idx],
                "neuron_type": neuron_type,
                "distance_to_hyperplane": results["distances_to_hyperplane"][neuron_idx],
                "avg_loss_delta": 0.0,
                "significant_changes": 0,
                "consistent_changes": 0,
            }

            neuron_deltas = []
            for context_changes in loss_changes:
                for token_id, delta in context_changes.items():
                    total_samples += 1
                    neuron_deltas.append(delta)
                    total_delta += abs(delta)
                    max_delta = max(max_delta, abs(delta))

                    # Count significant changes
                    if abs(delta) > threshold:
                        impact_metrics["significant_changes"] += 1
                        neuron_impact["significant_changes"] += 1

                        # Check consistency with hypothesis
                        # For loss, negative delta means better (lower loss)
                        # "special" neurons should decrease loss for target tokens
                        expected_direction = -1 if neuron_type == "special" else 1
                        if (delta * expected_direction) < 0:  # Reversed compared to probability
                            impact_metrics["consistent_with_hypothesis"] += 1
                            neuron_impact["consistent_changes"] += 1

            neuron_impact["avg_loss_delta"] = np.mean(np.abs(neuron_deltas))
            impact_metrics["neuron_level_impacts"].append(neuron_impact)

        if total_samples > 0:
            impact_metrics["average_loss_delta"] = total_delta / total_samples
        impact_metrics["max_loss_delta"] = max_delta

        # Calculate consistency percentage
        if impact_metrics["significant_changes"] > 0:
            impact_metrics["consistency_percentage"] = (
                impact_metrics["consistent_with_hypothesis"] / impact_metrics["significant_changes"] * 100
            )
        else:
            impact_metrics["consistency_percentage"] = 0.0

        return impact_metrics
