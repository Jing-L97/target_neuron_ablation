import logging
import typing as t
from pathlib import Path

import joblib
import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SVMHyperplaneReflector:
    """Performs reflection of neuron activations across an SVM decision boundary."""

    def __init__(
        self,
        svm_checkpoint_path: str | Path,
        device: str,
        model,
        tokenizer,
        step_num: int,
        save_path: Path,
        layer_name: str | None = None,
        layer_num: int = -1,
        model_name: str = "pythia-410m",
        use_pca: bool = False,
        n_components: int = 10,
    ):
        """Initialize the reflector with model and configuration."""
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.svm_checkpoint_path = svm_checkpoint_path
        self.use_pca = use_pca
        self.n_components = n_components
        self.model_name = model_name
        self.step_num = step_num

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
        self.save_path = save_path
        # Load SVM model and set up hooks
        self._load_svm_model()
        # set up hook for the model
        self._setup_hooks()

    def _load_svm_model(self) -> None:
        """Load SVM model and extract hyperplane parameters."""
        if not self.svm_checkpoint_path.exists():
            raise FileNotFoundError(f"SVM model file not found: {self.svm_checkpoint_path}")

        # Load the SVM model
        with open(self.svm_checkpoint_path, "rb") as f:
            svm_model = joblib.load(f)

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

    def _get_layer_module(self) -> torch.nn.Module:
        """Get the target layer module based on the layer name."""
        # Split the layer name into parts
        name_parts = self.layer_name.split(".")

        # Start from the model and navigate through the hierarchy
        current_module = self.model
        for part in name_parts:
            current_module = getattr(current_module, part)
        logger.info(f"Found target layer: {self.layer_name}")
        return current_module

    def _setup_hooks(self) -> None:
        """Set up forward hooks for neuron ablation."""

        def ablation_hook(module, input_tensor: tuple[torch.Tensor], output: torch.Tensor) -> torch.Tensor:
            """Forward hook to ablate specified neurons using zero, mean, or scaled activation."""
            if not self.reflection_enabled:
                return output
            # Clone the output to avoid modifying the original tensor in-place
            modified_output = output.clone()
            modified_output[:, :, self.neuron_idx] = 0
            return modified_output

        # Get the MLP layer
        layer = dict(self.model.named_modules())[self.layer_name]
        # Register the forward hook
        self.hook_handle = layer.register_forward_hook(ablation_hook)

    def reflect_across_hyperplane(self, activation: np.ndarray) -> np.ndarray:
        """Reflect an activation vector of multiple tokens across the SVM hyperplane."""
        # Compute signed distance to hyperplane
        dist_to_plane = np.dot(activation - self.hyperplane_point, self.normal_unit)
        # Apply reflection formula: x' = x - 2 * ((x - p) · n̂) * n̂
        reflected = activation - 2 * dist_to_plane * self.normal_unit
        return reflected

    def classify_neuron(self, activation: np.ndarray) -> str:
        """Classify a neuron based on SVM decision function."""
        # check whether the given neuron has the label
        decision_value = np.dot(activation, self.normal_vector) + self.intercept
        logger.info(decision_value)
        return "special" if decision_value > 0 else "common"

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

    def compute_loss(self, input_ids, start_pos, end_pos) -> float:
        """Compute the loss value of the target token or token sequence."""
        with torch.no_grad():
            # Get model outputs
            outputs = self.model(input_ids)

            # Validate positions are in bounds
            seq_length = input_ids.shape[1]
            if start_pos < 0 or end_pos > seq_length:
                logger.warning(f"Invalid position range: {start_pos} to {end_pos}, sequence length: {seq_length}")
                return 0.0
            # Check if we can predict any tokens
            if start_pos >= seq_length - 1 or start_pos >= end_pos:
                logger.warning(
                    f"Cannot predict tokens: start_pos={start_pos}, end_pos={end_pos}, seq_length={seq_length}"
                )
                return 0.0

            # Create loss function once
            loss_fn = torch.nn.CrossEntropyLoss()

            # Calculate loss for each position
            total_loss = 0.0
            valid_positions = 0
            # Loop through positions where we can predict the next token
            last_pos = min(end_pos - 1, seq_length - 2)  # We need at least one more token after this position
            for pos in range(start_pos, last_pos + 1):
                # Skip if next token is out of bounds
                if pos + 1 >= seq_length:
                    continue
                logits = outputs.logits[0, pos, :]
                # Get target (next token) - this is a scalar tensor
                target = input_ids[0, pos + 1]
                # Calculate loss
                loss = loss_fn(logits.unsqueeze(0), target.unsqueeze(0))
                # Add to total and count valid positions
                total_loss += loss.item()  # Convert to Python float
                valid_positions += 1

            # Return average loss, or 0.0 if no valid positions
            if valid_positions > 0:
                return total_loss / valid_positions
            logger.warning("No valid positions for loss calculation")
            return 0.0

    def get_target_positions(self) -> list[int]:
        """Return positions that correspond to target tokens we want to affect."""
        # Return the range from start_pos to end_pos if they're set
        if self.start_pos is not None and self.end_pos is not None:
            return list(range(self.start_pos, self.end_pos))

        # Otherwise return any stored target positions
        return self.target_positions

    def setup_reflection(self, neuron_idx: int, activation: np.ndarray) -> None:
        """Set up reflection for a specific neuron."""
        # Store original activation
        self.original_activations[neuron_idx] = activation
        # Calculate reflected activation
        reflected = self.reflect_across_hyperplane(activation)
        # Store reflected activation
        self.reflected_activations[neuron_idx] = reflected
        self.target_neurons.append(neuron_idx)
        logger.info(
            f"Set up reflection for neuron {neuron_idx}: original shape={activation.shape}, reflected shape={reflected.shape}"
        )

    def run_reflection_analysis(
        self,
        tokenized_input: list[int],  # Token IDs for the input string A
        target_token_ids: list[int],  # Token IDs of the target string B
        neurons: list[int],  # List of neuron indices to reflect
        activations: list[float],  # Activation values for each neuron of the given token
        original_loss: float | None = None,  # Optional pre-computed loss
    ) -> dict[str, t.Any]:
        """Run comprehensive reflection analysis on one tokenized input."""
        # Convert tokenized input to tensor
        input_ids = torch.tensor([tokenized_input], device=self.device)
        # get the positions of token
        start_pos, end_pos = self._compute_pos(target_token_ids, tokenized_input)

        # Prepare results structure
        results = {
            "neurons": neurons,
            "target_token_ids": target_token_ids,
            "original_loss": [],
            "reflected_loss": [],
            "loss_changes": [],
            "original_activations": activations.tolist() if isinstance(activations, np.ndarray) else activations,
            "reflected_activations": [],
            "distances_to_hyperplane": [],
            "neuron_types": [],
        }

        # Compute original loss
        results = self._compute_original_loss(original_loss, input_ids, start_pos, end_pos, results)

        # Process each neuron
        for neuron_idx, neuron in enumerate(tqdm(neurons, desc="Processing neurons")):
            # Get corresponding activation for this neuron
            neuron_activation = activations[neuron_idx]
            results = self._compute_reflected_loss(
                neuron_activation, neuron, original_loss, input_ids, start_pos, end_pos, results
            )
            results = self._compute_distance(neuron_activation, results)
        return results

    def _compute_original_loss(self, original_loss, input_ids, start_pos, end_pos, results) -> dict:
        """Save the original loss into a dict."""
        if original_loss is None:
            self.disable_reflection()
            original_loss = self.compute_loss(input_ids, start_pos, end_pos)
        # Store original loss
        results["original_loss"].append(original_loss)
        return results

    def _compute_reflected_loss(
        self,
        neuron_activation: float,
        neuron: int,
        original_loss: float,
        input_ids,
        start_pos,
        end_pos,
        results,
    ) -> dict:
        """Save the reflected loss into a dict."""
        # Set up reflection for this neuron
        self.setup_reflection(neuron, neuron_activation)
        # Compute reflected loss
        self.enable_reflection()
        reflected_loss = self.compute_loss(input_ids, start_pos, end_pos)
        # Compute loss change
        loss_change = reflected_loss - original_loss
        # Store results
        results["reflected_loss"].append(reflected_loss)
        results["loss_changes"].append(loss_change)
        # Disable reflection for the next neuron
        self.disable_reflection()
        return results

    def _compute_distance(self, neuron_activation: float, results) -> dict:
        """Save the reflected distance into a dict."""
        # Calculate distance to hyperplane
        dist = float(np.dot(neuron_activation - self.hyperplane_point, self.normal_unit))
        results["distances_to_hyperplane"].append(dist)
        # Reflect activation and store
        reflected = self.reflect_across_hyperplane(neuron_activation)
        results["reflected_activations"].append(reflected.tolist())
        return results

    def _compute_pos(self, target_token_ids, tokenized_input) -> tuple[int, int]:
        """Compute pos based on the string length."""
        # Find the last occurrence of target_token_ids in input_ids
        target_len = len(target_token_ids)

        # Search from the end to find the last occurrence
        last_pos = -1
        for i in range(len(tokenized_input) - target_len + 1):
            if tokenized_input[i : i + target_len] == target_token_ids:
                last_pos = i
        if last_pos == -1:
            raise ValueError(f"Target token sequence {target_token_ids} not found in input: {tokenized_input}")
        return last_pos, last_pos + target_len

    def save_results():
        """Convert resutls into a df."""
