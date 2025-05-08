import logging
import typing as t
from pathlib import Path

import joblib
import numpy as np
import torch
from tqdm import tqdm

from neuron_analyzer.load_util import JsonProcessor

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
            if hasattr(current_module, part):
                current_module = getattr(current_module, part)
            else:
                raise AttributeError(f"Cannot find submodule {part} in {current_module}")

        logger.info(f"Found target layer: {self.layer_name}")
        return current_module

    def _setup_hooks(self) -> None:
        """Set up forward hooks for neuron reflection."""
        try:
            target_layer = self._get_layer_module()

            # Register the hook
            self.hook_handle = target_layer.register_forward_hook(self.reflection_hook)
            logger.info(f"Registered reflection hook on layer: {self.layer_name}")

        except Exception as e:
            logger.error(f"Error setting up reflection hook: {e}")
            # Include more diagnostic information in the error
            logger.error(f"Layer name: {self.layer_name}")
            logger.error(f"Model structure keys: {[k for k in self.model._modules.keys()]}")
            raise

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
        # return "special" if decision_value > 0 else "common"
        return decision_value

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

    def tokenize_input(self, input_string: str) -> list[int]:
        """Tokenize inputs from the given tokenizer."""
        return self.tokenizer.encode(input_string, add_special_tokens=False)

    def compute_loss(self, input_ids, start_pos, end_pos) -> float:
        """Compute the loss value of the target token."""
        with torch.no_grad():
            outputs = self.model(input_ids)

            # For each position in the target sequence, we need to predict the next token
            total_loss = 0.0
            for pos in range(start_pos, end_pos - 1):
                # Get logits for current position (predicting the next token)
                logits = outputs.logits[0, pos, :] if hasattr(outputs, "logits") else outputs[0, pos, :]
                # Target is the next token
                target = input_ids[0, pos + 1]
                # Calculate loss
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(logits.unsqueeze(0), target.unsqueeze(0))
                total_loss += loss.item()
            # Average loss across target token positions
            num_positions = end_pos - start_pos - 1
            original_loss = total_loss / max(1, num_positions) if num_positions > 0 else 0.0

        return original_loss

    def reflection_hook(self, module, input_tensor: tuple[torch.Tensor], output: torch.Tensor) -> torch.Tensor:
        """Forward hook to reflect specified neurons across the hyperplane."""
        if not self.reflection_enabled or not self.target_neurons:
            return output

        # Clone the output to avoid modifying the original tensor in-place
        modified_output = output.clone()
        logger.info(f"reflected activation length: {len(self.reflected_activations)}")
        # Apply reflection to each target neuron
        for neuron_idx in self.target_neurons:
            # Get the reflected activation for this neuron

            reflected_value = self.reflected_activations[neuron_idx]

            # Convert to float if string
            if isinstance(reflected_value, str):
                try:
                    reflected_value = float(reflected_value)
                except ValueError:
                    logger.error(f"Cannot convert string '{reflected_value}' for neuron {neuron_idx}")
                    continue

            # Make sure neuron_idx is an integer
            neuron_idx = int(neuron_idx)

            # Create a properly shaped tensor for the batch
            batch_size = modified_output.shape[0]
            modified_output[:, -1, neuron_idx] = float(reflected_value[neuron_idx])

            # Handle the reflected value based on its type
            if isinstance(reflected_value, np.ndarray):
                # If it's a full activation vector, use it directly
                # Create tensor of the right shape
                reflected_tensor = torch.tensor(
                    reflected_value, device=modified_output.device, dtype=modified_output.dtype
                )
                # Ensure correct shape for batch broadcasting
                if len(reflected_tensor.shape) == 1:
                    # Expand to batch dimension
                    reflected_tensor = reflected_tensor.expand(batch_size, -1)
                modified_output[:, -1, neuron_idx] = reflected_tensor[neuron_idx]
            else:
                # Scalar value case
                modified_output[:, -1, neuron_idx] = float(reflected_value)

        return modified_output

    def setup_reflection(self, neuron_idx: int, activation: np.ndarray) -> None:
        """Set up reflection for a specific neuron."""
        # Store original activation
        self.original_activations[neuron_idx] = activation

        # Calculate reflected activation
        reflected = self.reflect_across_hyperplane(activation)

        # Store reflected activation
        self.reflected_activations[neuron_idx] = reflected

        # Add to target neurons list if not already there
        if neuron_idx not in self.target_neurons:
            self.target_neurons.append(neuron_idx)

        logger.info(
            f"Set up reflection for neuron {neuron_idx}: original shape={activation.shape}, reflected shape={reflected.shape}"
        )

    def run_reflection_analysis(
        self,
        tokenized_input: list[int],  # Token IDs for the input string A
        target_token_ids: list[int],  # Token IDs of the target string B
        neurons: list[int],  # List of neuron indices to reflect
        activations: list[list],  # Activation vectors for each neuron and token
        original_loss: float | None = None,  # Optional pre-computed loss
    ) -> dict[str, t.Any]:
        """Run comprehensive reflection analysis on a tokenized input."""
        # Convert tokenized input to tensor
        input_ids = torch.tensor([tokenized_input], device=self.device)

        # Find the last occurrence of target_token_ids in input_ids
        target_len = len(target_token_ids)

        # Search from the end to find the last occurrence
        last_pos = -1

        for i in range(len(tokenized_input) - target_len + 1):
            if tokenized_input[i : i + target_len] == target_token_ids:
                last_pos = i

        if last_pos == -1:
            raise ValueError(f"Target token sequence {target_token_ids} not found in input: {tokenized_input}")

        # Store the position of the target tokens (start and end position)
        start_pos = last_pos
        end_pos = last_pos + target_len

        # Log shapes for debugging
        logger.info(f"Neurons length: {len(neurons)}")

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

        # Calculate and store reflected activations for each neuron
        for i, neuron in enumerate(neurons):
            neuron_activation = activations[i]

            # Convert to numpy array if it's not already
            if not isinstance(neuron_activation, np.ndarray):
                neuron_activation = np.array(neuron_activation)

            # Calculate distance to hyperplane
            dist = float(np.dot(neuron_activation - self.hyperplane_point, self.normal_unit))
            results["distances_to_hyperplane"].append(dist)

            # Reflect activation and store
            reflected = self.reflect_across_hyperplane(neuron_activation)
            results["reflected_activations"].append(reflected.tolist())

        # Compute or use provided original loss
        if original_loss is None:
            # Compute original loss
            self.disable_reflection()
            original_loss = self.compute_loss(input_ids, start_pos, end_pos)

        # Process each neuron
        for neuron_idx, neuron in enumerate(tqdm(neurons, desc="Processing neurons")):
            # Store original loss
            results["original_loss"].append(original_loss)

            # Get corresponding activation for this neuron
            neuron_activation = activations[neuron_idx]

            # Make sure it's a numpy array
            if not isinstance(neuron_activation, np.ndarray):
                neuron_activation = np.array(neuron_activation)

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

    def run_pipeline_single(
        self, input_string: str, target_string: str, neurons: list[int], activations: list[list], original_loss=None
    ) -> dict:
        """Run the reflection analysis pipeline."""
        # tokenize the string
        tokenized_input = self.tokenize_input(input_string)
        target_token_ids = self.tokenize_input(target_string)

        # run analyses
        return self.run_reflection_analysis(
            tokenized_input=tokenized_input,
            target_token_ids=target_token_ids,
            neurons=neurons,
            activations=activations,
            original_loss=original_loss,
        )

    def run_pipeline_multi(
        self,
        input_string_lst: list[str],
        target_string_lst: list[str],
        neurons: list[int],
        activation_lst: list[list],
        original_loss=None,
    ) -> list:
        """Run the reflection analysis pipeline for multiple inputs."""
        # get the results list
        results = []

        # Process each input string with its corresponding activation data
        for i, input_string in enumerate(input_string_lst):
            input_activations = [sub[i] for sub in activation_lst]

            # Process this input
            result = self.run_pipeline_single(
                input_string,
                target_string_lst[i],
                neurons,
                input_activations,
                original_loss=original_loss,
            )
            results.append(result)

        # Save results
        result_dict = {self.step_num: results}
        JsonProcessor.save_json(result_dict, self.save_path)

        return results
