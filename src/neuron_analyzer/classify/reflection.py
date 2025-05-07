import logging
import pickle
import typing as t
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from neuron_analyzer.load_util import JsonProcessor

logger = logging.getLogger(__name__)


class SVMHyperplaneReflector:
    """Performs reflection of neuron activations across an SVM decision boundary.

    This class uses PyTorch hooks to modify activations during the forward pass,
    allowing analysis of how crossing the SVM hyperplane affects model behavior.
    """

    def __init__(
        self,
        svm_checkpoint_path: str | Path,
        device: str,
        model,
        tokenizer,
        step_num: int,
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
        # check whether the given neuron has the label
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

    def run_reflection_analysis(
        self,
        tokenized_input: list[int],  # Token IDs for the input string A
        target_token_ids: list[int],  # Token IDs of the target string B
        neurons: list[int],  # List of neuron indices to reflect
        activations: np.ndarray,  # Activation vectors for the neurons
        original_loss: float | None = None,  # Optional pre-computed loss
    ) -> dict[str, t.Any]:
        """Run comprehensive reflection analysis on a tokenized input."""
        if len(neurons) != len(activations):
            raise ValueError(f"Number of neurons ({len(neurons)}) must match activations ({len(activations)})")

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

        # Prepare results structure
        results = {
            "neurons": neurons,
            "target_token_ids": target_token_ids,
            "original_loss": [],
            "reflected_loss": [],
            "loss_changes": [],
            "original_activations": activations.tolist(),
            "reflected_activations": [],
            "distances_to_hyperplane": [],
            "neuron_types": [],
        }

        # Calculate and store reflected activations and neuron types
        for i, act in enumerate(activations):
            # Calculate distance to hyperplane
            dist = float(np.dot(act - self.hyperplane_point, self.normal_unit))
            results["distances_to_hyperplane"].append(dist)

            # Reflect activation and store
            reflected = self.reflect_across_hyperplane(act)
            results["reflected_activations"].append(reflected.tolist())

            # Classify neuron
            neuron_type = self.classify_neuron(act)
            results["neuron_types"].append(neuron_type)

        # Compute or use provided original loss
        if original_loss is None:
            # Compute original loss
            self.disable_reflection()
            original_loss = self.compute_loss(input_ids, start_pos, end_pos)

        # Process each neuron
        for neuron_idx, neuron in enumerate(tqdm(neurons, desc="Processing neurons")):
            # Store original loss
            results["original_loss"].append(original_loss)

            # Get corresponding activation
            activation = activations[neuron_idx]

            # Set up reflection for this neuron
            self.setup_reflection(neuron, activation)
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
        self, input_string: str, target_string: str, neurons: list, activations: np.ndarray, original_loss=None
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
        activations: np.ndarray,
        original_loss=None,
    ) -> list:
        """Run the reflection analysis pipeline."""
        # get the results list
        results = []
        for i, input_string in enumerate(input_string_lst):
            results.append(
                self.run_pipeline_single(
                    input_string, target_string_lst[i], neurons, activations, original_loss=original_loss
                )
            )
        result_dict = {}
        # save the results as a json
        # TODO: change the
        JsonProcessor.save_json
