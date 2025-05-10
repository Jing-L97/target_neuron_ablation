import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_ind

from neuron_analyzer.model_util import get_layer_name

logger = logging.getLogger(__name__)


#######################################################
# Util func to load svm model


def load_svm_model(svm_checkpoint_path: Path) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Load SVM model from path and extract hyperplane parameters."""
    if not svm_checkpoint_path.exists():
        raise FileNotFoundError(f"SVM model file not found: {svm_checkpoint_path}")

    # Load the SVM model
    with open(svm_checkpoint_path, "rb") as f:
        svm_model = joblib.load(f)

    # Extract hyperplane parameters
    if hasattr(svm_model, "coef_"):
        normal_vector = svm_model.coef_[0]  # w
        intercept = svm_model.intercept_[0]  # b
    else:
        if "w" not in svm_model or "b" not in svm_model:
            raise ValueError("SVM model does not contain expected hyperplane parameters (w and b)")
        normal_vector = svm_model["w"]
        intercept = svm_model["b"]

    # Normalize the normal vector
    norm = np.linalg.norm(normal_vector)
    if norm < 1e-10:
        raise ValueError("Normal vector norm is too small, suggesting an invalid hyperplane")
    normal_unit = normal_vector / norm

    # Find a point on the hyperplane
    hyperplane_point = -intercept * normal_unit

    logger.info(f"Loaded SVM model from {svm_checkpoint_path}")
    logger.info(f"Hyperplane intercept: {intercept}")

    return normal_vector, intercept, normal_unit, hyperplane_point


#######################################################
# Util func to load svm model


class SVMHyperplaneReflector:
    """Performs reflection of neuron activation across an SVM decision boundary."""

    def __init__(
        self,
        device: str,
        model,
        layer_num: int,
        neuron_idx: int,
        neuron_activation: float | np.ndarray,
        model_name: str,
        normal_vector: np.ndarray,
        intercept: float,
        normal_unit: np.ndarray,
        hyperplane_point: np.ndarray,
        n_components: int = 10,
    ):
        """Initialize the reflector with model and configuration."""
        self.device = device
        self.model = model
        self.n_components = n_components
        self.model_name = model_name
        self.neuron_idx = int(neuron_idx)
        self.neuron_activation = neuron_activation
        self.layer_num = layer_num
        # Setup hook and state variables
        self.hook_handle = None
        self.reflection_enabled = False

        # Target neurons and their activations
        self.target_neurons: list[int] = []
        self.original_activations: dict[int, np.ndarray] = {}
        self.reflected_activations: dict[int, np.ndarray] = {}

        # Hyperplane parameters
        self.normal_vector = normal_vector
        self.intercept = intercept
        self.normal_unit = normal_unit
        self.hyperplane_point = hyperplane_point

        # For position tracking
        self.start_pos: int | None = None
        self.end_pos: int | None = None
        self.target_positions: list[int] = []

        # set up hook for the model
        self.layer_name = get_layer_name(layer_num=self.layer_num, model_name=self.model_name)
        self._setup_hooks()

    def _setup_hooks(self) -> None:
        """Set up forward hooks for neuron ablation."""

        def ablation_hook(module, input_tensor: tuple[torch.Tensor], output: torch.Tensor) -> torch.Tensor:
            """Forward hook to ablate specified neurons using the original or reflected activation."""
            if not self.reflection_enabled:
                return output

            # Clone the output to avoid modifying the original tensor in-place
            modified_output = output.clone()

            # Use the reflected activation if available, otherwise use original activation
            if self.neuron_idx in self.reflected_activations:
                # Get the reflected value - ensure it's a scalar
                reflected_value = self.reflected_activations[self.neuron_idx]
                if isinstance(reflected_value, np.ndarray):
                    # Convert array to scalar if it's a single value
                    if reflected_value.size == 1:
                        reflected_value = float(reflected_value.item())

                # Set the neuron's activation to this scalar value
                modified_output[:, :, self.neuron_idx] = reflected_value
            else:
                # Use original activation (as scalar)
                activation_value = self.neuron_activation
                if isinstance(activation_value, np.ndarray) and activation_value.size == 1:
                    activation_value = float(activation_value.item())

                modified_output[:, :, self.neuron_idx] = activation_value

            return modified_output

        # Get the MLP layer
        layer = dict(self.model.named_modules())[self.layer_name]
        # Register the forward hook
        self.hook_handle = layer.register_forward_hook(ablation_hook)

    def reflect_across_hyperplane(self) -> float:
        """Reflect a scalar activation value across the SVM hyperplane."""
        # Get activation as scalar
        activation = self.neuron_activation
        if isinstance(activation, np.ndarray) and activation.size == 1:
            activation = float(activation.item())

        # Convert scalar to vector format for hyperplane math
        activation_vector = np.array([activation])
        # Compute signed distance to hyperplane
        dist_to_plane = np.dot(activation_vector - self.hyperplane_point, self.normal_unit)
        # Apply reflection formula: x' = x - 2 * ((x - p) · n̂) * n̂
        reflected_vector = activation_vector - 2 * dist_to_plane * self.normal_unit
        # Convert back to scalar
        reflected_scalar = float(reflected_vector.item()) if reflected_vector.size == 1 else float(reflected_vector[0])
        return reflected_scalar

    def setup_reflection(self) -> None:
        """Set up reflection for a specific neuron."""
        # Store original activation (as scalar)
        original_value = self.neuron_activation
        if isinstance(original_value, np.ndarray) and original_value.size == 1:
            original_value = float(original_value.item())

        self.original_activations[self.neuron_idx] = original_value
        # Calculate reflected activation (will return scalar)
        reflected = self.reflect_across_hyperplane()
        # Store reflected activation (as scalar)
        self.reflected_activations[self.neuron_idx] = reflected
        self.target_neurons.append(self.neuron_idx)

    def _compute_distance(self) -> tuple[float, float]:
        """Calculate distance to hyperplane and reflected value."""
        # Ensure we're working with scalar values
        activation = self.neuron_activation
        if isinstance(activation, np.ndarray) and activation.size == 1:
            activation = float(activation.item())

        # Convert to vector format for dot product
        activation_vector = np.array([activation])

        # Calculate distance to hyperplane (scalar)
        dist = float(np.dot(activation_vector - self.hyperplane_point, self.normal_unit))

        # Get the reflected value (scalar)
        reflected = self.reflect_across_hyperplane()

        return dist, reflected

    def enable_reflection(self) -> None:
        """Enable neuron reflection."""
        self.reflection_enabled = True

    def disable_reflection(self) -> None:
        """Disable neuron reflection."""
        self.reflection_enabled = False

    def cleanup(self) -> None:
        """Remove hooks and clean up resources."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

        # Clear stored data
        self.target_neurons = []
        self.original_activations = {}
        self.reflected_activations = {}

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

    def run_reflection_analysis(
        self,
        tokenized_input: list[int],  # Token IDs for the input string A
        target_token_ids: list[int],  # Token IDs of the target string B
        original_loss: float | None = None,  # Optional pre-computed loss
    ) -> pd.DataFrame:
        """Run comprehensive reflection analysis on one tokenized input."""
        # Convert tokenized input to tensor
        input_ids = torch.tensor([tokenized_input], device=self.device)
        # get the positions of token
        start_pos, end_pos = self._compute_pos(target_token_ids, tokenized_input)

        # Prepare results structure
        results = {
            "neurons": self.neuron_idx,
            "target_token_ids": target_token_ids,
            "original_activations": self.neuron_activation,
        }

        # Compute original loss
        original_loss = self._compute_original_loss(original_loss, input_ids, start_pos, end_pos)
        # comptue reflected activation
        reflected_loss, loss_change = self._compute_reflected_loss(original_loss, input_ids, start_pos, end_pos)
        dist, reflected = self._compute_distance()
        results["reflected_loss"] = reflected_loss
        results["delta_losses"] = loss_change
        results["abs_delta_losses"] = np.abs(loss_change)
        results["distances_to_hyperplane"] = dist
        results["reflected_activations"] = reflected
        results["original_loss"] = original_loss
        return pd.DataFrame(results)

    def _compute_original_loss(self, original_loss, input_ids, start_pos, end_pos) -> dict:
        """Save the original loss into a dict."""
        if original_loss is None:
            self.disable_reflection()
            original_loss = self.compute_loss(input_ids, start_pos, end_pos)
        return original_loss

    def _compute_reflected_loss(
        self,
        original_loss: float,
        input_ids,
        start_pos,
        end_pos,
    ) -> dict:
        """Save the reflected loss into a dict."""
        # Set up reflection for this neuron
        self.setup_reflection()
        # Compute reflected loss
        self.enable_reflection()
        reflected_loss = self.compute_loss(input_ids, start_pos, end_pos)
        # Compute loss change
        loss_change = reflected_loss - original_loss
        # Disable reflection for the next neuron
        self.disable_reflection()
        return reflected_loss, loss_change

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


#######################################################
# Util func to perform stat test


def safe_ttest(sample1: list[float], sample2: list[float]) -> tuple[float, float, bool, str]:
    """Safely perform an independent t-test between two samples, handling edge cases."""
    if not sample1 or not sample2:
        return 0.0, 1.0, False, "unknown"

    # If all values are identical and equal across both samples
    if all(v == sample1[0] for v in sample1 + sample2):
        return 0.0, 1.0, False, "equal"

    try:
        mean1 = np.mean(sample1)
        mean2 = np.mean(sample2)
        std1 = np.std(sample1)
        std2 = np.std(sample2)

        # If either standard deviation is zero
        if std1 == 0 or std2 == 0:
            comparison = "higher" if mean1 > mean2 else "lower"
            return 0.0, 1.0, False, comparison

        # Perform Welch’s t-test (doesn’t assume equal variance)
        tstat, pvalue = ttest_ind(sample1, sample2, equal_var=False)

        comparison = "higher" if mean1 > mean2 else "lower"
        return float(tstat), float(pvalue), bool(pvalue < 0.05), comparison

    except Exception as e:
        logger.warning(f"Error performing t-test: {e}")
        return 0.0, 1.0, False, "unknown"
