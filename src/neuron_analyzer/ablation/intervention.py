#!/usr/bin/env python
import logging
import random
import typing as t

import numpy as np
import torch
from transformers import GPTNeoXForCausalLM

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
random.seed(42)
T = t.TypeVar("T")


class AblationConfig:
    """Configuration for neuron ablation."""

    def __init__(
        self,
        layer_num: str,
        neurons: list[int] | None = None,
        ablation_mode: str = "base",
        k: int = 10,
        scaling_factor: float = 1.5,
        top_k_percent: float = 0.05,
        variance_threshold: float = 0.95,
        token_frequencies: torch.Tensor = None,
        model_name: str = "pythia-410m",
    ):
        """Initialize ablation configuration."""
        self.layer_name: str = f"gpt_neox.layers.{layer_num}.mlp.dense_h_to_4h"
        self.neurons = neurons
        self.ablation_mode = ablation_mode
        self.scaling_factor = scaling_factor
        self.top_k_percent = top_k_percent
        self.variance_threshold = variance_threshold
        self.token_frequencies = token_frequencies
        self.model_name = model_name
        self.k: int = k


class NeuronAblator:
    """Handles neuron ablation in transformer models."""

    def __init__(
        self, model: GPTNeoXForCausalLM, config: AblationConfig, token_frequencies: torch.Tensor | None = None
    ) -> None:
        """Initialize ablator with model and config."""
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

        # Load token frequencies if needed
        if token_frequencies is None and config.ablation_mode == "scaled":
            self.token_frequencies = self.config.token_frequencies
        else:
            self.token_frequencies = token_frequencies
        self.hook_handle = None
        self.ablation_enabled = False
        self.null_space_basis = None
        self.projection_matrix = None

        # Compute null space if token frequencies are provided and scaled_activation mode is used
        if self.token_frequencies is not None and config.ablation_mode == "scaled":
            self._compute_token_null_space()

        self._setup_hooks()

    def _get_unembedding_matrix(self) -> torch.Tensor:
        """Access the unembedding matrix from the model."""
        try:
            return self.model.embed_out.weight
        except AttributeError:
            pass

        try:
            return self.model.lm_head.weight
        except AttributeError:
            pass

        # Try other common names
        for name, param in self.model.named_parameters():
            if any(term in name.lower() for term in ["unembed", "lm_head", "embed_out", "output_embedding"]):
                logger.info(f"Found unembedding matrix: {name}")
                return param

        # If none found, raise an error
        raise ValueError("Cannot access the unembedding matrix from the model")

    def _compute_token_null_space(self) -> None:
        """Compute the null space of the most frequent tokens based on token frequencies"""
        if self.token_frequencies is None:
            logger.warning("No token frequencies provided. Cannot compute token null space.")
            return

        # Get the unembedding matrix
        W_U = self._get_unembedding_matrix()

        # Get the most frequent tokens
        vocab_size = self.token_frequencies.shape[0]
        top_k = int(vocab_size * self.config.top_k_percent)
        top_indices = torch.argsort(self.token_frequencies, descending=True)[:top_k]

        # Extract the embeddings of top frequent tokens
        frequent_token_embeddings = W_U[top_indices].to(self.device)

        # Get the embedding dimension and hidden size (for neuron activations)
        embedding_dim = frequent_token_embeddings.shape[1]
        hidden_size = None

        # Try to get the hidden size from the model configuration
        try:
            if hasattr(self.model, "config") and hasattr(self.model.config, "hidden_size"):
                hidden_size = self.model.config.hidden_size
            else:
                # For models where we can't directly get hidden_size from config
                # Find the activation dimensions by looking at the MLP layer
                layer_path = self.config.layer_name.split(".")
                module = self.model
                for part in layer_path:
                    if hasattr(module, part):
                        module = getattr(module, part)

                # Check the output dimension of the module
                if hasattr(module, "out_features"):
                    hidden_size = module.out_features
                elif hasattr(module, "weight") and hasattr(module.weight, "shape"):
                    # For typical linear layers, weight shape is [out_features, in_features]
                    hidden_size = module.weight.shape[0]
                else:
                    # If we can't determine it, use the first activation dimension
                    # This is a fallback and might not be correct for all models
                    sample_input = torch.zeros(1, 1, embedding_dim, device=self.device)
                    with torch.no_grad():
                        try:
                            sample_output = module(sample_input)
                            hidden_size = sample_output.shape[-1]
                        except:
                            # Fallback to embedding dimension if all else fails
                            logger.warning(f"Could not determine hidden size, using embedding dim: {embedding_dim}")
                            hidden_size = embedding_dim
        except Exception as e:
            logger.warning(f"Error determining hidden size: {e}. Using embedding dim: {embedding_dim}")
            hidden_size = embedding_dim

        # Convert to numpy for SVD calculation
        embeddings_np = frequent_token_embeddings.detach().cpu().numpy()

        # Perform SVD to find the principal components
        U, S, Vh = np.linalg.svd(embeddings_np, full_matrices=True)

        # Calculate cumulative explained variance
        explained_variance_ratio = S**2 / np.sum(S**2)
        cumulative_variance = np.cumsum(explained_variance_ratio)

        # Find how many components to keep based on threshold
        k = np.argmax(cumulative_variance >= self.config.variance_threshold) + 1
        logger.info(f"Using {k} principal components to span the space of top {top_k} tokens (out of {vocab_size})")

        # The null space is the remaining basis vectors
        null_space_basis = Vh[k:].T

        # Convert to PyTorch tensor
        null_space_basis_tensor = torch.tensor(null_space_basis, dtype=torch.float32, device=self.device)

        # Now we need to adapt the null space basis to match the hidden size
        if embedding_dim != hidden_size:
            # We'll create a simple linear mapping from embedding space to hidden space
            try:
                # Try to find a model component that maps between these spaces
                for name, param in self.model.named_parameters():
                    if ("embed" in name.lower() or "proj" in name.lower()) and param.shape == (
                        hidden_size,
                        embedding_dim,
                    ):
                        logger.info(f"Using existing projection from model: {name}")
                        projection = param.detach()
                        # Transform the null space basis
                        adapted_basis = torch.mm(projection, null_space_basis_tensor.T).T
                        break
                else:
                    # If no suitable projection is found, use a simple dimension adaptation
                    if embedding_dim > hidden_size:
                        # Reduce dimensions by taking the most significant ones
                        logger.info("Reducing null space dimensions")
                        adapted_basis = null_space_basis_tensor[:, :hidden_size]
                    else:
                        # Increase dimensions by padding with zeros
                        logger.info("Expanding null space dimensions with zero padding")
                        padding = torch.zeros(
                            null_space_basis_tensor.shape[0], hidden_size - embedding_dim, device=self.device
                        )
                        adapted_basis = torch.cat([null_space_basis_tensor, padding], dim=1)
            except Exception as e:
                logger.warning(f"Error in adaptation: {e}. Using dimension truncation/padding.")
                if embedding_dim > hidden_size:
                    adapted_basis = null_space_basis_tensor[:, :hidden_size]
                else:
                    padding = torch.zeros(
                        null_space_basis_tensor.shape[0], hidden_size - embedding_dim, device=self.device
                    )
                    adapted_basis = torch.cat([null_space_basis_tensor, padding], dim=1)
        else:
            # Dimensions already match, no adaptation needed
            adapted_basis = null_space_basis_tensor

        # Store the adapted null space basis
        self.null_space_basis = adapted_basis

        # Compute the projection matrix for the adapted basis
        self.projection_matrix = torch.mm(adapted_basis, adapted_basis.T)

    def _setup_hooks(self) -> None:
        """Set up forward hooks for neuron ablation."""

        def ablation_hook(module, input_tensor: tuple[torch.Tensor], output: torch.Tensor) -> torch.Tensor:
            """Forward hook to ablate specified neurons using zero, mean, or scaled activation."""
            if not self.ablation_enabled:
                return output

            # Clone the output to avoid modifying the original tensor in-place
            modified_output = output.clone()

            # If we reach here, ablation is enabled and we have neurons to ablate
            if self.config.ablation_mode in ["zero", "random"]:
                # Zero activation - set activations to 0
                for neuron_idx in self.config.neurons:
                    modified_output[:, :, int(neuron_idx)] = 0

            elif self.config.ablation_mode == "full":
                # Zero activation - set activations to 0
                for neuron_idx in self.config.neurons:
                    modified_output[:, :, int(neuron_idx)] = 1

            elif self.config.ablation_mode == "mean":
                # Calculate mean activation across all neurons (dimension 2)
                mean_activations = torch.mean(output, dim=2)
                # Replace each specified neuron's activation with the mean
                for neuron_idx in self.config.neurons:
                    for b in range(output.shape[0]):
                        for s in range(output.shape[1]):
                            modified_output[b, s, int(neuron_idx)] = mean_activations[b, s]

            elif self.config.ablation_mode == "scaled":
                # Verify that we have computed the null space basis
                if self.null_space_basis is None:
                    logger.warning("Null space basis not computed. Skipping scaled activation.")
                    return modified_output

                # Apply scaling in the null space for specified neurons
                for neuron_idx in self.config.neurons:
                    # Get dimensions
                    batch_size, seq_length = output.shape[0], output.shape[1]

                    # Process each position individually
                    for b in range(batch_size):
                        for s in range(seq_length):
                            # Get the activation value for this position
                            activation = output[b, s, int(neuron_idx)].item()

                            # Create a vector representation for this activation
                            # This maps it into the space where our null space basis exists
                            activation_vector = (
                                torch.ones(self.null_space_basis.shape[0], device=self.device) * activation
                            )

                            # Project the activation onto the null space
                            # The projection matrix is self.projection_matrix
                            null_space_component = torch.mv(self.projection_matrix, activation_vector)

                            # Calculate the magnitude of projection in the null space
                            projection_magnitude = torch.sum(null_space_component)

                            # Scale only the null space component, then add back to original activation
                            scaled_activation = activation + (self.config.scaling_factor - 1.0) * projection_magnitude

                            # Update the activation
                            modified_output[b, s, int(neuron_idx)] = scaled_activation

                    logger.debug(f"Applied null space scaling to neuron {neuron_idx}")

            return modified_output

        # Get the MLP layer
        try:
            layer = dict(self.model.named_modules())[self.config.layer_name]
            # Register the forward hook
            self.hook_handle = layer.register_forward_hook(ablation_hook)
        except KeyError:
            logger.error(f"Layer {self.config.layer_name} not found in model")
            raise ValueError(f"Layer {self.config.layer_name} not found in model")

    def enable_ablation(self) -> None:
        """Enable neuron ablation."""
        self.ablation_enabled = True

    def disable_ablation(self) -> None:
        """Disable neuron ablation."""
        self.ablation_enabled = False

    def cleanup(self) -> None:
        """Remove the hook and clean up resources."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
        # Clear tensors
        self.null_space_basis = None
        self.projection_matrix = None
