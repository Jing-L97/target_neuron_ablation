#!/usr/bin/env python
import logging
import pickle
import sys
import typing as t
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
sys.path.append("../")
T = t.TypeVar("T")


@dataclass
class NeuronEvalResult:
    """Result of a neuron group evaluation."""

    neurons: list[int]
    delta_loss: float


class NeuronGroupEvaluator:
    """Lightweight evaluator for measuring neuron group impact via ablation."""

    def __init__(
        self,
        model,
        tokenized_data,
        tokenizer,
        effect: str,
        device: str,
        layer_idx: int,
        cache_dir: str | Path | None = None,
    ):
        """Initialize the neuron group evaluator."""
        self.model = model
        self.tokenized_data = tokenized_data
        self.tokenizer = tokenizer
        self.device = device
        self.layer_idx = layer_idx

        # Set up caching
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True, parents=True)
            self.eval_cache = self._load_cache()
        else:
            self.eval_cache = {}

    def _get_cache_path(self) -> Path:
        """Get the path to the evaluation cache file."""
        return self.cache_dir / f"neuron_eval_cache_layer_{self.layer_idx}.pkl"

    def _load_cache(self) -> dict:
        """Load cached evaluation results if available."""
        if not self.cache_dir:
            return {}

        cache_path = self._get_cache_path()
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                # If loading fails, return empty cache
                return {}
        return {}

    def _save_cache(self) -> None:
        """Save evaluation cache to disk."""
        if not self.cache_dir:
            return

        cache_path = self._get_cache_path()
        with open(cache_path, "wb") as f:
            pickle.dump(self.eval_cache, f)

    def get_act_name(self, hook_type: str, layer_idx: int) -> str:
        # Based on your actual cache keys
        if hook_type == "post":
            # This should fix the KeyError: 'blocks.5.post'
            return f"blocks.{layer_idx}.mlp.hook_post"
        if hook_type == "resid_post":
            return f"blocks.{layer_idx}.hook_resid_post"
        if hook_type == "mlp_post":
            return f"blocks.{layer_idx}.mlp.hook_post"
        if hook_type == "mlp_out":
            return f"blocks.{layer_idx}.hook_mlp_out"
        # For other hook types
        return f"blocks.{layer_idx}.{hook_type}"

    def evaluate_neuron_group(self, neuron_group: list[int], sample_batches: int = 3) -> float:
        """Evaluate the impact of a neuron group by measuring delta loss."""
        # Skip empty groups
        if not neuron_group:
            return 0.0

        # Check cache first
        cache_key = tuple(sorted(neuron_group))
        if cache_key in self.eval_cache:
            return self.eval_cache[cache_key]

        # Sample random batches for evaluation
        available_batches = list(range(len(self.tokenized_data["tokens"])))
        batch_indices = np.random.choice(available_batches, min(sample_batches, len(available_batches)), replace=False)

        total_delta_loss = 0.0

        for batch_idx in batch_indices:
            # Get the delta loss for this batch
            batch_delta = self._compute_delta_loss_for_batch(batch_idx, neuron_group)
            total_delta_loss += batch_delta

        # Average across batches
        avg_delta_loss = total_delta_loss / len(batch_indices)

        # Cache the result
        self.eval_cache[cache_key] = avg_delta_loss

        # Periodically save cache
        if len(self.eval_cache) % 50 == 0:
            self._save_cache()

        return avg_delta_loss

    def _compute_delta_loss_for_batch(self, batch_idx: int, neuron_group: list[int]) -> float:
        """Compute delta loss for a specific batch and neuron group."""
        # Get the input sequence
        tok_seq = self.tokenized_data["tokens"][batch_idx]
        if isinstance(tok_seq, str):
            tok_seq = self.tokenizer(tok_seq, return_tensors="pt")["input_ids"]
            logger.info("Tokenizing the input string")

        # dimenison handling for the right shape: [batch_size, sequence_length]
        # Make sure we have the right shape: [batch_size, sequence_length]
        inp = tok_seq.unsqueeze(0).to(self.device) if tok_seq.dim() == 1 else tok_seq.to(self.device)

        # Get original loss
        self.model.reset_hooks()
        original_logits, cache = self.model.run_with_cache(inp)
        original_loss = self.model.loss_fn(original_logits, inp, per_token=True).mean().item()

        # Get neuron activations
        activations = cache[self.get_act_name("post", self.layer_idx)][0]

        # across the entire dataset
        neuron_means = activations.mean(dim=0)

        # Create activation deltas for the specified neurons
        activation_deltas = torch.zeros_like(activations)
        for neuron_idx in neuron_group:
            activation_deltas[:, neuron_idx] = neuron_means[neuron_idx] - activations[:, neuron_idx]

        # Compute residual stream deltas
        res_deltas = activation_deltas.unsqueeze(-1) * self.model.W_out[self.layer_idx]
        res_stream = cache[self.get_act_name("resid_post", self.layer_idx)][0]

        # Apply the deltas to the residual stream
        updated_res_stream = res_stream + res_deltas.sum(dim=1)

        # Apply layer normalization
        normalized_res_stream = self.model.ln_final(updated_res_stream.unsqueeze(0))[0]

        # Project to logit space
        ablated_logits = normalized_res_stream @ self.model.W_U + self.model.b_U

        # Compute ablated loss
        ablated_loss = self.model.loss_fn(ablated_logits.unsqueeze(0), inp, per_token=True).mean().item()

        # TODO: add neurons based on the different conditions
        # Calculate delta loss (original - ablated)
        # Positive delta means the original loss was higher (neurons are important)
        delta_loss = original_loss - ablated_loss

        return delta_loss

    def ablate_and_record(self, neuron_group: list[int], batch_indices: list[int]) -> pd.DataFrame:
        """Ablate neurons in the specified group and record results."""
        results = []

        for batch_idx in batch_indices:
            # Get the input sequence
            tok_seq = self.tokenized_data["tokens"][batch_idx]
            inp = tok_seq.unsqueeze(0).to(self.device)

            # Get original loss and cache
            self.model.reset_hooks()
            original_logits, cache = self.model.run_with_cache(inp)
            original_loss_per_token = self.model.loss_fn(original_logits, inp, per_token=True)[0].cpu().numpy()

            # Get neuron activations
            activations = cache[self.get_act_name("post", self.layer_idx)][0]

            # Calculate mean activation values
            neuron_means = activations.mean(dim=0)

            # Create activation deltas for the specified neurons
            activation_deltas = torch.zeros_like(activations)
            for neuron_idx in neuron_group:
                activation_deltas[:, neuron_idx] = neuron_means[neuron_idx] - activations[:, neuron_idx]

            # Compute residual stream deltas
            res_deltas = activation_deltas.unsqueeze(-1) * self.model.W_out[self.layer_idx]
            res_stream = cache[self.get_act_name("resid_post", self.layer_idx)][0]

            # Apply the deltas to the residual stream
            updated_res_stream = res_stream + res_deltas.sum(dim=1)

            # Apply layer normalization
            normalized_res_stream = self.model.ln_final(updated_res_stream.unsqueeze(0))[0]

            # Project to logit space
            ablated_logits = normalized_res_stream @ self.model.W_U + self.model.b_U

            # Compute ablated loss
            ablated_loss_per_token = (
                self.model.loss_fn(ablated_logits.unsqueeze(0), inp, per_token=True)[0].cpu().numpy()
            )

            # Get token string representations if available
            if hasattr(self.tokenized_data, "tokens") and self.tokenized_data["tokens"] is not None:
                tokens_str = self.tokenized_data["tokens"][batch_idx]
            else:
                tokens_str = [f"<token_{t.item()}>" for t in tok_seq]

            # Create records for each token
            for i in range(len(tok_seq)):
                results.append(
                    {
                        "batch_idx": batch_idx,
                        "token_idx": i,
                        "token": tokens_str[i] if i < len(tokens_str) else "<unknown>",
                        "original_loss": float(original_loss_per_token[i]),
                        "ablated_loss": float(ablated_loss_per_token[i]),
                        "delta_loss": float(original_loss_per_token[i] - ablated_loss_per_token[i]),
                    }
                )

        return pd.DataFrame(results)
