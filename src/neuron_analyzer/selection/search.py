#!/usr/bin/env python
import logging
import pickle
import random
import sys
import typing as t
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering

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
        device: str,
        layer_idx: int,
        cache_dir: str | Path | None = None,
    ):
        """Initialize the neuron group evaluator.

        Args:
            model: Neural network model to evaluate
            tokenized_data: Dictionary containing tokenized sequences
            device: Device to run computations on ('cpu' or 'cuda')
            layer_idx: Layer index to ablate neurons from
            cache_dir: Optional directory to cache evaluation results

        """
        self.model = model
        self.tokenized_data = tokenized_data
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
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(self.eval_cache, f)
        except Exception:
            # If saving fails, we continue without caching
            pass

    def get_act_name(self, prefix: str, layer_idx: int) -> str:
        """Get the activation name for a layer.

        Args:
            prefix: Type of activation ('post', 'pre', 'resid_post', etc.)
            layer_idx: Layer index

        Returns:
            Formatted activation name string

        """
        return f"blocks.{layer_idx}.{prefix}"

    def evaluate_neuron_group(self, neuron_group: list[int], sample_batches: int = 3) -> float:
        """Evaluate the impact of a neuron group by measuring delta loss.

        Args:
            neuron_group: List of neuron indices to ablate
            sample_batches: Number of batches to sample for evaluation

        Returns:
            Average delta loss (higher means more important neurons)

        """
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
        """Compute delta loss for a specific batch and neuron group.

        Args:
            batch_idx: Index of the batch to evaluate
            neuron_group: List of neuron indices to ablate

        Returns:
            Delta loss value (original_loss - ablated_loss)

        """
        # Get the input sequence
        tok_seq = self.tokenized_data["tokens"][batch_idx]
        inp = tok_seq.unsqueeze(0).to(self.device)

        # Get original loss
        self.model.reset_hooks()
        original_logits, cache = self.model.run_with_cache(inp)
        original_loss = self.model.loss_fn(original_logits, inp, per_token=True).mean().item()

        # Get neuron activations
        activations = cache[self.get_act_name("post", self.layer_idx)][0]

        # Calculate mean activation values across the dataset (or use a pre-computed value)
        # This is approximate - in a full implementation you'd want to pre-compute this
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

        # Calculate delta loss (original - ablated)
        # Positive delta means the original loss was higher (neurons are important)
        delta_loss = original_loss - ablated_loss

        return delta_loss

    def ablate_and_record(self, neuron_group: list[int], batch_indices: list[int]) -> pd.DataFrame:
        """Ablate neurons in the specified group and record results.

        Args:
            neuron_group: List of neuron indices to ablate
            batch_indices: List of batch indices to ablate

        Returns:
            DataFrame with token-level results

        """
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
            if hasattr(self.tokenized_data, "tokens_str") and self.tokenized_data["tokens_str"] is not None:
                tokens_str = self.tokenized_data["tokens_str"][batch_idx]
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


#######################################################################################################
# Functions applying search strategy
#######################################################################################################


@dataclass
class SearchResult:
    """Result of a neuron group search."""

    neurons: list[int]
    delta_loss: float


class NeuronGroupSearch:
    """Class for finding optimal neuron groups using delta loss as the optimization metric."""

    def __init__(
        self,
        neurons: list[int],
        evaluation_fn: t.Callable[[list[int]], float],
        target_size: int,
        individual_delta_loss: list[float] | None = None,
    ):
        """Initialize the neuron group search.

        Args:
            neurons: List of neuron IDs to search from
            evaluation_fn: Function that evaluates a list of neurons and returns delta loss
            target_size: The desired size of the optimal neuron group
            individual_delta_loss: Optional precomputed delta loss for individual neurons

        """
        self.neurons = neurons
        self.evaluation_fn = evaluation_fn
        self.target_size = min(target_size, len(neurons))

        # Compute individual scores if not provided
        if individual_delta_loss is None:
            self._compute_individual_scores()
        else:
            self.individual_delta_loss = individual_delta_loss

    def _compute_individual_scores(self) -> None:
        """Compute delta loss for individual neurons."""
        self.individual_delta_loss = []

        for neuron in self.neurons:
            delta_loss = self.evaluation_fn([neuron])
            self.individual_delta_loss.append(delta_loss)

    def _evaluate_group(self, group: list[int]) -> float:
        """Evaluate a neuron group using the evaluation function."""
        return self.evaluation_fn(group)

    def _get_sorted_neurons(self) -> list[tuple[int, float]]:
        """Get neurons sorted by their delta loss scores."""
        scores = [(i, self.individual_delta_loss[i]) for i in range(len(self.neurons))]
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def progressive_beam_search(self, beam_width: int = 2) -> SearchResult:
        """Find the best neuron group using progressive beam search.

        Args:
            beam_width: Number of candidate groups to keep during search

        Returns:
            SearchResult with the best performing group of neurons

        """
        # Get individual neuron scores
        sorted_indices = self._get_sorted_neurons()

        # Initialize beam with top individual neurons
        current_beam = []
        for i in range(min(beam_width, len(sorted_indices))):
            idx, _ = sorted_indices[i]
            neuron = self.neurons[idx]
            delta_loss = self.individual_delta_loss[idx]
            current_beam.append(({neuron}, delta_loss))

        # Expand to larger groups
        for size in range(2, self.target_size + 1):
            candidates = []

            for group, _ in current_beam:
                # Try adding each neuron not already in the group
                for idx, _ in sorted_indices:
                    neuron = self.neurons[idx]
                    if neuron not in group:
                        new_group = group.union({neuron})

                        # Skip if already evaluated
                        if any(g == new_group for g, _ in candidates):
                            continue

                        # Evaluate the new group
                        delta_loss = self._evaluate_group(list(new_group))
                        candidates.append((new_group, delta_loss))

            # Sort candidates by delta loss
            candidates.sort(key=lambda x: x[1], reverse=True)

            # Keep top beam_width candidates
            current_beam = candidates[:beam_width]

        # Return the best group of the target size
        best_group = max(current_beam, key=lambda x: x[1])
        return SearchResult(neurons=list(best_group[0]), delta_loss=best_group[1])

    def hierarchical_cluster_search(self, n_clusters: int = 5, expansion_factor: int = 3) -> SearchResult:
        """Find the best neuron group using hierarchical clustering-based search.

        Args:
            n_clusters: Number of clusters to form
            expansion_factor: Number of top neurons to take from each cluster

        Returns:
            SearchResult with the best performing group of neurons

        """
        # Create feature vectors for clustering using delta loss
        features = np.array(self.individual_delta_loss).reshape(-1, 1)

        # Normalize features
        if features.std() > 0:  # Avoid division by zero
            features = (features - features.mean()) / features.std()

        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(n_clusters=min(n_clusters, len(self.neurons)))
        cluster_labels = clustering.fit_predict(features)

        # Group neurons by cluster
        clusters = defaultdict(list)
        for i, neuron_idx in enumerate(range(len(self.neurons))):
            clusters[cluster_labels[i]].append(neuron_idx)

        # Find representatives from each cluster (highest individual scores)
        representatives = []
        for cluster_id, cluster_neurons in clusters.items():
            # Sort neurons in this cluster by delta loss
            sorted_cluster = sorted(
                [(self.neurons[idx], self.individual_delta_loss[idx]) for idx in cluster_neurons],
                key=lambda x: x[1],
                reverse=True,
            )

            # Take top representatives from each cluster
            representatives.extend([n for n, _ in sorted_cluster[:expansion_factor]])

        # If we have fewer representatives than target_size, use all representatives
        if len(representatives) <= self.target_size:
            delta_loss = self._evaluate_group(representatives)
            return SearchResult(neurons=representatives, delta_loss=delta_loss)

        # Start with greedy selection based on individual scores
        sorted_reps = sorted(
            [(n, self.individual_delta_loss[self.neurons.index(n)]) for n in representatives],
            key=lambda x: x[1],
            reverse=True,
        )

        # Start with top scoring neurons
        current_group = [n for n, _ in sorted_reps[: min(5, self.target_size)]]

        # Complete the group
        while len(current_group) < self.target_size:
            best_candidate = None
            best_candidate_score = -float("inf")

            for n, _ in [(n, s) for n, s in sorted_reps if n not in current_group]:
                test_group = current_group + [n]
                delta_loss = self._evaluate_group(test_group)

                if delta_loss > best_candidate_score:
                    best_candidate = n
                    best_candidate_score = delta_loss

            if best_candidate is not None:
                current_group.append(best_candidate)
            else:
                break

        # Evaluate the final group
        delta_loss = self._evaluate_group(current_group)
        best_group = current_group
        best_score = delta_loss

        # Refine the group with local search
        for _ in range(min(10, len(best_group))):
            for i, n in enumerate(best_group):
                # Try replacing each neuron in the group
                for candidate in [neuron for neuron in self.neurons if neuron not in best_group]:
                    test_group = [x for x in best_group if x != n] + [candidate]
                    delta_loss = self._evaluate_group(test_group)

                    if delta_loss > best_score:
                        best_group = test_group
                        best_score = delta_loss

        return SearchResult(neurons=best_group, delta_loss=best_score)

    def iterative_pruning(self) -> SearchResult:
        """Find the best neuron group using iterative pruning.

        Returns:
            SearchResult with the best performing group of neurons

        """
        # Start with all neurons
        current_group = self.neurons.copy()

        # Iteratively remove neurons until we reach target_size
        while len(current_group) > self.target_size:
            worst_neuron = None
            best_remaining_score = -float("inf")

            # Try removing each neuron and keep the best resulting group
            for n in current_group:
                test_group = [x for x in current_group if x != n]
                delta_loss = self._evaluate_group(test_group)

                # Determine if this is the best group after removal
                if delta_loss > best_remaining_score:
                    best_remaining_score = delta_loss
                    worst_neuron = n

            # Remove the worst neuron
            if worst_neuron is not None:
                current_group.remove(worst_neuron)
            else:
                # This shouldn't happen, but break to avoid infinite loop
                break

        # Evaluate final group
        delta_loss = self._evaluate_group(current_group)

        return SearchResult(neurons=current_group, delta_loss=delta_loss)

    def importance_weighted_sampling(self, n_iterations: int = 100, learning_rate: float = 0.1) -> SearchResult:
        """Find the best neuron group using importance weighted sampling.

        Args:
            n_iterations: Number of sampling iterations
            learning_rate: Rate for updating sampling weights

        Returns:
            SearchResult with the best performing group of neurons

        """
        # Initialize weights from delta loss scores
        weights = np.array(self.individual_delta_loss)

        # Ensure all weights are positive
        min_weight = min(weights)
        weights = weights - min_weight + 0.000001

        # Normalize to probabilities
        weights = weights / np.sum(weights)

        best_group = None
        best_score = -float("inf")

        for _ in range(n_iterations):
            # Sample neurons according to current weights
            if len(self.neurons) <= self.target_size:
                sampled_indices = np.arange(len(self.neurons))
            else:
                sampled_indices = np.random.choice(len(self.neurons), size=self.target_size, replace=False, p=weights)

            sampled_group = [self.neurons[idx] for idx in sampled_indices]

            # Evaluate group
            delta_loss = self._evaluate_group(sampled_group)

            # Update best group
            if delta_loss > best_score:
                best_group = sampled_group
                best_score = delta_loss

                # Update weights for neurons in successful groups
                for idx in sampled_indices:
                    weights[idx] *= 1 + learning_rate

                # Re-normalize
                weights = weights / np.sum(weights)

        return SearchResult(neurons=best_group, delta_loss=best_score)

    def hybrid_search(self, n_init: int = 3, n_refinement: int = 50) -> SearchResult:
        """Find the best neuron group using a hybrid approach.

        Args:
            n_init: Number of initial random groups to try
            n_refinement: Maximum number of refinement iterations

        Returns:
            SearchResult with the best performing group of neurons

        """
        candidate_groups = []

        # 1. Start with top individual neurons (greedy)
        sorted_indices = self._get_sorted_neurons()
        top_neurons = [self.neurons[idx] for idx, _ in sorted_indices[: self.target_size]]
        candidate_groups.append(top_neurons)

        # 2. Start with random initialization (n_init different starts)
        for _ in range(n_init):
            random_group = random.sample(self.neurons, min(len(self.neurons), self.target_size))
            candidate_groups.append(random_group)

        # 3. Start with biased sampling based on individual scores
        for _ in range(n_init):
            # Create probability distribution based on delta loss scores
            probs = np.array(self.individual_delta_loss)
            min_prob = min(probs)
            probs = probs - min_prob + 0.000001
            probs = probs / np.sum(probs)

            # Sample based on probabilities
            sampled_indices = np.random.choice(
                len(self.neurons), size=min(len(self.neurons), self.target_size), replace=False, p=probs
            )
            sampled_group = [self.neurons[idx] for idx in sampled_indices]
            candidate_groups.append(sampled_group)

        # Evaluate all candidate groups
        best_group = None
        best_score = -float("inf")

        for group in candidate_groups:
            delta_loss = self._evaluate_group(group)
            if delta_loss > best_score:
                best_group = group
                best_score = delta_loss

        # Refine the best group
        current_group = best_group.copy()
        current_score = best_score

        for _ in range(n_refinement):
            improved = False

            # Try replacing each neuron in the group
            for i, current_neuron in enumerate(current_group):
                for candidate_neuron in [n for n in self.neurons if n not in current_group]:
                    # Create a new group with one neuron replaced
                    test_group = current_group.copy()
                    test_group[i] = candidate_neuron

                    # Evaluate the new group
                    delta_loss = self._evaluate_group(test_group)

                    # Update if better
                    if delta_loss > current_score:
                        current_group = test_group
                        current_score = delta_loss
                        improved = True

            # Early stopping if no improvement
            if not improved:
                break

        return SearchResult(neurons=current_group, delta_loss=current_score)

    def run_all_methods(self) -> dict[str, SearchResult]:
        """Run all search methods and return the results.

        Returns:
            Dictionary mapping method names to their search results

        """
        results = {}

        # Run all methods
        results["progressive_beam"] = self.progressive_beam_search()
        results["hierarchical_cluster"] = self.hierarchical_cluster_search()
        results["iterative_pruning"] = self.iterative_pruning()
        results["importance_weighted"] = self.importance_weighted_sampling()
        results["hybrid"] = self.hybrid_search()

        return results

    def get_best_result(self) -> tuple[str, SearchResult]:
        """Run all methods and return the best result.

        Returns:
            Tuple of (method_name, search_result) with the best performing method

        """
        results = self.run_all_methods()
        # Find the best method (highest delta loss)
        best_method = max(results.items(), key=lambda x: x[1].delta_loss)
        return best_method
