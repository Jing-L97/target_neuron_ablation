#!/usr/bin/env python
import logging
import pickle
import sys
import typing as t
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering

from neuron_analyzer.load_util import cleanup

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


#######################################################################################################
# Functions applying search strategy
#######################################################################################################


@dataclass
class SearchResult:
    neurons: list[int]
    delta_loss: float
    is_target_size: bool = False


class NeuronGroupSearch:
    def __init__(
        self,
        neurons: list[int],
        evaluator,
        target_size: int,
        individual_delta_loss: list[float] | None = None,
        cache_dir: str | Path | None = None,
    ):
        self.neurons = neurons
        self.evaluator = evaluator  # the class should be properly initialized
        self.target_size = min(target_size, len(neurons))
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True, parents=True)

        self.individual_delta_loss = (
            individual_delta_loss if individual_delta_loss is not None else self._compute_individual_scores()
        )

    def _compute_individual_scores(self) -> list[float]:
        return [self.evaluator.evaluate_neuron_group([n]) for n in self.neurons]

    def _evaluate_group(self, group: list[int]) -> float:
        return self.evaluator.evaluate_neuron_group(group)

    def _get_sorted_neurons(self) -> list[tuple[int, float]]:
        return sorted(
            [(i, self.individual_delta_loss[i]) for i in range(len(self.neurons))],
            key=lambda x: x[1],
            reverse=True,
        )

    def _save_search_state(self, method: str, state: dict) -> None:
        if not self.cache_dir:
            return
        path = self.cache_dir / f"{method}_search_state.pkl"

        with open(path, "wb") as f:
            pickle.dump(state, f)

    def _load_search_state(self, method: str) -> dict | None:
        if not self.cache_dir:
            return None
        path = self.cache_dir / f"{method}_search_state.pkl"
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)

        return None

    def _ensure_target_size_group(self, result: SearchResult) -> tuple[SearchResult, SearchResult]:
        """Ensures we have a group with the target size."""
        best_result = result

        # If result already has target size, both results are the same
        if len(result.neurons) == self.target_size:
            target_size_result = SearchResult(
                neurons=result.neurons.copy(), delta_loss=result.delta_loss, is_target_size=True
            )
            return best_result, target_size_result

        # Create a target size group
        if len(result.neurons) < self.target_size:
            # Add more neurons to reach target size
            sorted_neurons = self._get_sorted_neurons()

            # Find neurons that aren't already in the result
            available_neurons = []
            for idx, _ in sorted_neurons:
                neuron = self.neurons[idx]
                if neuron not in result.neurons:
                    available_neurons.append(neuron)

            # Add neurons until we reach target size
            neurons_to_add = min(self.target_size - len(result.neurons), len(available_neurons))
            target_size_group = result.neurons.copy()
            target_size_group.extend(available_neurons[:neurons_to_add])

            # Evaluate the new group
            delta_loss = self._evaluate_group(target_size_group)
            target_size_result = SearchResult(neurons=target_size_group, delta_loss=delta_loss, is_target_size=True)

        else:
            # Sort neurons by individual importance
            neuron_scores = [(n, self.individual_delta_loss[self.neurons.index(n)]) for n in result.neurons]
            neuron_scores.sort(key=lambda x: x[1], reverse=True)

            # Keep the top neurons
            target_size_group = [n for n, _ in neuron_scores[: self.target_size]]
            delta_loss = self._evaluate_group(target_size_group)
            target_size_result = SearchResult(neurons=target_size_group, delta_loss=delta_loss, is_target_size=True)

        return best_result, target_size_result

    def progressive_beam_search(self, beam_width: int = 2) -> tuple[SearchResult, SearchResult]:
        """Progressive beam search for finding neuron groups."""
        state = self._load_search_state("progressive_beam")

        # Check if state exists AND is marked as completed AND contains both result keys
        if state and state.get("completed") and "best_result" in state and "target_size_result" in state:
            best_result = SearchResult(**state["best_result"])
            target_size_result = SearchResult(**state["target_size_result"])
            return best_result, target_size_result

        # If we have a partial state but it's not completed
        if state and not state.get("completed"):
            current_beam = state.get("current_beam", [])
            start_size = state.get("next_size", beam_width)
            best_overall = state.get("best_overall")
            best_per_size = state.get("best_per_size", {})
        else:
            # Start from scratch
            sorted_indices = self._get_sorted_neurons()
            current_beam = []
            for i in range(min(beam_width, len(sorted_indices))):
                idx, _ = sorted_indices[i]
                neuron = self.neurons[idx]
                delta_loss = self.individual_delta_loss[idx]
                current_beam.append(({neuron}, delta_loss))

            # Initialize tracking variables
            start_size = 2
            best_overall = None
            best_per_size = (
                {1: {"neurons": [self.neurons[sorted_indices[0][0]]], "delta_loss": sorted_indices[0][1]}}
                if sorted_indices
                else {}
            )

        # Progressive beam search
        for size in range(start_size, self.target_size + 1):
            candidates = []
            sorted_indices = self._get_sorted_neurons()

            for group, _ in current_beam:
                for idx, _ in sorted_indices:
                    neuron = self.neurons[idx]
                    if neuron not in group:
                        new_group = group.union({neuron})
                        if any(g == new_group for g, _ in candidates):
                            continue
                        delta_loss = self._evaluate_group(list(new_group))
                        candidates.append((new_group, delta_loss))
                        if len(candidates) % 10 == 0:
                            self._save_search_state(
                                "progressive_beam",
                                {
                                    "current_beam": current_beam,
                                    "next_size": size,
                                    "best_overall": best_overall,
                                    "best_per_size": best_per_size,
                                    "completed": False,
                                },
                            )

            if not candidates:
                break

            candidates.sort(key=lambda x: x[1], reverse=True)
            current_beam = candidates[:beam_width]

            # Update best for current size
            if current_beam:
                best_candidate = max(current_beam, key=lambda x: x[1])
                best_per_size[size] = {
                    "neurons": list(best_candidate[0]),
                    "delta_loss": best_candidate[1],
                }

                # Update best overall if needed
                if best_overall is None or best_candidate[1] > best_overall["delta_loss"]:
                    best_overall = {
                        "neurons": list(best_candidate[0]),
                        "delta_loss": best_candidate[1],
                    }

        # Create the best overall result
        best_result = None
        if best_overall:
            best_result = SearchResult(neurons=best_overall["neurons"], delta_loss=best_overall["delta_loss"])
        elif current_beam:
            best_group = max(current_beam, key=lambda x: x[1])
            best_result = SearchResult(neurons=list(best_group[0]), delta_loss=best_group[1])
        else:
            # Fallback to empty result
            best_result = SearchResult(neurons=[], delta_loss=0.0)

        # Get target size result
        target_size_result = None
        if self.target_size in best_per_size:
            # We have a result with exactly the target size
            target_size_result = SearchResult(
                neurons=best_per_size[self.target_size]["neurons"],
                delta_loss=best_per_size[self.target_size]["delta_loss"],
                is_target_size=True,
            )
        else:
            # Need to create a target size group
            _, target_size_result = self._ensure_target_size_group(best_result)

        # Save final state
        if self.cache_dir:
            self._save_search_state(
                "progressive_beam",
                {
                    "current_beam": current_beam,
                    "next_size": self.target_size + 1,
                    "best_overall": best_overall,
                    "best_per_size": best_per_size,
                    "best_result": best_result.__dict__,
                    "target_size_result": target_size_result.__dict__,
                    "completed": True,
                },
            )
        cleanup()
        return best_result, target_size_result

    def hierarchical_cluster_search(
        self, n_clusters: int = 5, expansion_factor: int = 3
    ) -> tuple[SearchResult, SearchResult]:
        """Hierarchical clustering search for finding neuron groups."""
        state = self._load_search_state("hierarchical_cluster")
        if state and state.get("completed") and "best_result" in state and "target_size_result" in state:
            best_result = SearchResult(**state["best_result"])
            target_size_result = SearchResult(**state["target_size_result"])
            return best_result, target_size_result

        features = np.array(self.individual_delta_loss).reshape(-1, 1)
        if features.std() > 0:
            features = (features - features.mean()) / features.std()

        clustering = AgglomerativeClustering(n_clusters=min(n_clusters, len(self.neurons)))
        cluster_labels = clustering.fit_predict(features)

        clusters = defaultdict(list)
        for i, neuron_idx in enumerate(range(len(self.neurons))):
            clusters[cluster_labels[i]].append(neuron_idx)

        representatives = []
        for cluster_id, cluster_neurons in clusters.items():
            sorted_cluster = sorted(
                [(self.neurons[idx], self.individual_delta_loss[idx]) for idx in cluster_neurons],
                key=lambda x: x[1],
                reverse=True,
            )
            representatives.extend([n for n, _ in sorted_cluster[:expansion_factor]])

        # Track best result of any size
        best_result = None

        # Handle case where representatives are fewer than target size
        if len(representatives) <= self.target_size:
            delta_loss = self._evaluate_group(representatives)
            best_result = SearchResult(neurons=representatives, delta_loss=delta_loss)

            # Create target size result
            _, target_size_result = self._ensure_target_size_group(best_result)

            if self.cache_dir:
                self._save_search_state(
                    "hierarchical_cluster",
                    {
                        "best_result": best_result.__dict__,
                        "target_size_result": target_size_result.__dict__,
                        "completed": True,
                    },
                )

            return best_result, target_size_result

        # Sort representatives by importance
        sorted_reps = sorted(
            [(n, self.individual_delta_loss[self.neurons.index(n)]) for n in representatives],
            key=lambda x: x[1],
            reverse=True,
        )

        # Initialize with top representatives
        current_group = [n for n, _ in sorted_reps[: min(5, self.target_size)]]
        best_per_size = {len(current_group): (current_group.copy(), self._evaluate_group(current_group))}

        # Keep track of best group of any size
        best_group = current_group.copy()
        best_score = self._evaluate_group(best_group)

        # Continue adding neurons
        while len(current_group) < self.target_size:
            best_candidate = None
            best_candidate_score = -float("inf")

            for n, _ in sorted_reps:
                if n in current_group:
                    continue

                test_group = current_group + [n]
                delta_loss = self._evaluate_group(test_group)

                # Update best candidate
                if delta_loss > best_candidate_score:
                    best_candidate = n
                    best_candidate_score = delta_loss

                # Update best overall if better
                if delta_loss > best_score:
                    best_group = test_group.copy()
                    best_score = delta_loss

                if self.cache_dir and len(current_group) % 2 == 0:
                    self._save_search_state(
                        "hierarchical_cluster",
                        {
                            "current_group": current_group,
                            "best_per_size": best_per_size,
                            "best_group": best_group,
                            "best_score": best_score,
                            "completed": False,
                        },
                    )

            if best_candidate:
                current_group.append(best_candidate)
                # Store best for this size
                best_per_size[len(current_group)] = (current_group.copy(), best_candidate_score)
            else:
                break

        # Create best overall result
        best_result = SearchResult(neurons=best_group, delta_loss=best_score)

        # Create target size result
        if len(current_group) == self.target_size:
            target_loss = self._evaluate_group(current_group)
            target_size_result = SearchResult(neurons=current_group, delta_loss=target_loss, is_target_size=True)
        else:
            _, target_size_result = self._ensure_target_size_group(best_result)

        # Save state
        if self.cache_dir:
            self._save_search_state(
                "hierarchical_cluster",
                {
                    "best_result": best_result.__dict__,
                    "target_size_result": target_size_result.__dict__,
                    "completed": True,
                },
            )
        cleanup()
        return best_result, target_size_result

    def iterative_pruning(self) -> tuple[SearchResult, SearchResult]:
        """Iterative pruning search for finding neuron groups."""
        state = self._load_search_state("iterative_pruning")
        if state and state.get("completed") and "best_result" in state and "target_size_result" in state:
            best_result = SearchResult(**state["best_result"])
            target_size_result = SearchResult(**state["target_size_result"])
            return best_result, target_size_result

        # Initialize variables
        current_group = state.get("current_group") if state and state.get("current_group") else self.neurons.copy()
        best_per_size = state.get("best_per_size", {})

        # Track best group of any size
        best_group = current_group.copy()
        best_score = self._evaluate_group(best_group)

        # Start with full set and record initial score
        initial_score = self._evaluate_group(current_group)
        best_per_size[len(current_group)] = (current_group.copy(), initial_score)

        # Prune until we reach target size
        while len(current_group) > self.target_size:
            worst_neuron = None
            best_pruned_score = -float("inf")

            # Try removing each neuron and find the one that hurts performance least
            for n in current_group:
                test_group = [x for x in current_group if x != n]
                delta_loss = self._evaluate_group(test_group)

                if delta_loss > best_pruned_score:
                    best_pruned_score = delta_loss
                    worst_neuron = n

                # Update best overall if better
                if delta_loss > best_score:
                    best_group = test_group.copy()
                    best_score = delta_loss

            # Remove worst neuron
            if worst_neuron is not None:
                current_group.remove(worst_neuron)
                # Store best for this size
                best_per_size[len(current_group)] = (current_group.copy(), best_pruned_score)

                if self.cache_dir and len(current_group) % 5 == 0:
                    self._save_search_state(
                        "iterative_pruning",
                        {
                            "current_group": current_group,
                            "best_per_size": best_per_size,
                            "best_group": best_group,
                            "best_score": best_score,
                            "completed": False,
                        },
                    )
            else:
                break

        # Create best overall result
        best_result = SearchResult(neurons=best_group, delta_loss=best_score)

        # Create target size result
        if self.target_size in best_per_size:
            target_group, target_loss = best_per_size[self.target_size]
            target_size_result = SearchResult(neurons=target_group, delta_loss=target_loss, is_target_size=True)
        else:
            _, target_size_result = self._ensure_target_size_group(best_result)

        # Save state
        if self.cache_dir:
            self._save_search_state(
                "iterative_pruning",
                {
                    "best_result": best_result.__dict__,
                    "target_size_result": target_size_result.__dict__,
                    "completed": True,
                },
            )
        cleanup()
        return best_result, target_size_result

    def importance_weighted_sampling(
        self, n_iterations: int = 100, learning_rate: float = 0.1, checkpoint_freq: int = 20
    ) -> tuple[SearchResult, SearchResult]:
        """Find neuron groups using importance weighted sampling."""
        method = "importance_weighted"
        state = self._load_search_state(method)

        if state and state.get("completed") and "best_result" in state and "target_size_result" in state:
            best_result = SearchResult(**state["best_result"])
            target_size_result = SearchResult(**state["target_size_result"])
            return best_result, target_size_result

        # Initialize variables
        if state and not state.get("completed"):
            weights = state["weights"]
            best_group = state["best_group"]
            best_score = state["best_score"]
            best_target_size_group = state.get("best_target_size_group")
            best_target_size_score = state.get("best_target_size_score", -float("inf"))
            start_iteration = state["iteration"]
        else:
            # Start from scratch
            scores = np.array(self.individual_delta_loss)
            min_score = scores.min()
            weights = scores - min_score + 1e-6
            weights = weights / weights.sum()
            best_group = None
            best_score = -float("inf")
            best_target_size_group = None
            best_target_size_score = -float("inf")
            start_iteration = 0

        # Run the sampling
        for i in range(start_iteration, n_iterations):
            if len(self.neurons) <= self.target_size:
                sampled_indices = np.arange(len(self.neurons))
            else:
                sampled_indices = np.random.choice(len(self.neurons), size=self.target_size, replace=False, p=weights)

            sampled_group = [self.neurons[idx] for idx in sampled_indices]
            score = self._evaluate_group(sampled_group)

            # Update best overall
            if score > best_score:
                best_group = sampled_group.copy()
                best_score = score

            # Update best target size (if exactly target size)
            if len(sampled_group) == self.target_size and score > best_target_size_score:
                best_target_size_group = sampled_group.copy()
                best_target_size_score = score

            # Update weights
            for idx in sampled_indices:
                weights[idx] += learning_rate * score
            weights = weights / weights.sum()

            # Checkpoint
            if self.cache_dir and (i + 1) % checkpoint_freq == 0:
                self._save_search_state(
                    method,
                    {
                        "iteration": i + 1,
                        "weights": weights,
                        "best_group": best_group,
                        "best_score": best_score,
                        "best_target_size_group": best_target_size_group,
                        "best_target_size_score": best_target_size_score,
                        "completed": False,
                    },
                )

        # Create best overall result
        best_result = SearchResult(
            neurons=best_group if best_group else [], delta_loss=best_score if best_score > -float("inf") else 0.0
        )

        # Create target size result
        if best_target_size_group:
            target_size_result = SearchResult(
                neurons=best_target_size_group, delta_loss=best_target_size_score, is_target_size=True
            )
        else:
            _, target_size_result = self._ensure_target_size_group(best_result)

        # Save state
        if self.cache_dir:
            self._save_search_state(
                method,
                {
                    "best_result": best_result.__dict__,
                    "target_size_result": target_size_result.__dict__,
                    "completed": True,
                },
            )
        cleanup()
        return best_result, target_size_result

    def hybrid_search(
        self, n_clusters: int = 5, expansion_factor: int = 3, beam_width: int = 2
    ) -> tuple[SearchResult, SearchResult]:
        """Hybrid clustering and beam search for finding neuron groups."""
        state = self._load_search_state("hybrid")
        if state and state.get("completed") and "best_result" in state and "target_size_result" in state:
            best_result = SearchResult(**state["best_result"])
            target_size_result = SearchResult(**state["target_size_result"])
            return best_result, target_size_result

        # Step 1: Get representatives using clustering
        features = np.array(self.individual_delta_loss).reshape(-1, 1)
        if features.std() > 0:
            features = (features - features.mean()) / features.std()

        clustering = AgglomerativeClustering(n_clusters=min(n_clusters, len(self.neurons)))
        cluster_labels = clustering.fit_predict(features)

        clusters = defaultdict(list)
        for i, neuron_idx in enumerate(range(len(self.neurons))):
            clusters[cluster_labels[i]].append(neuron_idx)

        representatives = []
        for cluster_id, cluster_neurons in clusters.items():
            sorted_cluster = sorted(
                [(self.neurons[idx], self.individual_delta_loss[idx]) for idx in cluster_neurons],
                key=lambda x: x[1],
                reverse=True,
            )
            representatives.extend([n for n, _ in sorted_cluster[:expansion_factor]])

        # Step 2: Run beam search over reduced set
        reduced_search = NeuronGroupSearch(
            neurons=representatives,
            evaluator=self.evaluator,
            target_size=self.target_size,
            individual_delta_loss=[self.individual_delta_loss[self.neurons.index(n)] for n in representatives],
        )
        best_result, target_size_result = reduced_search.progressive_beam_search(beam_width=beam_width)

        # Save state
        if self.cache_dir:
            self._save_search_state(
                "hybrid",
                {
                    "best_result": best_result.__dict__,
                    "target_size_result": target_size_result.__dict__,
                    "completed": True,
                },
            )
        cleanup()
        return best_result, target_size_result

    def run_all_methods(self) -> dict[str, tuple[SearchResult, SearchResult]]:
        """Run all search methods and return the results."""
        results = {}

        # Run all methods with error handling
        try:
            results["progressive_beam"] = self.progressive_beam_search()
        except Exception as e:
            logger.error(f"Error in progressive_beam_search: {e}")
            empty_result = SearchResult(neurons=[], delta_loss=0.0)
            results["progressive_beam"] = (empty_result, empty_result)

        try:
            results["hierarchical_cluster"] = self.hierarchical_cluster_search()
        except Exception as e:
            logger.error(f"Error in hierarchical_cluster_search: {e}")
            empty_result = SearchResult(neurons=[], delta_loss=0.0)
            results["hierarchical_cluster"] = (empty_result, empty_result)

        try:
            results["iterative_pruning"] = self.iterative_pruning()
        except Exception as e:
            logger.error(f"Error in iterative_pruning: {e}")
            empty_result = SearchResult(neurons=[], delta_loss=0.0)
            results["iterative_pruning"] = (empty_result, empty_result)

        try:
            results["importance_weighted"] = self.importance_weighted_sampling()
        except Exception as e:
            logger.error(f"Error in importance_weighted_sampling: {e}")
            empty_result = SearchResult(neurons=[], delta_loss=0.0)
            results["importance_weighted"] = (empty_result, empty_result)

        try:
            results["hybrid"] = self.hybrid_search()
        except Exception as e:
            logger.error(f"Error in hybrid_search: {e}")
            empty_result = SearchResult(neurons=[], delta_loss=0.0)
            results["hybrid"] = (empty_result, empty_result)

        return results

    def get_best_result(self) -> tuple[str, tuple[SearchResult, SearchResult]]:
        """Run all methods and return the best method and its results."""
        results = self.run_all_methods()
        # Find the best method (highest delta loss from the 'best' result, not target size result)
        best_method_name = max(results.keys(), key=lambda x: results[x][0].delta_loss)
        total_method_name = max(results.keys(), key=lambda x: results[x][1].delta_loss)
        return {"best": results[best_method_name][0], "target_size": results[total_method_name][1], "toal": results}
