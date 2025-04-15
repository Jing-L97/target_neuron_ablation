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
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(self.eval_cache, f)
        except Exception:
            # If saving fails, we continue without caching
            pass

    def get_act_name(self, prefix: str, layer_idx: int) -> str:
        """Get the activation name for a layer."""
        return f"blocks.{layer_idx}.{prefix}"

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
        available_batches = list(range(len(self.tokenized_data["str_tokens"])))
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
        tok_seq = self.tokenized_data["str_tokens"][batch_idx]
        if isinstance(tok_seq, str):
            tok_seq = self.tokenizer(tok_seq, return_tensors="pt")["input_ids"]
            logger.info("Tokenizing the input string")

        # dimenison handling for the right shape: [batch_size, sequence_length]
        # Make sure we have the right shape: [batch_size, sequence_length]
        inp = tok_seq.unsqueeze(0).to(self.device) if tok_seq.dim() == 1 else tok_seq.to(self.device)
        # inp = tok_seq.unsqueeze(0).to(self.device)

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
            tok_seq = self.tokenized_data["str_tokens"][batch_idx]
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
            if hasattr(self.tokenized_data, "str_tokens") and self.tokenized_data["str_tokens"] is not None:
                tokens_str = self.tokenized_data["str_tokens"][batch_idx]
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
        self.evaluator = evaluator  # the class should be properly initilialized
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

    def progressive_beam_search(self, beam_width: int = 2) -> SearchResult:
        state = self._load_search_state("progressive_beam")
        if state and state.get("completed"):
            return SearchResult(**state["result"])

        if state:
            current_beam = state["current_beam"]
            start_size = state["next_size"]
            best_result = state["best_result"]
        else:
            sorted_indices = self._get_sorted_neurons()
            current_beam = []
            for i in range(min(beam_width, len(sorted_indices))):
                idx, _ = sorted_indices[i]
                neuron = self.neurons[idx]
                delta_loss = self.individual_delta_loss[idx]
                current_beam.append(({neuron}, delta_loss))
            start_size = 2
            best_result = None

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
                                    "best_result": best_result,
                                    "completed": False,
                                },
                            )

            candidates.sort(key=lambda x: x[1], reverse=True)
            current_beam = candidates[:beam_width]

            if current_beam:
                best_candidate = max(current_beam, key=lambda x: x[1])
                if best_result is None or best_candidate[1] > best_result["delta_loss"]:
                    best_result = {
                        "neurons": list(best_candidate[0]),
                        "delta_loss": best_candidate[1],
                    }

        if self.cache_dir:
            final_result = (
                best_result
                if best_result
                else {
                    "neurons": list(current_beam[0][0]) if current_beam else [],
                    "delta_loss": current_beam[0][1] if current_beam else 0.0,
                }
            )
            self._save_search_state(
                "progressive_beam",
                {
                    "current_beam": current_beam,
                    "next_size": self.target_size + 1,
                    "best_result": final_result,
                    "completed": True,
                },
            )
            return SearchResult(**final_result)

        if not current_beam:
            return SearchResult(neurons=[], delta_loss=0.0)
        best_group = max(current_beam, key=lambda x: x[1])
        return SearchResult(neurons=list(best_group[0]), delta_loss=best_group[1])

    def hierarchical_cluster_search(self, n_clusters: int = 5, expansion_factor: int = 3) -> SearchResult:
        state = self._load_search_state("hierarchical_cluster")
        if state and state.get("completed"):
            return SearchResult(**state["result"])

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

        if len(representatives) <= self.target_size:
            delta_loss = self._evaluate_group(representatives)
            result = {"neurons": representatives, "delta_loss": delta_loss}
            self._save_search_state("hierarchical_cluster", {"result": result, "completed": True})
            return SearchResult(**result)

        sorted_reps = sorted(
            [(n, self.individual_delta_loss[self.neurons.index(n)]) for n in representatives],
            key=lambda x: x[1],
            reverse=True,
        )
        current_group = [n for n, _ in sorted_reps[: min(5, self.target_size)]]

        while len(current_group) < self.target_size:
            best_candidate = None
            best_score = -float("inf")
            for n, _ in sorted_reps:
                if n in current_group:
                    continue
                test_group = current_group + [n]
                delta_loss = self._evaluate_group(test_group)
                if delta_loss > best_score:
                    best_candidate = n
                    best_score = delta_loss
                if self.cache_dir and len(current_group) % 2 == 0:
                    self._save_search_state(
                        "hierarchical_cluster",
                        {
                            "current_group": current_group,
                            "completed": False,
                        },
                    )
            if best_candidate:
                current_group.append(best_candidate)
            else:
                break

        delta_loss = self._evaluate_group(current_group)
        result = {"neurons": current_group, "delta_loss": delta_loss}
        self._save_search_state("hierarchical_cluster", {"result": result, "completed": True})
        return SearchResult(**result)

    def iterative_pruning(self) -> SearchResult:
        state = self._load_search_state("iterative_pruning")
        if state and state.get("completed"):
            return SearchResult(**state["result"])

        current_group = state["current_group"] if state else self.neurons.copy()

        while len(current_group) > self.target_size:
            worst_neuron = None
            best_score = -float("inf")
            for n in current_group:
                test_group = [x for x in current_group if x != n]
                delta_loss = self._evaluate_group(test_group)
                if delta_loss > best_score:
                    best_score = delta_loss
                    worst_neuron = n
            if worst_neuron is not None:
                current_group.remove(worst_neuron)
                if self.cache_dir and len(current_group) % 5 == 0:
                    self._save_search_state(
                        "iterative_pruning",
                        {
                            "current_group": current_group,
                            "completed": False,
                        },
                    )
            else:
                break

        delta_loss = self._evaluate_group(current_group)
        result = {"neurons": current_group, "delta_loss": delta_loss}
        self._save_search_state("iterative_pruning", {"result": result, "completed": True})
        return SearchResult(**result)

    def hybrid_search(self, n_clusters: int = 5, expansion_factor: int = 3, beam_width: int = 2) -> SearchResult:
        state = self._load_search_state("hybrid")
        if state and state.get("completed"):
            return SearchResult(**state["result"])

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
        result = reduced_search.progressive_beam_search(beam_width=beam_width)

        if self.cache_dir:
            self._save_search_state("hybrid", {"result": result.__dict__, "completed": True})

        return result

    def run_all_methods(self) -> dict[str, SearchResult]:
        """Run all search methods and return the results."""
        results = {}

        # Run all methods
        results["progressive_beam"] = self.progressive_beam_search()
        results["hierarchical_cluster"] = self.hierarchical_cluster_search()
        results["iterative_pruning"] = self.iterative_pruning()
        results["importance_weighted"] = self.importance_weighted_sampling()
        results["hybrid"] = self.hybrid_search()

        return results

    def get_best_result(self) -> tuple[str, SearchResult]:
        """Run all methods and return the best result."""
        results = self.run_all_methods()
        # Find the best method (highest delta loss)
        best_method = max(results.items(), key=lambda x: x[1].delta_loss)
        return best_method, results
