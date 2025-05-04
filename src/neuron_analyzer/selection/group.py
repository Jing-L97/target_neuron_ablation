#!/usr/bin/env python
import logging
import pickle
import sys
import threading
import time
import traceback
import typing as t
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from queue import Queue

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering
from transformer_lens import utils

from neuron_analyzer.ablation.ablation import ModelAblationAnalyzer
from neuron_analyzer.load_util import cleanup

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
sys.path.append("../")
T = t.TypeVar("T")

#######################################################################################################
# Group eval class


@dataclass
class NeuronEvalResult:
    """Result of a neuron group evaluation."""

    neurons: list[int]
    delta_loss: float


class GroupModelAblationAnalyzer(ModelAblationAnalyzer):
    """Class for analyzing neural network components through group ablations."""

    def __init__(
        self,
        model,
        unigram_distrib,
        tokenized_data,
        unigram_analyzer,
        sel_freq,
        entropy_df: pd.DataFrame,
        layer_idx: int,  # Layer index for neuron groups
        device: str = "cuda",
        k: int = 10,
        chunk_size: int = 40,
        ablation_mode: str = "mean",
        longtail_threshold: float = 0.001,
    ):
        """Initialize the GroupModelAblationAnalyzer."""
        # Initialize with placeholder for components_to_ablate (will be set by neuron_group later)
        super().__init__(
            model,
            unigram_distrib,
            tokenized_data,
            entropy_df,
            components_to_ablate=[],  # Will be set based on neuron_group in evaluate_neuron_group
            device=device,
            k=k,
            chunk_size=chunk_size,
            ablation_mode=ablation_mode,
            longtail_threshold=longtail_threshold,
        )
        self.unigram_analyzer = unigram_analyzer

        self.sel_freq = sel_freq
        self.layer_idx = layer_idx
        self.results = pd.DataFrame()  # Override the dict with a DataFrame for this class
        # Add a model access lock for thread safety
        self.model_lock = threading.Lock()
        logger.info(f"ablate_components: ablate with k = {self.k}, long-tail threshold = {self.longtail_threshold}")

    def build_group_activation_means(self, filtered_entropy_df: pd.DataFrame, neuron_group: list[int]) -> torch.Tensor:
        """Build mean activation values for each neuron in each group."""
        neuron_means = []

        for neuron in neuron_group:
            # Get the mean activation for this neuron
            mean_activation = filtered_entropy_df[f"{self.layer_idx}.{neuron}_activation"].mean()
            neuron_means.append(mean_activation)

            # Log some statistics for debugging
            logger.debug(f"Neuron {neuron}: mean activation: {mean_activation:.4f}")

        # Convert to tensor and add to group means
        group_means = torch.tensor(neuron_means)

        return group_means

    def mean_ablate_components(self, neuron_group: list[int]) -> pd.DataFrame:
        """Perform mean ablation on specified neuron groups."""
        # Sample a set of random batch indices
        try:
            random_sequence_indices = np.random.choice(self.entropy_df.batch.unique(), self.k, replace=True)
        except ValueError as e:
            logger.error(f"Error sampling batch indices: {e}")
            # If there are fewer batches than k, use all available batches
            random_sequence_indices = self.entropy_df.batch.unique()
            logger.info(f"Using all available batches: {len(random_sequence_indices)}")

        # new_entropy_df with only the random sequences
        filtered_entropy_df = self.entropy_df[self.entropy_df.batch.isin(random_sequence_indices)].copy()

        # Build group activation means
        try:
            group_activation_means = self.build_group_activation_means(filtered_entropy_df, neuron_group)
        except Exception as e:
            logger.error(f"Error building group activation means: {e}")
            raise

        # Build frequency vectors
        self.build_vector()

        batch_count = 0
        for batch_n in filtered_entropy_df.batch.unique():
            try:
                # Get original token sequence
                original_tok_seq = self.tokenized_data["tokens"][batch_n]

                # Get positions for this batch from the filtered DataFrame
                batch_df = filtered_entropy_df[filtered_entropy_df.batch == batch_n]
                positions = sorted(batch_df["pos"].unique())

                # Create a mapping from original positions to sequential indices (0, 1, 2, ...)
                pos_to_idx = {pos: idx for idx, pos in enumerate(positions)}

                # Get the model's outputs on the full sequence to preserve context
                self.model.reset_hooks()
                inp = original_tok_seq.unsqueeze(0).to(self.device)
                logits, cache = self.model.run_with_cache(inp)
                logprobs = logits[0, :, :].log_softmax(dim=-1)

                # Extract the unigram projection values only for positions we care about
                unigram_projection_values = self.project_logits(logits)[positions]

                # Process each group
                loss_post_ablation = []
                entropy_post_ablation = []
                loss_post_ablation_with_frozen_unigram = []
                entropy_post_ablation_with_frozen_unigram = []

                # Extract residual stream and activations only for positions we care about
                # We need to get the full data first, then extract positions
                res_stream_full = cache[utils.get_act_name("resid_post", self.layer_idx)][0]
                res_stream = res_stream_full[positions]

                # Extract activations for each neuron in the group, for each position we care about
                previous_activations_full = cache[utils.get_act_name("post", self.layer_idx)][0]
                previous_activations = previous_activations_full[positions][:, neuron_group]

                # Get mean activations for this group
                group_mean = group_activation_means.to(previous_activations.device)

                # Compute activation deltas for all neurons in the group
                activation_deltas = group_mean - previous_activations  # Shape: [filtered_seq_len, num_neurons_in_group]

                # Get the W_out for all neurons in this group
                w_out_group = self.model.W_out[self.layer_idx, neuron_group, :]  # [num_neurons_in_group, d_model]

                # Compute residual stream deltas for all neurons together
                res_deltas = activation_deltas.unsqueeze(-1) * w_out_group

                # Sum across neurons to get the combined effect
                res_deltas_sum = res_deltas.sum(dim=1)

                # Add a batch dimension for processing
                res_deltas_sum = res_deltas_sum.unsqueeze(0)  # [1, filtered_seq_len, d_model]

                # Process in chunks - here we only have one batch since we're doing the combined effect
                updated_res_stream = res_stream + res_deltas_sum[0]  # [filtered_seq_len, d_model]
                updated_res_stream = updated_res_stream.unsqueeze(0)  # Add batch dim: [1, filtered_seq_len, d_model]

                # Apply layer normalization
                updated_res_stream = self.model.ln_final(updated_res_stream)

                # Project to logit space
                ablated_logits = updated_res_stream @ self.model.W_U + self.model.b_U

                # Project neurons for frequency adjustment if needed
                ablated_logits_with_frozen_unigram = self.project_neurons(
                    logits[:, positions, :],  # Only use logits for positions we care about
                    unigram_projection_values,
                    ablated_logits,
                    res_deltas_sum,
                )

                # Create filtered input token sequence for loss computation
                filtered_inp = original_tok_seq[positions].unsqueeze(0).to(self.device)

                # Compute loss
                loss_post_ablation_batch = (
                    self.model.loss_fn(ablated_logits, filtered_inp, per_token=True).detach().cpu()
                )
                loss_post_ablation_batch = np.concatenate(
                    (loss_post_ablation_batch, np.zeros((loss_post_ablation_batch.shape[0], 1))), axis=1
                )
                loss_post_ablation.append(loss_post_ablation_batch)

                # Compute entropy
                entropy_post_ablation_batch = self.get_entropy(ablated_logits)
                entropy_post_ablation.append(entropy_post_ablation_batch.detach().cpu())

                # Process frozen unigram results
                loss_post_ablation_with_frozen_unigram_batch = (
                    self.model.loss_fn(ablated_logits_with_frozen_unigram, filtered_inp, per_token=True).detach().cpu()
                )
                loss_post_ablation_with_frozen_unigram_batch = np.concatenate(
                    (
                        loss_post_ablation_with_frozen_unigram_batch,
                        np.zeros((loss_post_ablation_with_frozen_unigram_batch.shape[0], 1)),
                    ),
                    axis=1,
                )
                loss_post_ablation_with_frozen_unigram.append(loss_post_ablation_with_frozen_unigram_batch)

                # Compute entropy for frozen unigram
                entropy_post_ablation_with_frozen_unigram_batch = self.get_entropy(ablated_logits_with_frozen_unigram)
                entropy_post_ablation_with_frozen_unigram.append(
                    entropy_post_ablation_with_frozen_unigram_batch.detach().cpu()
                )

                # Clean up
                del ablated_logits, ablated_logits_with_frozen_unigram

                # Process results - note that we need to pass the pos_to_idx mapping
                # to ensure correct position handling in the results processing
                batch_results = self.process_group_batch_results(
                    batch_n,
                    batch_df,  # Pass only the filtered DataFrame for this batch
                    loss_post_ablation,
                    loss_post_ablation_with_frozen_unigram,
                    entropy_post_ablation,
                    entropy_post_ablation_with_frozen_unigram,
                    pos_to_idx,  # Pass the position mapping
                )

                self.results = pd.concat([batch_results, self.results])
                batch_count += 1

                # Clean up to avoid memory issues
                del logits, cache, logprobs
                cleanup()

            except Exception as e:
                logger.error(f"Error processing batch {batch_n}: {e}")
                logger.error(traceback.format_exc())
                continue

        return self.results

    def process_group_batch_results(
        self,
        batch_n: int,
        batch_df: pd.DataFrame,
        loss_post_ablation: list,
        loss_post_ablation_with_frozen_unigram: list,
        entropy_post_ablation: list,
        entropy_post_ablation_with_frozen_unigram: list,
        pos_to_idx: dict[int, int] = None,
    ) -> pd.DataFrame:
        """Process results from ablation and prepare a DataFrame."""
        # Get unique positions for this batch
        positions = batch_df["pos"].unique()
        positions.sort()

        # Create a dictionary to store results for each position
        results_dict = defaultdict(list)

        # Access the data for this specific batch
        # Each of these lists has entries for each batch we've processed
        current_batch_losses = loss_post_ablation[0]  # First (and only) batch in the list
        current_batch_frozen_losses = loss_post_ablation_with_frozen_unigram[0]
        current_batch_entropy = entropy_post_ablation[0]
        current_batch_frozen_entropy = entropy_post_ablation_with_frozen_unigram[0]

        # Process each position
        for pos in positions:
            # Get the sequential index for this position if pos_to_idx is provided
            pos_idx = pos_to_idx[pos] if pos_to_idx else pos

            # Skip if position is out of range for sequence (should no longer happen with proper filtering)
            if pos_idx >= current_batch_losses.shape[1]:
                logger.warning(f"Position {pos} (idx {pos_idx}) is out of range for loss data")
                continue

            # Basic data
            results_dict["batch"].append(batch_n)
            results_dict["pos"].append(pos)  # Store original position, not the index

            # Get token information if available
            pos_rows = batch_df[batch_df["pos"] == pos]
            if "token_id" in pos_rows.columns:
                token_id = pos_rows["token_id"].iloc[0]
                results_dict["token_id"].append(token_id)

            # Process results for each group
            # Access the correct position in the current batch data using pos_idx (the sequential index)
            results_dict["loss_post_ablation"].append(current_batch_losses[0, pos_idx])
            results_dict["loss_post_ablation_with_frozen_unigram"].append(current_batch_frozen_losses[0, pos_idx])

            # Access the correct element in the entropy tensor
            results_dict["entropy_post_ablation"].append(current_batch_entropy[0, pos_idx].item())
            results_dict["entropy_post_ablation_with_frozen_unigram"].append(
                current_batch_frozen_entropy[0, pos_idx].item()
            )

        results_df = pd.DataFrame(results_dict)

        # Add metadata columns
        results_df["ablation_mode"] = self.ablation_mode
        results_df["longtail_threshold"] = self.longtail_threshold

        # Extract loss values from the original DataFrame for these positions
        results_df["loss"] = batch_df["loss"].to_list()

        if "longtail" in self.ablation_mode and self.longtail_mask is not None:
            results_df["num_longtail_tokens"] = self.longtail_mask.sum().item()

        return results_df

    def compute_heuristic(self, results) -> float:
        """Get heuristics from the results."""
        # filter based on freq
        results = self._filter_tokens(results)
        results["delta_loss"] = results["loss"] - results["loss_post_ablation"]
        return results["delta_loss"].mean()

    def _filter_tokens(self, results) -> pd.DataFrame:
        """Filter tokens by frequency."""
        # annotate token id with freq
        results["freq"] = results["freq"].apply(lambda token_id: self.unigram_analyzer.extract_freq(token_id)[1])
        # filter based on different conditions
        if "longtail" in self.sel_freq:
            logger.info(f"Compute heuristics on rare tokens. Before filtering: {results.shape[0]}")
            results = results[results["freq"] < self.longtail_threshold]
            logger.info(f"After filtering: {results.shape[0]}")
        if "common" in self.sel_freq:
            logger.info(f"Compute heuristics on common tokens. Before filtering: {results.shape[0]}")
            results = results[results["freq"] > self.longtail_threshold]
            logger.info(f"After filtering: {results.shape[0]}")
        return results

    def evaluate_neuron_group(self, neuron_group: list[int]) -> float:
        """Evaluate a neuron group and return a heuristic."""
        # Use the lock to ensure exclusive access to the model and cache
        with self.model_lock:
            results = self.mean_ablate_components(neuron_group)
            delta_loss = self.compute_heuristic(results)
            logger.info(f"Heuristic for current search: {delta_loss}")
            cleanup()
            return delta_loss


#######################################################################################################
# Functions applying search strategy


def get_heuristics(effect) -> bool:
    """Set the heuristic directions."""
    return effect == "boost"


@dataclass
class SearchResult:
    """Result of a neuron group search."""

    neurons: list[int]
    delta_loss: float
    is_target_size: bool = False


class NeuronGroupSearch:
    """Class for searching optimal neuron groups using various algorithms."""

    def __init__(
        self,
        neurons: list[int],
        evaluator,
        target_size: int,
        individual_delta_loss: list[float] | None = None,
        cache_dir: str | Path | None = None,
        maximize: bool = True,
        max_iterations: int = 1000,  # Parameter for early stopping
        timeout: int = 600,  # Timeout in seconds
        parallel_methods: bool = True,  # Run different search methods in parallel
        batch_size: int = 8,  # Size of evaluation batches
    ):
        """Initialize the neuron group search."""
        self.neurons = neurons
        self.evaluator = evaluator
        self.target_size = min(target_size, len(neurons))
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.maximize = maximize
        self.max_iterations = max_iterations
        self.timeout = timeout
        self.parallel_methods = parallel_methods
        self.batch_size = batch_size
        self._evaluation_cache = {}  # Cache for group evaluations
        self._evaluation_lock = threading.Lock()  # Lock for thread safety
        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True, parents=True)
        # Initialize individual neuron scores
        self.individual_delta_loss = individual_delta_loss
        # For early stopping
        self.best_score_history = []
        self.patience = 5  # Number of iterations with no improvement before stopping

    def _compute_individual_scores(self) -> list[float]:
        """Compute scores for individual neurons in batches for GPU efficiency."""
        logger.info("Computing individual neuron scores")
        scores = [None] * len(self.neurons)

        # Process neurons in batches
        for i in range(0, len(self.neurons), self.batch_size):
            batch_neurons = self.neurons[i : min(i + self.batch_size, len(self.neurons))]
            batch_indices = list(range(i, min(i + self.batch_size, len(self.neurons))))

            # Evaluate each neuron in the batch
            for j, (neuron, idx) in enumerate(zip(batch_neurons, batch_indices, strict=False)):
                scores[idx] = self.evaluator.evaluate_neuron_group([neuron])

            # Log progress
            if (i + self.batch_size) % 50 == 0 or i + self.batch_size >= len(self.neurons):
                logger.info(f"Processed {min(i + self.batch_size, len(self.neurons))}/{len(self.neurons)} neurons")

        return scores

    def _evaluate_group(self, group: list[int]) -> float:
        """Evaluate a group of neurons with caching."""
        # Create a hashable key for the group
        group_key = tuple(sorted(group))

        # Check cache first
        with self._evaluation_lock:
            if group_key in self._evaluation_cache:
                return self._evaluation_cache[group_key]

        # Evaluate the group
        result = self.evaluator.evaluate_neuron_group(group)

        # Cache the result
        with self._evaluation_lock:
            self._evaluation_cache[group_key] = result
        return result

    def _evaluate_groups_batch(self, groups: list[list[int]]) -> list[float]:
        """Evaluate multiple groups with efficient GPU batching."""
        # Check which groups are already in cache
        results = [None] * len(groups)
        groups_to_evaluate = []
        group_indices = []

        # This part checks the cache - need thread safety
        with self._evaluation_lock:
            for i, group in enumerate(groups):
                group_key = tuple(sorted(group))
                if group_key in self._evaluation_cache:
                    results[i] = self._evaluation_cache[group_key]
                else:
                    groups_to_evaluate.append(group)
                    group_indices.append(i)

        # If all groups were in cache, return results
        if not groups_to_evaluate:
            return results

        for i in range(0, len(groups_to_evaluate), self.batch_size):
            batch_groups = groups_to_evaluate[i : min(i + self.batch_size, len(groups_to_evaluate))]
            batch_indices = group_indices[i : min(i + self.batch_size, len(group_indices))]

            # Evaluate each group in the batch - the evaluator now has its own lock
            for j, (group, idx) in enumerate(zip(batch_groups, batch_indices, strict=False)):
                group_result = self.evaluator.evaluate_neuron_group(group)

                # Cache the result - need thread safety
                with self._evaluation_lock:
                    self._evaluation_cache[tuple(sorted(group))] = group_result

                # Store in results array
                results[idx] = group_result

        return results

    def _should_stop_early(self, new_score: float) -> bool:
        """Check if early stopping should be triggered."""
        if not self.best_score_history:
            self.best_score_history.append(new_score)
            return False

        best_so_far = self.best_score_history[-1]
        improved = (self.maximize and new_score > best_so_far) or (not self.maximize and new_score < best_so_far)

        if improved:
            self.best_score_history.append(new_score)
        else:
            # Track the same best score
            self.best_score_history.append(best_so_far)

        # Check for early stopping condition
        if len(self.best_score_history) >= self.patience:
            # If the best score hasn't improved for 'patience' iterations
            if len(set(self.best_score_history[-self.patience :])) == 1:
                logger.info(f"Early stopping triggered after {len(self.best_score_history)} iterations")
                return True

        return False

    def _get_sorted_neurons(self) -> list[tuple[int, float]]:
        """Get neurons sorted by their individual importance scores."""
        return sorted(
            [(i, self.individual_delta_loss[i]) for i in range(len(self.neurons))],
            key=lambda x: x[1],
            reverse=self.maximize,
        )

    def _save_search_state(self, method: str, state: dict) -> None:
        """Save search state to cache directory."""
        if not self.cache_dir:
            return
        path = self.cache_dir / f"{method}_search_state.pkl"

        with open(path, "wb") as f:
            pickle.dump(state, f)

    def _load_search_state(self, method: str) -> dict | None:
        """Load search state from cache directory."""
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

            # Use set for fast membership testing
            result_set = set(result.neurons)

            # Find neurons that aren't already in the result - more efficient
            available_neurons = []
            for idx, _ in sorted_neurons:
                neuron = self.neurons[idx]
                if neuron not in result_set:
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
            neuron_scores.sort(key=lambda x: x[1], reverse=self.maximize)

            # Keep the top neurons
            target_size_group = [n for n, _ in neuron_scores[: self.target_size]]
            delta_loss = self._evaluate_group(target_size_group)
            target_size_result = SearchResult(neurons=target_size_group, delta_loss=delta_loss, is_target_size=True)

        return best_result, target_size_result

    def progressive_beam_search(self, beam_width: int = 10) -> tuple[SearchResult, SearchResult]:
        """Progressive beam search for finding neuron groups."""
        start_time = time.time()
        state = self._load_search_state("progressive_beam")

        # Check if state exists and is marked as completed
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
            iteration_counter = state.get("iteration_counter", 0)
        else:
            # Start from scratch
            sorted_indices = self._get_sorted_neurons()
            current_beam = []
            for i in range(min(beam_width, len(sorted_indices))):
                idx, _ = sorted_indices[i]
                neuron = self.neurons[idx]
                delta_loss = self.individual_delta_loss[idx]
                current_beam.append((frozenset([neuron]), delta_loss))  # Use frozenset for faster set operations
            logger.info("start progressive beam search from scratch.")
            # Initialize tracking variables
            start_size = 2
            best_overall = None
            best_per_size = (
                {1: {"neurons": [self.neurons[sorted_indices[0][0]]], "delta_loss": sorted_indices[0][1]}}
                if sorted_indices
                else {}
            )
            iteration_counter = 0

        # Progressive beam search with early stopping and timeout
        logger.info("Performing progressive beam search")
        for size in range(start_size, self.target_size + 1):
            iteration_counter += 1

            # Check for timeout
            if time.time() - start_time > self.timeout:
                logger.info(f"Timeout reached after {time.time() - start_time:.2f} seconds")
                break

            # Check for max iterations
            if iteration_counter > self.max_iterations:
                logger.info(f"Maximum iterations ({self.max_iterations}) reached")
                break

            # Generate all candidate groups
            candidates = []
            sorted_indices = self._get_sorted_neurons()

            # Create all candidate groups first
            candidate_groups = []
            for group, _ in current_beam:
                for idx, _ in sorted_indices:
                    neuron = self.neurons[idx]
                    if neuron not in group:
                        new_group = group.union(frozenset([neuron]))
                        # Check if this group is already a candidate
                        if not any(g == new_group for g, _ in candidates):
                            candidate_groups.append((new_group, list(new_group)))

            # Batch evaluate all candidate groups
            if candidate_groups:
                group_lists = [group_list for _, group_list in candidate_groups]

                # Process in batches for GPU efficiency
                for i in range(0, len(group_lists), self.batch_size):
                    batch_groups = group_lists[i : min(i + self.batch_size, len(group_lists))]
                    batch_indices = list(range(i, min(i + self.batch_size, len(group_lists))))

                    # Evaluate the batch
                    batch_results = self._evaluate_groups_batch(batch_groups)

                    # Create candidates with their scores
                    for j, (idx, result) in enumerate(zip(batch_indices, batch_results, strict=False)):
                        group, group_list = candidate_groups[idx]
                        candidates.append((group, result))

                # Checkpoint periodically
                if iteration_counter % 2 == 0:
                    self._save_search_state(
                        "progressive_beam",
                        {
                            "current_beam": current_beam,
                            "next_size": size,
                            "best_overall": best_overall,
                            "best_per_size": best_per_size,
                            "iteration_counter": iteration_counter,
                            "completed": False,
                        },
                    )

            if not candidates:
                break

            candidates.sort(key=lambda x: x[1], reverse=self.maximize)
            current_beam = candidates[:beam_width]

            # Update best for current size
            if current_beam:
                # Use maximize flag to determine comparison function
                best_candidate = (
                    max(current_beam, key=lambda x: x[1]) if self.maximize else min(current_beam, key=lambda x: x[1])
                )
                best_per_size[size] = {
                    "neurons": list(best_candidate[0]),
                    "delta_loss": best_candidate[1],
                }

                # Update best overall if needed
                if best_overall is None or (
                    best_candidate[1] > best_overall["delta_loss"]
                    if self.maximize
                    else best_candidate[1] < best_overall["delta_loss"]
                ):
                    best_overall = {
                        "neurons": list(best_candidate[0]),
                        "delta_loss": best_candidate[1],
                    }

                # Check for early stopping
                if self._should_stop_early(best_candidate[1]):
                    logger.info(f"Early stopping at size {size}")
                    break

        # Create the best overall result
        best_result = None
        if best_overall:
            best_result = SearchResult(neurons=best_overall["neurons"], delta_loss=best_overall["delta_loss"])
        elif current_beam:
            # Use maximize flag to determine comparison function
            best_group = (
                max(current_beam, key=lambda x: x[1]) if self.maximize else min(current_beam, key=lambda x: x[1])
            )
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
                    "iteration_counter": iteration_counter,
                },
            )
        cleanup()
        return best_result, target_size_result

    def iterative_pruning(self) -> tuple[SearchResult, SearchResult]:
        """Iterative pruning search for finding neuron groups."""
        start_time = time.time()
        state = self._load_search_state("iterative_pruning")

        if state and state.get("completed") and "best_result" in state and "target_size_result" in state:
            best_result = SearchResult(**state["best_result"])
            target_size_result = SearchResult(**state["target_size_result"])
            return best_result, target_size_result

        # Initialize variables
        current_group = state["current_group"] if state and "current_group" in state else self.neurons.copy()
        best_per_size = {} if not state else state.get("best_per_size", {})
        iteration_counter = 0 if not state else state.get("iteration_counter", 0)

        # Track best group of any size
        if state and "best_group" in state and "best_score" in state:
            best_group = state["best_group"]
            best_score = state["best_score"]
        else:
            best_group = current_group.copy()
            best_score = self._evaluate_group(current_group)

        # Start with full set and record initial score
        initial_score = self._evaluate_group(current_group)
        best_per_size[len(current_group)] = (current_group.copy(), initial_score)

        # For early stopping
        best_scores = [best_score]

        # Prune until we reach target size
        while len(current_group) > self.target_size:
            iteration_counter += 1

            # Check for timeout
            if time.time() - start_time > self.timeout:
                logger.info(f"Timeout reached after {time.time() - start_time:.2f} seconds")
                break

            # Check for max iterations
            if iteration_counter > self.max_iterations:
                logger.info(f"Maximum iterations ({self.max_iterations}) reached")
                break

            # Prepare groups to evaluate
            candidate_groups = []
            neurons_to_remove = []

            # Try removing each neuron
            for n in current_group:
                test_group = [x for x in current_group if x != n]
                candidate_groups.append(test_group)
                neurons_to_remove.append(n)

            # Process in batches for GPU efficiency
            best_idx = -1
            best_pruned_score = float("-inf") if self.maximize else float("inf")

            for i in range(0, len(candidate_groups), self.batch_size):
                batch_groups = candidate_groups[i : min(i + self.batch_size, len(candidate_groups))]
                batch_indices = list(range(i, min(i + self.batch_size, len(candidate_groups))))

                # Evaluate the batch
                batch_results = self._evaluate_groups_batch(batch_groups)

                # Find best candidate in this batch
                for j, (idx, result) in enumerate(zip(batch_indices, batch_results, strict=False)):
                    if (self.maximize and result > best_pruned_score) or (
                        not self.maximize and result < best_pruned_score
                    ):
                        best_idx = idx
                        best_pruned_score = result

            # If we found a better result
            if best_idx >= 0:
                # Update best overall if better
                if (self.maximize and best_pruned_score > best_score) or (
                    not self.maximize and best_pruned_score < best_score
                ):
                    best_group = candidate_groups[best_idx].copy()
                    best_score = best_pruned_score
                    best_scores.append(best_score)
                else:
                    # No improvement
                    best_scores.append(best_score)

                # Remove worst neuron
                worst_neuron = neurons_to_remove[best_idx]
                current_group.remove(worst_neuron)

                # Store best for this size
                best_per_size[len(current_group)] = (current_group.copy(), best_pruned_score)

                # Check for early stopping
                if len(best_scores) >= self.patience and len(set(best_scores[-self.patience :])) == 1:
                    logger.info(f"Early stopping triggered after {iteration_counter} iterations")
                    break

                # Checkpoint periodically
                if iteration_counter % 5 == 0 and self.cache_dir:
                    self._save_search_state(
                        "iterative_pruning",
                        {
                            "current_group": current_group,
                            "best_per_size": best_per_size,
                            "best_group": best_group,
                            "best_score": best_score,
                            "iteration_counter": iteration_counter,
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

    def hierarchical_cluster_search(
        self, n_clusters: int = 10, expansion_factor: int = 10
    ) -> tuple[SearchResult, SearchResult]:
        """Hierarchical clustering search for finding neuron groups."""
        start_time = time.time()
        state = self._load_search_state("hierarchical_cluster")

        if state and state.get("completed") and "best_result" in state and "target_size_result" in state:
            best_result = SearchResult(**state["best_result"])
            target_size_result = SearchResult(**state["target_size_result"])
            return best_result, target_size_result

        logger.info("Performing hierarchical cluster search")
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
                reverse=self.maximize,
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
            reverse=self.maximize,
        )

        # Initialize with top representatives
        current_group = [n for n, _ in sorted_reps[: min(5, self.target_size)]]
        best_per_size = {len(current_group): (current_group.copy(), self._evaluate_group(current_group))}

        # Keep track of best group of any size
        best_group = current_group.copy()
        best_score = self._evaluate_group(best_group)

        # For early stopping
        iteration_counter = 0
        best_scores = [best_score]

        # Continue adding neurons
        while len(current_group) < self.target_size:
            iteration_counter += 1

            # Check for timeout
            if time.time() - start_time > self.timeout:
                logger.info(f"Timeout reached after {time.time() - start_time:.2f} seconds")
                break

            # Check for max iterations
            if iteration_counter > self.max_iterations:
                logger.info(f"Maximum iterations ({self.max_iterations}) reached")
                break

            # Find best candidates to add
            candidate_neurons = [n for n, _ in sorted_reps if n not in current_group]

            # Batch evaluate adding each candidate
            candidate_groups = []
            for n in candidate_neurons[: min(len(candidate_neurons), 10)]:  # Limit candidates for efficiency
                test_group = current_group + [n]
                candidate_groups.append(test_group)

            # Process in batches for GPU efficiency
            best_idx = -1
            best_candidate_score = float("-inf") if self.maximize else float("inf")

            for i in range(0, len(candidate_groups), self.batch_size):
                batch_groups = candidate_groups[i : min(i + self.batch_size, len(candidate_groups))]
                batch_indices = list(range(i, min(i + self.batch_size, len(candidate_groups))))

                # Evaluate the batch
                batch_results = self._evaluate_groups_batch(batch_groups)

                # Find best candidate in this batch
                for j, (idx, result) in enumerate(zip(batch_indices, batch_results, strict=False)):
                    if (self.maximize and result > best_candidate_score) or (
                        not self.maximize and result < best_candidate_score
                    ):
                        best_idx = idx
                        best_candidate_score = result

            # If we found a better candidate
            if best_idx >= 0:
                # Update best overall if better
                if (self.maximize and best_candidate_score > best_score) or (
                    not self.maximize and best_candidate_score < best_score
                ):
                    best_group = candidate_groups[best_idx].copy()
                    best_score = best_candidate_score
                    best_scores.append(best_score)
                else:
                    best_scores.append(best_score)

                # Add best candidate to current group
                current_group = candidate_groups[best_idx]

                # Store best for this size
                best_per_size[len(current_group)] = (current_group.copy(), best_candidate_score)

                # Check for early stopping
                if len(best_scores) >= self.patience and len(set(best_scores[-self.patience :])) == 1:
                    logger.info(f"Early stopping triggered after {iteration_counter} iterations")
                    break

                # Checkpoint periodically
                if iteration_counter % 5 == 0 and self.cache_dir:
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

    def importance_weighted_sampling(
        self, n_iterations: int = 100, learning_rate: float = 0.1, checkpoint_freq: int = 20
    ) -> tuple[SearchResult, SearchResult]:
        """Find neuron groups using importance weighted sampling."""
        start_time = time.time()
        method = "importance_weighted"
        state = self._load_search_state(method)

        if state and state.get("completed") and "best_result" in state and "target_size_result" in state:
            best_result = SearchResult(**state["best_result"])
            target_size_result = SearchResult(**state["target_size_result"])
            return best_result, target_size_result

        logger.info("Performing importance weighted sampling")
        # Initialize variables
        if state and not state.get("completed"):
            weights = state["weights"]
            best_group = state["best_group"]
            best_score = state["best_score"]
            best_target_size_group = state.get("best_target_size_group")
            best_target_size_score = state.get(
                "best_target_size_score", float("-inf") if self.maximize else float("inf")
            )
            start_iteration = state["iteration"]
        else:
            # Start from scratch
            scores = np.array(self.individual_delta_loss)
            # Adjust initialization based on maximization goal
            if self.maximize:
                min_score = scores.min()
                weights = scores - min_score + 1e-6
            else:
                max_score = scores.max()
                weights = max_score - scores + 1e-6

            weights = weights / weights.sum()
            best_group = None
            best_score = float("-inf") if self.maximize else float("inf")
            best_target_size_group = None
            best_target_size_score = float("-inf") if self.maximize else float("inf")
            start_iteration = 0

        # For early stopping
        best_scores = []
        if best_score != (float("-inf") if self.maximize else float("inf")):
            best_scores = [best_score]

        # Run the sampling with batched evaluations
        iteration = start_iteration
        while iteration < n_iterations:
            # Check for timeout
            if time.time() - start_time > self.timeout:
                logger.info(f"Timeout reached after {time.time() - start_time:.2f} seconds")
                break

            # Generate multiple samples at once for batch evaluation
            batch_size = min(self.batch_size, n_iterations - iteration)
            sampled_groups = []

            for _ in range(batch_size):
                if len(self.neurons) <= self.target_size:
                    sampled_indices = np.arange(len(self.neurons))
                else:
                    sampled_indices = np.random.choice(
                        len(self.neurons), size=self.target_size, replace=False, p=weights
                    )
                sampled_group = [self.neurons[idx] for idx in sampled_indices]
                sampled_groups.append((sampled_group, sampled_indices))

            # Evaluate all samples in batch
            group_lists = [group for group, _ in sampled_groups]
            scores = self._evaluate_groups_batch(group_lists)

            # Process results
            for i, ((sampled_group, sampled_indices), score) in enumerate(zip(sampled_groups, scores, strict=False)):
                # Update best overall
                if (self.maximize and score > best_score) or (not self.maximize and score < best_score):
                    best_group = sampled_group.copy()
                    best_score = score
                    best_scores.append(best_score)
                # No improvement
                elif best_scores:
                    best_scores.append(best_scores[-1])

                # Update best target size (if exactly target size)
                if len(sampled_group) == self.target_size and (
                    (self.maximize and score > best_target_size_score)
                    or (not self.maximize and score < best_target_size_score)
                ):
                    best_target_size_group = sampled_group.copy()
                    best_target_size_score = score

                # Update weights - if maximizing, reward higher scores, if minimizing, reward lower scores
                adjustment = learning_rate * score if self.maximize else learning_rate * (1.0 / (score + 1e-10))
                for idx in sampled_indices:
                    weights[idx] += adjustment

            # Normalize weights
            weights = weights / weights.sum()

            # Update iteration count
            iteration += batch_size

            # Check for early stopping
            if len(best_scores) >= self.patience:
                if len(set(best_scores[-self.patience :])) == 1:
                    logger.info(f"Early stopping triggered after {iteration} iterations")
                    break

            # Checkpoint
            if self.cache_dir and (iteration % checkpoint_freq == 0 or iteration >= n_iterations):
                self._save_search_state(
                    method,
                    {
                        "iteration": iteration,
                        "weights": weights,
                        "best_group": best_group,
                        "best_score": best_score,
                        "best_target_size_group": best_target_size_group,
                        "best_target_size_score": best_target_size_score,
                        "completed": False,
                    },
                )

        # Create best overall result
        default_score = 0.0 if self.maximize else float("inf")
        best_result = SearchResult(
            neurons=best_group if best_group else [],
            delta_loss=best_score
            if best_score != (float("-inf") if self.maximize else float("inf"))
            else default_score,
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
        self, n_clusters: int = 5, expansion_factor: int = 3, beam_width: int = 10
    ) -> tuple[SearchResult, SearchResult]:
        """Hybrid clustering and beam search for finding neuron groups."""
        start_time = time.time()
        state = self._load_search_state("hybrid")

        if state and state.get("completed") and "best_result" in state and "target_size_result" in state:
            best_result = SearchResult(**state["best_result"])
            target_size_result = SearchResult(**state["target_size_result"])
            return best_result, target_size_result

        logger.info("Performing hybrid search")
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
                reverse=self.maximize,
            )
            representatives.extend([n for n, _ in sorted_cluster[:expansion_factor]])

        # Step 2: Create a reduced search with shared cache
        # We'll create a wrapper for the evaluator that uses our existing cache
        class CachedEvaluator:
            def __init__(self, parent):
                self.parent = parent

            def evaluate_neuron_group(self, group):
                return self.parent._evaluate_group(group)

        cached_evaluator = CachedEvaluator(self)

        # Create reduced search
        reduced_search = NeuronGroupSearch(
            neurons=representatives,
            evaluator=cached_evaluator,
            target_size=self.target_size,
            individual_delta_loss=[self.individual_delta_loss[self.neurons.index(n)] for n in representatives],
            maximize=self.maximize,
            max_iterations=self.max_iterations,
            timeout=max(0, self.timeout - (time.time() - start_time)),  # Remaining time
            parallel_methods=False,  # Disable nested parallelism
            batch_size=self.batch_size,  # Maintain same batch size
        )

        # Run beam search on the reduced set
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

        return best_result, target_size_result

    def run_methods_sequentially(self) -> dict[str, tuple[SearchResult, SearchResult]]:
        """Run all search methods sequentially and return the results."""
        results = {}
        methods = [
            ("progressive_beam", self.progressive_beam_search),
            ("hierarchical_cluster", self.hierarchical_cluster_search),
            ("iterative_pruning", self.iterative_pruning),
            ("importance_weighted", self.importance_weighted_sampling),
            ("hybrid", self.hybrid_search),
        ]

        for method_name, method_func in methods:
            try:
                logger.info(f"Starting {method_name}")
                result = method_func()
                logger.info(f"Completed {method_name}")
                results[method_name] = result
            except Exception as e:
                logger.error(f"Error in {method_name}: {e}")
                import traceback

                logger.error(traceback.format_exc())
                empty_result = SearchResult(neurons=[], delta_loss=0.0)
                results[method_name] = (empty_result, empty_result)

        return results

    def run_methods_parallel(self) -> dict[str, tuple[SearchResult, SearchResult]]:
        """Run all search methods in parallel and return the results."""
        results = {}
        methods = [
            ("progressive_beam", self.progressive_beam_search),
            ("hierarchical_cluster", self.hierarchical_cluster_search),
            ("iterative_pruning", self.iterative_pruning),
            ("importance_weighted", self.importance_weighted_sampling),
            ("hybrid", self.hybrid_search),
        ]
        # Thread-based parallelization for running methods
        threads = []
        results_queue = Queue()

        def run_method_thread(method_name, method_func):
            try:
                logger.info(f"Starting {method_name}")
                # The evaluator has its own locks now, so methods can run in parallel
                result = method_func()
                logger.info(f"Completed {method_name}")
                results_queue.put((method_name, result))
            except Exception as e:
                logger.error(f"Error in {method_name}: {e}")
                import traceback

                logger.error(traceback.format_exc())
                empty_result = SearchResult(neurons=[], delta_loss=0.0)
                results_queue.put((method_name, (empty_result, empty_result)))

        # Start threads for each method
        for method_name, method_func in methods:
            thread = threading.Thread(target=run_method_thread, args=(method_name, method_func))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Collect results
        while not results_queue.empty():
            method_name, result = results_queue.get()
            results[method_name] = result
        cleanup()
        return results

    def get_best_result(self) -> dict[str, t.Any]:
        """Run all methods and return the best method and its results."""
        if self.parallel_methods:
            results = self.run_methods_parallel()
        else:
            results = self.run_methods_sequentially()

        # Comparison function based on maximization goal
        if self.maximize:
            best_method_name = max(results.keys(), key=lambda x: results[x][0].delta_loss)
            target_method_name = max(results.keys(), key=lambda x: results[x][1].delta_loss)
        else:
            best_method_name = min(results.keys(), key=lambda x: results[x][0].delta_loss)
            target_method_name = min(results.keys(), key=lambda x: results[x][1].delta_loss)

        return {
            "best": results[best_method_name][0],
            "target_size": results[target_method_name][1],
            "total": results,
            "best_method": best_method_name,
            "target_method": target_method_name,
        }
