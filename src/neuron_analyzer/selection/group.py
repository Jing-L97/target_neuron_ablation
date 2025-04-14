#!/usr/bin/env python
import json
import logging
import pickle
import random
import sys
import typing as t
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neuron_analyzer.ablation.ablation import ModelAblationAnalyzer
from neuron_analyzer.selection.group import NeuronGroupSearch

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
sys.path.append("../")
# Type variable for generic functions
T = t.TypeVar("T")

#######################################################################################################
# Fucntions applied in the main scripts
#######################################################################################################


@dataclass
class SearchResult:
    """Result of a neuron group search."""

    neurons: list[int]
    mediation: float
    kl_divergence: float

    def __repr__(self) -> str:
        return f"SearchResult(neurons={self.neurons}, mediation={self.mediation:.4f}, kl={self.kl_divergence:.4f})"


class GroupAblationAnalyzer(ModelAblationAnalyzer):
    """Extended version of ModelAblationAnalyzer with neuron group search and resumable experiments."""

    def __init__(
        self,
        model,
        unigram_distrib,
        tokenized_data,
        entropy_df,
        components_to_ablate,
        device: str,
        logger=None,
        k: int = 10,
        chunk_size: int = 20,
        ablation_mode: str = "mean",  # "mean" or "longtail"
        longtail_threshold: float = 0.001,  # Threshold for long-tail tokens
        cache_dir: str | Path | None = None,
    ):
        """Initialize with all original parameters plus cache directory for resumable experiments."""
        # Call the parent class constructor
        super().__init__(
            model=model,
            unigram_distrib=unigram_distrib,
            tokenized_data=tokenized_data,
            entropy_df=entropy_df,
            components_to_ablate=components_to_ablate,
            device=device,
            logger=logger,
            k=k,
            chunk_size=chunk_size,
            ablation_mode=ablation_mode,
            longtail_threshold=longtail_threshold,
        )

        # Add new attributes for caching
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./ablation_cache")
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        # Cache for evaluation results
        self.eval_cache = {}
        self._load_eval_cache()

    def _get_cache_path(self, prefix: str = "") -> Path:
        """Generate a path for cache files with optional prefix."""
        filename = f"{prefix}_ablation_cache.pkl" if prefix else "ablation_cache.pkl"
        return self.cache_dir / filename

    def _load_eval_cache(self) -> None:
        """Load evaluation cache from disk if it exists."""
        cache_path = self._get_cache_path("eval")
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    self.eval_cache = pickle.load(f)
                if self.logger:
                    self.logger.info(f"Loaded {len(self.eval_cache)} cached evaluations from {cache_path}")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to load evaluation cache: {e}")
                self.eval_cache = {}

    def _save_eval_cache(self) -> None:
        """Save evaluation cache to disk."""
        cache_path = self._get_cache_path("eval")
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(self.eval_cache, f)
            if self.logger:
                self.logger.info(f"Saved {len(self.eval_cache)} evaluations to {cache_path}")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to save evaluation cache: {e}")

    def create_neuron_group_search(
        self, neurons: list[int], target_size: int, resume_path: Path | str | None = None
    ) -> "NeuronGroupSearch":
        """Create a NeuronGroupSearch instance with an evaluation function based on this analyzer."""

        # Define the evaluation function that will be used by the search
        def evaluation_fn(neuron_group: list[int]) -> tuple[float, float]:
            return self.evaluate_neuron_group(neuron_group)

        # First compute individual scores
        individual_mediation = []
        individual_kl = []

        for neuron in neurons:
            mediation, kl = self.evaluate_neuron_group([neuron])
            individual_mediation.append(mediation)
            individual_kl.append(kl)

        # Create and return the search object
        return NeuronGroupSearch(
            neurons=neurons,
            evaluation_fn=evaluation_fn,
            target_size=target_size,
            individual_mediation=individual_mediation,
            individual_kl=individual_kl,
            cache_dir=self.cache_dir,
            resume_path=resume_path,
        )

    def evaluate_neuron_group(self, neuron_group: list[int]) -> tuple[float, float]:
        """Evaluate a group of neurons using ablation and return mediation and KL divergence metrics.

        Args:
            neuron_group: List of neuron indices to evaluate as a group

        Returns:
            Tuple of (mediation_score, kl_divergence)

        """
        # Check if this evaluation is already cached
        cache_key = tuple(sorted(neuron_group))
        if cache_key in self.eval_cache:
            return self.eval_cache[cache_key]

        # Build frequency vectors if not already done
        if not hasattr(self, "unigram_direction_vocab"):
            self.build_vector()

        # Sample a random batch for evaluation if not using all batches
        if len(neuron_group) > 0:
            # Get layer index from the first component
            self.layer_idx = int(self.components_to_ablate[0].split(".")[0])

            # Set components to ablate to the neuron group for this evaluation
            original_components = self.components_to_ablate
            self.components_to_ablate = [f"{self.layer_idx}.{neuron}" for neuron in neuron_group]

            # Perform ablation on a single batch for efficiency
            random_batch = np.random.choice(self.entropy_df.batch.unique())
            ablation_results = self._ablate_single_batch(random_batch)

            # Extract mediation and KL metrics from results
            mediation_score = ablation_results["mediation_score"].mean()
            kl_divergence = ablation_results["kl_divergence_after_frozen_unigram"].mean()

            # Restore original components
            self.components_to_ablate = original_components

            # Cache the result
            result = (mediation_score, kl_divergence)
            self.eval_cache[cache_key] = result

            # Periodically save cache
            if len(self.eval_cache) % 50 == 0:
                self._save_eval_cache()

            return result
        # Return default values for empty group
        return (0.0, 0.0)

    def _ablate_single_batch(self, batch_n: int) -> pd.DataFrame:
        """Ablate a single batch and return the results for neuron group evaluation.

        Args:
            batch_n: Batch index to ablate

        Returns:
            DataFrame with ablation metrics

        """
        tok_seq = self.tokenized_data["tokens"][batch_n]

        # Get unaltered logits
        self.model.reset_hooks()
        inp = tok_seq.unsqueeze(0).to(self.device)
        logits, cache = self.model.run_with_cache(inp)
        logprobs = logits[0, :, :].log_softmax(dim=-1)

        res_stream = cache[self.utils.get_act_name("resid_post", self.layer_idx)][0]

        # Get the entropy_df entries for the current sequence
        rows = self.entropy_df[self.entropy_df.batch == batch_n]

        # Get the value of the logits projected onto the b_U direction
        unigram_projection_values = self.project_logits(logits)

        # Get neuron indices from components_to_ablate
        self.neuron_indices = [int(neuron_name.split(".")[1]) for neuron_name in self.components_to_ablate]

        # Get layer indices
        layer_indices = [int(neuron_name.split(".")[0]) for neuron_name in self.components_to_ablate]
        self.layer_idx = layer_indices[0]

        previous_activation = cache[self.utils.get_act_name("post", self.layer_idx)][0, :, self.neuron_indices]
        del cache

        # Calculate activation means for the ablation
        activation_mean_values = torch.tensor(
            self.entropy_df[[f"{component_name}_activation" for component_name in self.components_to_ablate]].mean()
        )

        activation_deltas = activation_mean_values.to(previous_activation.device) - previous_activation
        # Multiple deltas by W_out
        res_deltas = activation_deltas.unsqueeze(-1) * self.model.W_out[self.layer_idx, self.neuron_indices, :]
        res_deltas = res_deltas.permute(1, 0, 2)

        loss_post_ablation = []
        entropy_post_ablation = []
        loss_post_ablation_with_frozen_unigram = []
        entropy_post_ablation_with_frozen_unigram = []
        kl_divergence_after = []
        kl_divergence_after_frozen_unigram = []

        kl_divergence_before = (
            self.kl_div(logprobs, self.log_unigram_distrib, reduction="none", log_target=True)
            .sum(axis=-1)
            .cpu()
            .numpy()
        )

        # Process in chunks to manage memory
        for i in range(0, res_deltas.shape[0], self.chunk_size):
            res_deltas_chunk = res_deltas[i : i + self.chunk_size]
            updated_res_stream_chunk = res_stream.repeat(res_deltas_chunk.shape[0], 1, 1) + res_deltas_chunk
            # Apply ln_final
            updated_res_stream_chunk = self.model.ln_final(updated_res_stream_chunk)

            # Project to logit space
            ablated_logits_chunk = updated_res_stream_chunk @ self.model.W_U + self.model.b_U

            ablated_logits_with_frozen_unigram_chunk = self.project_neurons(
                logits,
                unigram_projection_values,
                ablated_logits_chunk,
                res_deltas_chunk,
            )

            # Compute loss for the chunk
            loss_post_ablation_chunk = self.model.loss_fn(
                ablated_logits_chunk, inp.repeat(res_deltas_chunk.shape[0], 1), per_token=True
            ).cpu()
            loss_post_ablation_chunk = np.concatenate(
                (loss_post_ablation_chunk, np.zeros((loss_post_ablation_chunk.shape[0], 1))), axis=1
            )
            loss_post_ablation.append(loss_post_ablation_chunk)

            # Compute entropy for the chunk
            entropy_post_ablation_chunk = self.get_entropy(ablated_logits_chunk)
            entropy_post_ablation.append(entropy_post_ablation_chunk.cpu())

            abl_logprobs = ablated_logits_chunk.log_softmax(dim=-1)

            del ablated_logits_chunk

            # Compute loss for ablated_logits_with_frozen_unigram_chunk
            loss_post_ablation_with_frozen_unigram_chunk = self.model.loss_fn(
                ablated_logits_with_frozen_unigram_chunk, inp.repeat(res_deltas_chunk.shape[0], 1), per_token=True
            ).cpu()
            loss_post_ablation_with_frozen_unigram_chunk = np.concatenate(
                (
                    loss_post_ablation_with_frozen_unigram_chunk,
                    np.zeros((loss_post_ablation_with_frozen_unigram_chunk.shape[0], 1)),
                ),
                axis=1,
            )
            loss_post_ablation_with_frozen_unigram.append(loss_post_ablation_with_frozen_unigram_chunk)

            # Compute entropy for ablated_logits_with_frozen_unigram_chunk
            entropy_post_ablation_with_frozen_unigram_chunk = self.get_entropy(ablated_logits_with_frozen_unigram_chunk)
            entropy_post_ablation_with_frozen_unigram.append(entropy_post_ablation_with_frozen_unigram_chunk.cpu())

            # Calculate KL divergence
            kl_divergence_after, kl_divergence_after_frozen_unigram = self.compute_kl(
                ablated_logits_with_frozen_unigram_chunk,
                abl_logprobs,
                kl_divergence_after,
                kl_divergence_after_frozen_unigram,
            )

        # Concatenate results
        loss_post_ablation = np.concatenate(loss_post_ablation, axis=0)
        entropy_post_ablation = np.concatenate(entropy_post_ablation, axis=0)
        loss_post_ablation_with_frozen_unigram = np.concatenate(loss_post_ablation_with_frozen_unigram, axis=0)
        entropy_post_ablation_with_frozen_unigram = np.concatenate(entropy_post_ablation_with_frozen_unigram, axis=0)
        kl_divergence_after = np.concatenate(kl_divergence_after, axis=0)
        kl_divergence_after_frozen_unigram = np.concatenate(kl_divergence_after_frozen_unigram, axis=0)

        # Clean up
        del res_deltas
        torch.cuda.empty_cache()

        # Create a simplified results DataFrame with just the evaluation metrics we need
        results_df = pd.DataFrame(
            {
                "loss_post_ablation": loss_post_ablation.mean(axis=1),
                "loss_post_ablation_with_frozen_unigram": loss_post_ablation_with_frozen_unigram.mean(axis=1),
                "entropy_post_ablation": entropy_post_ablation.mean(axis=1),
                "entropy_post_ablation_with_frozen_unigram": entropy_post_ablation_with_frozen_unigram.mean(axis=1),
                "kl_divergence_before": np.repeat(kl_divergence_before.mean(), len(kl_divergence_after)),
                "kl_divergence_after": kl_divergence_after.mean(axis=1),
                "kl_divergence_after_frozen_unigram": kl_divergence_after_frozen_unigram.mean(axis=1),
                # Calculate mediation score (custom metric for neuron importance)
                "mediation_score": (
                    loss_post_ablation.mean(axis=1) - loss_post_ablation_with_frozen_unigram.mean(axis=1)
                ),
            }
        )

        return results_df

    def run_neuron_group_search(
        self, neurons: list[int], target_size: int, method: str = "hybrid", resume_path: Path | str | None = None
    ) -> SearchResult:
        """Run a neuron group search using the specified method.

        Args:
            neurons: List of neuron indices to search from
            target_size: Target size of the neuron group
            method: Search method to use ("progressive_beam", "hierarchical_cluster",
                    "iterative_pruning", "importance_weighted", "hybrid", or "all")
            resume_path: Optional path to resume search from

        Returns:
            SearchResult with the best found neuron group

        """
        # Create the search object
        search = self.create_neuron_group_search(neurons, target_size, resume_path)

        # Run the specified search method
        if method == "progressive_beam":
            return search.progressive_beam_search()
        if method == "hierarchical_cluster":
            return search.hierarchical_cluster_search()
        if method == "iterative_pruning":
            return search.iterative_pruning()
        if method == "importance_weighted":
            return search.importance_weighted_sampling()
        if method == "hybrid":
            return search.hybrid_search()
        if method == "all":
            # Run all methods and return the best result
            _, best_result = search.get_best_result()
            return best_result
        raise ValueError(f"Unknown search method: {method}")

    def search_and_ablate(
        self,
        neurons: list[int],
        target_size: int,
        search_method: str = "hybrid",
        resume_path: Path | str | None = None,
        n_batches: int = 10,
    ) -> tuple[SearchResult, pd.DataFrame]:
        """Perform a neuron group search followed by ablation of the found group.

        Args:
            neurons: List of neuron indices to search from
            target_size: Target size of the neuron group
            search_method: Method to use for the search
            resume_path: Path to resume search from
            n_batches: Number of batches to ablate

        Returns:
            Tuple of (SearchResult, ablation_results_df)

        """
        # Run the search
        search_result = self.run_neuron_group_search(neurons, target_size, search_method, resume_path)

        # Get the neurons from the search result
        neuron_group = search_result.neurons

        # Convert to component format
        self.layer_idx = int(self.components_to_ablate[0].split(".")[0])  # Assuming all components are from same layer
        components = [f"{self.layer_idx}.{neuron}" for neuron in neuron_group]

        # Save original components
        original_components = self.components_to_ablate

        # Set components to the search result
        self.components_to_ablate = components

        # Run ablation on multiple batches
        random_batches = np.random.choice(
            self.entropy_df.batch.unique(), min(n_batches, len(self.entropy_df.batch.unique())), replace=False
        )
        ablation_results = {}

        for batch_n in random_batches:
            ablation_results[batch_n] = self._ablate_single_batch(batch_n)

        # Restore original components
        self.components_to_ablate = original_components

        # Combine results
        combined_results = pd.concat(ablation_results.values())

        # Add search information
        combined_results["search_method"] = search_method
        combined_results["target_size"] = target_size
        combined_results["neuron_group"] = str(neuron_group)
        combined_results["neuron_group_size"] = len(neuron_group)
        combined_results["mediation"] = search_result.mediation
        combined_results["kl_divergence"] = search_result.kl_divergence

        return search_result, combined_results

    def save_search_results(self, results: dict[str, SearchResult], filename: str) -> None:
        """Save search results to a file.

        Args:
            results: Dictionary mapping method names to SearchResult objects
            filename: Name of the file to save to

        """
        output_path = self.cache_dir / filename

        # Convert SearchResult objects to dictionaries
        serializable_results = {}
        for method, result in results.items():
            serializable_results[method] = asdict(result)

        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        if self.logger:
            self.logger.info(f"Search results saved to {output_path}")

    def group_ablation_experiment(
        self,
        neurons: list[int],
        target_sizes: list[int],
        search_methods: list[str] = ["hybrid"],
        n_batches: int = 10,
        resume_dir: Path | str | None = None,
    ) -> dict[str, dict[int, tuple[SearchResult, pd.DataFrame]]]:
        """Run comprehensive ablation experiments with different group sizes and search methods.

        Args:
            neurons: List of all neurons to consider
            target_sizes: List of target group sizes to test
            search_methods: List of search methods to use
            n_batches: Number of batches to ablate for each group
            resume_dir: Directory to save/resume results from

        Returns:
            Nested dictionary: {method: {size: (search_result, ablation_df)}}

        """
        if resume_dir:
            resume_dir = Path(resume_dir)
            resume_dir.mkdir(exist_ok=True, parents=True)

        results = defaultdict(dict)

        # Iterate through all method and size combinations
        for method in search_methods:
            for size in target_sizes:
                # Check if we have cached results
                result_cached = False
                if resume_dir:
                    result_path = resume_dir / f"{method}_size_{size}_result.pkl"
                    if result_path.exists():
                        try:
                            with open(result_path, "rb") as f:
                                search_result, ablation_df = pickle.load(f)
                                results[method][size] = (search_result, ablation_df)
                                result_cached = True
                                if self.logger:
                                    self.logger.info(f"Loaded cached result for method={method}, size={size}")
                        except Exception as e:
                            if self.logger:
                                self.logger.warning(f"Failed to load cached result: {e}")

                # Run the experiment if not cached
                if not result_cached:
                    # Determine resume path
                    resume_path = None
                    if resume_dir:
                        resume_path = resume_dir / f"{method}_size_{size}_state.pkl"
                        if not resume_path.exists():
                            resume_path = None

                    # Run search and ablation
                    search_result, ablation_df = self.search_and_ablate(
                        neurons=neurons,
                        target_size=size,
                        search_method=method,
                        resume_path=resume_path,
                        n_batches=n_batches,
                    )

                    results[method][size] = (search_result, ablation_df)

                    # Save result if we have a resume directory
                    if resume_dir:
                        result_path = resume_dir / f"{method}_size_{size}_result.pkl"
                        with open(result_path, "wb") as f:
                            pickle.dump((search_result, ablation_df), f)
                            if self.logger:
                                self.logger.info(f"Saved result for method={method}, size={size}")

        return results

    def compare_neuron_groups(
        self,
        experiment_results: dict[str, dict[int, tuple[SearchResult, pd.DataFrame]]],
        metric: str = "mediation_score",
    ) -> pd.DataFrame:
        """Compare the performance of different neuron groups from experiments.

        Args:
            experiment_results: Results from group_ablation_experiment
            metric: Metric to use for comparison

        Returns:
            DataFrame with comparison metrics

        """
        comparison = []

        for method, size_results in experiment_results.items():
            for size, (search_result, ablation_df) in size_results.items():
                row = {
                    "method": method,
                    "size": size,
                    "neurons": str(search_result.neurons),
                    "mediation": search_result.mediation,
                    "kl_divergence": search_result.kl_divergence,
                    f"mean_{metric}": ablation_df[metric].mean(),
                    f"std_{metric}": ablation_df[metric].std(),
                    f"max_{metric}": ablation_df[metric].max(),
                    f"min_{metric}": ablation_df[metric].min(),
                }
                comparison.append(row)

        comparison_df = pd.DataFrame(comparison)
        # Sort by the primary metric
        comparison_df = comparison_df.sort_values(by=f"mean_{metric}", ascending=False)

        return comparison_df


#######################################################################################################
# Functions applying search strategy
#######################################################################################################


class GroupSearcher(NeuronGroupSearch):
    """Extended version of NeuronGroupSearch with resume capability."""

    def __init__(
        self,
        neurons: list[int],
        evaluation_fn: t.Callable[[list[int]], tuple[float, float]],
        target_size: int,
        individual_mediation: list[float] | None = None,
        individual_kl: list[float] | None = None,
        cache_dir: str | Path | None = None,
        resume_path: str | Path | None = None,
    ):
        """Initialize the neuron group search with resume capability."""
        # Initialize the parent class
        super().__init__(
            neurons=neurons,
            evaluation_fn=evaluation_fn,
            target_size=target_size,
            individual_mediation=individual_mediation,
            individual_kl=individual_kl,
        )

        # Setup cache directory
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./search_cache")
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        # Cache for evaluated groups
        self.evaluated_groups = {}

        # Resume from previous search if path is provided
        if resume_path:
            self._load_search_state(resume_path)

    def _get_cache_path(self, method: str, suffix: str = "cache") -> Path:
        """Generate a path for cache files."""
        filename = f"search_{method}_{self.target_size}_{suffix}.pkl"
        return self.cache_dir / filename

    def _evaluate_group(self, group: list[int]) -> tuple[float, float]:
        """Evaluate a neuron group using the evaluation function with caching."""
        # Use frozenset for cache key to ensure order-independence
        cache_key = frozenset(group)

        # Check cache first
        if cache_key in self.evaluated_groups:
            return self.evaluated_groups[cache_key]

        # Evaluate if not in cache
        mediation, kl = self.evaluation_fn(group)

        # Cache the result
        self.evaluated_groups[cache_key] = (mediation, kl)

        return mediation, kl

    def _save_search_state(self, method: str, state: dict) -> Path:
        """Save the current search state to resume later."""
        cache_path = self._get_cache_path(method, "state")

        with open(cache_path, "wb") as f:
            pickle.dump(
                {
                    "state": state,
                    "evaluated_groups": self.evaluated_groups,
                    "target_size": self.target_size,
                    "neurons": self.neurons,
                    "individual_mediation": self.individual_mediation,
                    "individual_kl": self.individual_kl,
                },
                f,
            )

        return cache_path

    def _load_search_state(self, path: str | Path) -> None:
        """Load a previous search state."""
        path = Path(path)

        if not path.exists():
            print(f"Warning: No search state found at {path}")
            return

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            # Restore saved state
            self.evaluated_groups.update(data.get("evaluated_groups", {}))

            # Verify compatibility
            if data.get("target_size") != self.target_size:
                print(
                    f"Warning: Loaded state has different target size ({data.get('target_size')}) than current ({self.target_size})"
                )

            # Log recovery
            print(f"Resumed search with {len(self.evaluated_groups)} cached evaluations from {path}")

        except Exception as e:
            print(f"Error loading search state: {e}")

    def progressive_beam_search(self, beam_width: int = 20, checkpoint_freq: int = 50) -> SearchResult:
        """Find the best neuron group using progressive beam search with checkpointing."""
        method = "progressive_beam"
        checkpoint_path = self._get_cache_path(method, "state")

        # Check if we have a saved state for this method
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, "rb") as f:
                    checkpoint = pickle.load(f)

                # Resume from the saved state
                current_beam = checkpoint["state"]["current_beam"]
                size = checkpoint["state"]["size"]
                best_final = checkpoint.get("state", {}).get("best_final")

                print(f"Resuming progressive beam search from size {size} with {len(current_beam)} beam candidates")

            except Exception as e:
                print(f"Failed to load checkpoint, starting from scratch: {e}")
                current_beam = None
                size = 1
                best_final = None
        else:
            current_beam = None
            size = 1
            best_final = None

        # If no valid state was loaded, initialize from scratch
        if current_beam is None:
            # Get individual neuron scores
            sorted_indices = self._get_sorted_neurons()

            # Initialize beam with top individual neurons
            current_beam = []
            for i in range(min(beam_width, len(sorted_indices))):
                idx, _ = sorted_indices[i]
                neuron = self.neurons[idx]
                mediation, kl = self.individual_mediation[idx], self.individual_kl[idx]
                current_beam.append(({neuron}, mediation, kl))

        # Track evaluations for checkpointing
        eval_count = 0

        # Expand to larger groups, starting from where we left off
        for current_size in range(size, self.target_size + 1):
            candidates = []
            sorted_indices = self._get_sorted_neurons()

            for group, _, _ in current_beam:
                # Try adding each neuron not already in the group
                for idx, _ in sorted_indices:
                    neuron = self.neurons[idx]
                    if neuron not in group:
                        new_group = group.union({neuron})

                        # Skip if already evaluated
                        if any(g == new_group for g, _, _ in candidates):
                            continue

                        # Evaluate the new group
                        mediation, kl = self._evaluate_group(list(new_group))
                        candidates.append((new_group, mediation, kl))

                        # Increment evaluation counter
                        eval_count += 1

                        # Checkpoint periodically
                        if eval_count % checkpoint_freq == 0:
                            # Save current state
                            state = {"current_beam": current_beam, "size": current_size, "best_final": best_final}
                            self._save_search_state(method, state)
                            print(f"Checkpoint saved after {eval_count} evaluations")

            # Sort candidates by mediation first, then KL
            candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)

            # Keep top beam_width candidates
            current_beam = candidates[:beam_width]

            # Save progress at the end of each size level
            state = {
                "current_beam": current_beam,
                "size": current_size + 1,  # Next size to process
                "best_final": best_final,
            }
            self._save_search_state(method, state)
            print(f"Completed size {current_size} expansion")

        # Find the best group of the target size
        best_group = max(current_beam, key=lambda x: (x[1], x[2]))
        result = SearchResult(neurons=list(best_group[0]), mediation=best_group[1], kl_divergence=best_group[2])

        # Save final result
        state = {
            "current_beam": current_beam,
            "size": self.target_size + 1,
            "best_final": (list(best_group[0]), best_group[1], best_group[2]),
        }
        self._save_search_state(method, state)

        return result

    def iterative_pruning(self, checkpoint_freq: int = 50) -> SearchResult:
        """Find the best neuron group using iterative pruning with checkpointing."""
        method = "iterative_pruning"
        checkpoint_path = self._get_cache_path(method, "state")

        # Check if we have a saved state for this method
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, "rb") as f:
                    checkpoint = pickle.load(f)

                # Resume from the saved state
                current_group = checkpoint["state"]["current_group"]
                best_scores = checkpoint["state"]["best_scores"]

                print(f"Resuming iterative pruning from size {len(current_group)}")

            except Exception as e:
                print(f"Failed to load checkpoint, starting from scratch: {e}")
                current_group = self.neurons.copy()
                best_scores = None
        else:
            # Start with all neurons
            current_group = self.neurons.copy()
            best_scores = None

        # Evaluate initial group if needed
        if best_scores is None and len(current_group) > self.target_size:
            mediation, kl = self._evaluate_group(current_group)
            best_scores = (mediation, kl)

        # Track evaluations for checkpointing
        eval_count = 0

        # Iteratively remove neurons until we reach target_size
        while len(current_group) > self.target_size:
            worst_neuron = None
            best_remaining_scores = (-float("inf"), -float("inf"))

            # Try removing each neuron and keep the best resulting group
            for n in current_group:
                test_group = [x for x in current_group if x != n]
                mediation, kl = self._evaluate_group(test_group)

                # Increment evaluation counter
                eval_count += 1

                # Determine if this is the best group after removal
                if (mediation, kl) > best_remaining_scores:
                    best_remaining_scores = (mediation, kl)
                    worst_neuron = n

                # Checkpoint periodically
                if eval_count % checkpoint_freq == 0:
                    # Save current state
                    state = {"current_group": current_group, "best_scores": best_scores}
                    self._save_search_state(method, state)
                    print(f"Checkpoint saved after {eval_count} evaluations")

            # Remove the worst neuron
            if worst_neuron is not None:
                current_group.remove(worst_neuron)
                best_scores = best_remaining_scores
            else:
                # This shouldn't happen, but break to avoid infinite loop
                break

            # Save progress after each neuron removal
            state = {"current_group": current_group, "best_scores": best_scores}
            self._save_search_state(method, state)
            print(f"Pruned to size {len(current_group)}")

        # Evaluate final group
        mediation, kl = self._evaluate_group(current_group)

        return SearchResult(neurons=current_group, mediation=mediation, kl_divergence=kl)

    def importance_weighted_sampling(
        self, n_iterations: int = 100, learning_rate: float = 0.1, checkpoint_freq: int = 20
    ) -> SearchResult:
        """Find the best neuron group using importance weighted sampling with checkpointing."""
        method = "importance_weighted"
        checkpoint_path = self._get_cache_path(method, "state")

        # Check if we have a saved state for this method
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, "rb") as f:
                    checkpoint = pickle.load(f)

                # Resume from the saved state
                weights = checkpoint["state"]["weights"]
                best_group = checkpoint["state"]["best_group"]
                best_scores = checkpoint["state"]["best_scores"]
                start_iteration = checkpoint["state"]["iteration"]

                print(f"Resuming importance weighted sampling from iteration {start_iteration}")

            except Exception as e:
                print(f"Failed to load checkpoint, starting from scratch: {e}")
                weights = None
                best_group = None
                best_scores = (-float("inf"), -float("inf"))
                start_iteration = 0
        else:
            weights = None
            best_group = None
            best_scores = (-float("inf"), -float("inf"))
            start_iteration = 0

        # Initialize weights if needed
        if weights is None:
            # Initialize weights from composite scores
            weights = np.array(self.composite_scores)

            # Ensure all weights are positive
            min_weight = min(weights)
            weights = weights - min_weight + 0.000001

            # Normalize to probabilities
            weights = weights / np.sum(weights)

        for i in range(start_iteration, n_iterations):
            # Sample neurons according to current weights
            if len(self.neurons) <= self.target_size:
                sampled_indices = np.arange(len(self.neurons))
            else:
                sampled_indices = np.random.choice(len(self.neurons), size=self.target_size, replace=False, p=weights)

            sampled_group = [self.neurons[idx] for idx in sampled_indices]

            # Evaluate group
            mediation, kl = self._evaluate_group(sampled_group)

            # Update best group
            if (mediation, kl) > best_scores:
                best_group = sampled_group
                best_scores = (mediation, kl)

                # Update weights for neurons in successful groups
                for idx in sampled_indices:
                    weights[idx] *= 1 + learning_rate

                # Re-normalize
                weights = weights / np.sum(weights)

            # Checkpoint periodically
            if (i + 1) % checkpoint_freq == 0:
                # Save current state
                state = {"weights": weights, "best_group": best_group, "best_scores": best_scores, "iteration": i + 1}
                self._save_search_state(method, state)
                print(f"Checkpoint saved after {i + 1} iterations")

        # Final checkpoint
        state = {"weights": weights, "best_group": best_group, "best_scores": best_scores, "iteration": n_iterations}
        self._save_search_state(method, state)

        return SearchResult(neurons=best_group, mediation=best_scores[0], kl_divergence=best_scores[1])

    def hybrid_search(self, n_init: int = 3, n_refinement: int = 50, checkpoint_freq: int = 20) -> SearchResult:
        """Find the best neuron group using a hybrid approach with checkpointing."""
        method = "hybrid"
        checkpoint_path = self._get_cache_path(method, "state")

        # Check if we have a saved state for this method
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, "rb") as f:
                    checkpoint = pickle.load(f)

                # Resume from the saved state
                current_group = checkpoint["state"]["current_group"]
                current_scores = checkpoint["state"]["current_scores"]
                refinement_iteration = checkpoint["state"]["refinement_iteration"]

                print(f"Resuming hybrid search from refinement iteration {refinement_iteration}")

            except Exception as e:
                print(f"Failed to load checkpoint, starting from scratch: {e}")
                current_group = None
                current_scores = None
                refinement_iteration = 0
        else:
            current_group = None
            current_scores = None
            refinement_iteration = 0

        # If no valid state was loaded, run the initialization phase
        if current_group is None:
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
                # Create probability distribution based on composite scores
                probs = np.array(self.composite_scores)
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
            best_scores = (-float("inf"), -float("inf"))

            for group in candidate_groups:
                mediation, kl = self._evaluate_group(group)
                if (mediation, kl) > best_scores:
                    best_group = group
                    best_scores = (mediation, kl)

            # Initialize refinement
            current_group = best_group.copy()
            current_scores = best_scores

            # Save checkpoint after initialization
            state = {
                "current_group": current_group,
                "current_scores": current_scores,
                "refinement_iteration": refinement_iteration,
            }
            self._save_search_state(method, state)
            print("Hybrid search initialization complete. Starting refinement.")

        # Refine the best group
        for i in range(refinement_iteration, n_refinement):
            improved = False

            # Try replacing each neuron in the group
            for idx, current_neuron in enumerate(current_group):
                for candidate_neuron in [n for n in self.neurons if n not in current_group]:
                    # Create a new group with one neuron replaced
                    test_group = current_group.copy()
                    test_group[idx] = candidate_neuron

                    # Evaluate the new group
                    mediation, kl = self._evaluate_group(test_group)

                    # Update if better
                    if (mediation, kl) > current_scores:
                        current_group = test_group
                        current_scores = (mediation, kl)
                        improved = True

            # Checkpoint periodically
            if (i + 1) % checkpoint_freq == 0:
                # Save current state
                state = {
                    "current_group": current_group,
                    "current_scores": current_scores,
                    "refinement_iteration": i + 1,
                }
                self._save_search_state(method, state)
                print(f"Checkpoint saved after {i + 1} refinement iterations")

            # Early stopping if no improvement
            if not improved:
                break

        # Final checkpoint
        state = {
            "current_group": current_group,
            "current_scores": current_scores,
            "refinement_iteration": n_refinement,  # Mark as complete
        }
        self._save_search_state(method, state)

        return SearchResult(neurons=current_group, mediation=current_scores[0], kl_divergence=current_scores[1])
