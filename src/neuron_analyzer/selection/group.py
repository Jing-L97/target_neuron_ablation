#!/usr/bin/env python
import logging
import random
import sys
import typing as t
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.cluster import AgglomerativeClustering
from torch.nn.functional import kl_div
from transformer_lens import utils

from neuron_analyzer.ablation.abl_util import get_entropy
from neuron_analyzer.ablation.ablation import adjust_vectors_3dim

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
sys.path.append("../")


#######################################################################################################
# Fucntions applied in the main scripts
#######################################################################################################


class NeuronAblation:
    """Class for performing ablation experiments on neural network components."""

    def __init__(
        self,
        components_to_ablate: list[str] | dict[str, list[tuple[int, int]]],
        unigram_distrib: torch.Tensor,
        tokenized_data: dict,
        entropy_df: pd.DataFrame,
        model: t.Any,
        k: int = 10,
        device: str = "mps",
        chunk_size: int = 20,
        ablation_mode: t.Literal["mean", "longtail"] = "mean",
        longtail_threshold: float = 0.001,
    ):
        self.components_to_ablate = components_to_ablate
        self.unigram_distrib = unigram_distrib
        self.tokenized_data = tokenized_data
        self.entropy_df = entropy_df
        self.model = model
        self.k = k
        self.device = device
        self.chunk_size = chunk_size
        self.ablation_mode = ablation_mode
        self.longtail_threshold = longtail_threshold

        # Processed data that will be set during initialization
        self.is_grouped = isinstance(components_to_ablate, dict)
        self.group_names = []
        self.group_to_indices = {}
        self.layer_idx = None
        self.activation_columns = []
        self.activation_mean_values = None
        self.unigram_direction = None
        self.longtail_mask = None

        # Initialize component information
        self._process_components()
        # Prepare unigram direction
        self._prepare_unigram_direction()
        # Calculate mean activations
        self._calculate_mean_activations()

    def _process_components(self) -> None:
        """Process component specification to extract group names, neuron indices, and layer information."""
        if self.is_grouped:
            # Using the group format
            self.group_names = list(self.components_to_ablate.keys())
            # Create a flattened list of all neurons for activation calculation
            all_neurons = []
            for group_neurons in self.components_to_ablate.values():
                all_neurons.extend(group_neurons)
            # Create mapping from (layer, neuron) tuples to component names
            self.layer_neuron_to_group = {}
            for group_name, neurons in self.components_to_ablate.items():
                for layer_idx, neuron_idx in neurons:
                    self.layer_neuron_to_group[(layer_idx, neuron_idx)] = group_name
            # Extract unique layer indices and check if they're all the same
            unique_layers = set(layer_idx for neurons in self.components_to_ablate.values() for layer_idx, _ in neurons)
            if len(unique_layers) > 1:
                logger.warning(f"Multiple layers detected: {unique_layers}. Using the first layer.")
            self.layer_idx = list(unique_layers)[0]
            # Get flat list of neuron indices
            self.all_neuron_indices = [neuron_idx for _, neuron_idx in all_neurons]
            # Dictionary mapping group names to their neuron indices
            self.group_to_indices = {
                group: [n_idx for _, n_idx in neurons] for group, neurons in self.components_to_ablate.items()
            }
            # Get activation column names
            self.activation_columns = [f"{self.layer_idx}.{neuron_idx}_activation" for _, neuron_idx in all_neurons]

        else:
            # Original format: list of "layer.neuron" strings
            self.group_names = self.components_to_ablate
            # Extract neuron and layer indices
            self.neuron_indices = [int(name.split(".")[1]) for name in self.components_to_ablate]
            layer_indices = [int(name.split(".")[0]) for name in self.components_to_ablate]
            self.all_neuron_indices = self.neuron_indices
            self.layer_idx = layer_indices[0]
            # One-to-one mapping of components_to_ablate
            self.group_to_indices = {name: [int(name.split(".")[1])] for name in self.components_to_ablate}
            # Get activation column names
            self.activation_columns = [f"{name}_activation" for name in self.components_to_ablate]

    def _prepare_unigram_direction(self) -> None:
        """Prepare the unigram direction based on the ablation mode."""
        if self.ablation_mode == "longtail":
            # Create long-tail token mask (1 for long-tail tokens, 0 for common tokens)
            self.longtail_mask = (self.unigram_distrib < self.longtail_threshold).float()
            logger.info(
                f"Number of long-tail tokens: {self.longtail_mask.sum().item()} out of {len(self.longtail_mask)}"
            )
            # Original token frequency vector from the unigram distribution
            full_unigram_direction = self.unigram_distrib.log() - self.unigram_distrib.log().mean()
            full_unigram_direction /= full_unigram_direction.norm()

            # Modified token frequency vector that zeros out common tokens
            self.unigram_direction = full_unigram_direction * self.longtail_mask
            # Re-normalize to keep it a unit vector
            if self.unigram_direction.norm() > 0:
                self.unigram_direction /= self.unigram_direction.norm()
        else:
            # Standard frequency direction for regular mean ablation
            self.unigram_direction = self.unigram_distrib.log() - self.unigram_distrib.log().mean()
            self.unigram_direction /= self.unigram_direction.norm()

    def _calculate_mean_activations(self) -> None:
        """Calculate mean activation values for each neuron."""
        self.activation_mean_values = torch.tensor(self.entropy_df[self.activation_columns].mean())

    def _get_group_activation_means(self, group_name: str) -> torch.Tensor:
        """Get mean activation values for a specific group."""
        if self.is_grouped:
            # Extract group-specific neurons
            group_neuron_indices = self.group_to_indices[group_name]
            group_activation_cols = [f"{self.layer_idx}.{neuron_idx}_activation" for neuron_idx in group_neuron_indices]
            return torch.tensor(self.entropy_df[group_activation_cols].mean())
        # In the original format, each group is just one neuron
        group_idx = self.components_to_ablate.index(group_name)
        return torch.tensor([self.activation_mean_values[group_idx]])

    def _get_group_activation(self, group_name: str, cache: dict) -> torch.Tensor:
        """Get neuron activation for a specific group from the cache."""
        group_neuron_indices = self.group_to_indices[group_name]
        if self.is_grouped:
            return cache[utils.get_act_name("post", self.layer_idx)][0, :, group_neuron_indices]
        group_idx = self.components_to_ablate.index(group_name)
        return cache[utils.get_act_name("post", self.layer_idx)][0, :, [self.neuron_indices[group_idx]]]

    def _compute_activation_deltas(self, group_name: str, cache: dict) -> torch.Tensor:
        """Compute activation deltas for a group."""
        previous_activation = self._get_group_activation(group_name, cache)
        group_activation_means = self._get_group_activation_means(group_name)
        # Calculate deltas
        return group_activation_means.to(previous_activation.device) - previous_activation

    def _compute_residual_deltas(
        self, group_name: str, activation_deltas: torch.Tensor, res_stream: torch.Tensor
    ) -> torch.Tensor:
        """Compute residual stream deltas based on activation deltas."""
        group_neuron_indices = self.group_to_indices[group_name]

        # Initialize residual deltas
        res_deltas = torch.zeros_like(res_stream).unsqueeze(0)

        # For each neuron, add its contribution to the residual stream
        for i, neuron_idx in enumerate(group_neuron_indices):
            neuron_delta = (
                activation_deltas[:, i].unsqueeze(-1)
                if activation_deltas.dim() > 1
                else activation_deltas.unsqueeze(-1)
            )
            res_deltas += neuron_delta * self.model.W_out[self.layer_idx, neuron_idx, :]

        return res_deltas

    def _process_chunk(
        self,
        group_name: str,
        res_stream: torch.Tensor,
        res_deltas: torch.Tensor,
        unigram_projection_values: torch.Tensor,
        log_unigram_distrib: torch.Tensor,
        inp: torch.Tensor,
        logits: torch.Tensor,
        i: int,
        chunk_end: int,
    ) -> dict:
        """Process a chunk of the residual stream for ablation."""
        # Extract chunk
        res_stream_chunk = res_stream[i:chunk_end].unsqueeze(0)
        res_deltas_chunk = res_deltas[:, i:chunk_end, :]

        # Update residual stream with deltas
        updated_res_stream_chunk = res_stream_chunk + res_deltas_chunk
        # Apply layer normalization
        updated_res_stream_chunk = self.model.ln_final(updated_res_stream_chunk)
        # Project to logit space
        ablated_logits_chunk = updated_res_stream_chunk @ self.model.W_U + self.model.b_U
        # If we're in long-tail mode, apply the mask
        if self.ablation_mode == "longtail":
            # Get the original logits to preserve for common tokens
            original_logits = logits[:, i:chunk_end, :]
            # Create a binary mask for the vocabulary dimension
            vocab_mask = self.longtail_mask.unsqueeze(0).unsqueeze(0)  # Shape: 1 x 1 x vocab_size
            # Apply the mask: use original_logits where mask is 0, use ablated_logits where mask is 1
            ablated_logits_chunk = (1 - vocab_mask) * original_logits + vocab_mask * ablated_logits_chunk

        # Adjust vectors to maintain unigram projections
        unigram_dir = self.unigram_direction
        proj_values = unigram_projection_values[i:chunk_end]

        # Use adjust_vectors_3dim function to maintain unigram projections
        ablated_logits_with_frozen_unigram_chunk = adjust_vectors_3dim(ablated_logits_chunk, unigram_dir, proj_values)

        # Compute loss for the chunk
        loss_post_ablation_chunk = self.model.loss_fn(ablated_logits_chunk, inp[:, i:chunk_end], per_token=True).cpu()
        loss_post_ablation_chunk = np.concatenate(
            (loss_post_ablation_chunk, np.zeros((loss_post_ablation_chunk.shape[0], 1))), axis=1
        )

        # Compute entropy for the chunk
        entropy_post_ablation_chunk = get_entropy(ablated_logits_chunk)
        # Compute log probabilities
        abl_logprobs = ablated_logits_chunk.log_softmax(dim=-1)
        # Compute loss for ablated_logits_with_frozen_unigram_chunk
        loss_post_ablation_with_frozen_unigram_chunk = self.model.loss_fn(
            ablated_logits_with_frozen_unigram_chunk, inp[:, i:chunk_end], per_token=True
        ).cpu()
        loss_post_ablation_with_frozen_unigram_chunk = np.concatenate(
            (
                loss_post_ablation_with_frozen_unigram_chunk,
                np.zeros((loss_post_ablation_with_frozen_unigram_chunk.shape[0], 1)),
            ),
            axis=1,
        )
        # Compute entropy for ablated_logits_with_frozen_unigram_chunk
        entropy_post_ablation_with_frozen_unigram_chunk = get_entropy(ablated_logits_with_frozen_unigram_chunk)
        # Compute KL divergence between the ablated distribution and the unigram distribution
        kl_divergence_after_chunk = (
            kl_div(abl_logprobs, log_unigram_distrib.expand_as(abl_logprobs), reduction="none", log_target=True)
            .sum(axis=-1)
            .cpu()
            .numpy()
        )
        # Compute KL divergence for frozen unigram
        abl_logprobs_with_frozen_unigram = ablated_logits_with_frozen_unigram_chunk.log_softmax(dim=-1)
        if self.ablation_mode == "longtail":
            # For long-tail mode, compute KL divergence with focus on the long-tail tokens
            masked_logprobs = abl_logprobs_with_frozen_unigram.clone()
            masked_logprobs = masked_logprobs + (1 - self.longtail_mask).unsqueeze(0).unsqueeze(0) * -1e10
            masked_logprobs = torch.nn.functional.log_softmax(masked_logprobs, dim=-1)
            kl_divergence_after_frozen_unigram_chunk = (
                kl_div(
                    masked_logprobs,
                    log_unigram_distrib.expand_as(masked_logprobs),
                    reduction="none",
                    log_target=True,
                )
                .sum(axis=-1)
                .cpu()
                .numpy()
            )
        else:
            # Standard KL divergence for regular mean ablation
            kl_divergence_after_frozen_unigram_chunk = (
                kl_div(
                    abl_logprobs_with_frozen_unigram,
                    log_unigram_distrib.expand_as(abl_logprobs_with_frozen_unigram),
                    reduction="none",
                    log_target=True,
                )
                .sum(axis=-1)
                .cpu()
                .numpy()
            )
        # Return chunk metrics
        return {
            "loss_post_ablation": loss_post_ablation_chunk,
            "entropy_post_ablation": entropy_post_ablation_chunk.cpu(),
            "loss_post_ablation_with_frozen_unigram": loss_post_ablation_with_frozen_unigram_chunk,
            "entropy_post_ablation_with_frozen_unigram": entropy_post_ablation_with_frozen_unigram_chunk.cpu(),
            "kl_divergence_after": kl_divergence_after_chunk,
            "kl_divergence_after_frozen_unigram": kl_divergence_after_frozen_unigram_chunk,
        }

    def _create_result_dataframe(
        self, batch_data: pd.DataFrame, group_name: str, group_metrics: dict, kl_divergence_before: np.ndarray
    ) -> pd.DataFrame:
        """Create a DataFrame with ablation results for a group."""
        df_to_append = batch_data.copy()

        # If using the grouped format, create a combined activation column
        if self.is_grouped:
            # Create a new activation column that represents the group
            group_neuron_indices = self.group_to_indices[group_name]
            group_activation_cols = [f"{self.layer_idx}.{neuron_idx}_activation" for neuron_idx in group_neuron_indices]
            # Calculate mean activation for the group
            df_to_append["activation"] = df_to_append[group_activation_cols].mean(axis=1)

            # Drop individual neuron activation columns
            df_to_append = df_to_append.drop(
                columns=[col for col in df_to_append.columns if "_activation" in col and col != "activation"]
            )
        else:
            # Original format - keep only the relevant neuron's activation
            # drop all the columns that are not the component_name
            df_to_append = df_to_append.drop(
                columns=[f"{neuron}_activation" for neuron in self.components_to_ablate if neuron != group_name]
            )

            # rename the component_name column to 'activation'
            df_to_append = df_to_append.rename(columns={f"{group_name}_activation": "activation"})

        # Add group name and results
        df_to_append["component_name"] = group_name
        df_to_append["loss_post_ablation"] = group_metrics["loss_post_ablation"][0]
        df_to_append["loss_post_ablation_with_frozen_unigram"] = group_metrics[
            "loss_post_ablation_with_frozen_unigram"
        ][0]
        df_to_append["entropy_post_ablation"] = group_metrics["entropy_post_ablation"][0]
        df_to_append["entropy_post_ablation_with_frozen_unigram"] = group_metrics[
            "entropy_post_ablation_with_frozen_unigram"
        ][0]
        df_to_append["kl_divergence_before"] = kl_divergence_before
        df_to_append["kl_divergence_after"] = group_metrics["kl_divergence_after"][0]
        df_to_append["kl_divergence_after_frozen_unigram"] = group_metrics["kl_divergence_after_frozen_unigram"][0]

        # Add ablation information
        df_to_append["ablation_mode"] = self.ablation_mode
        if self.ablation_mode == "longtail":
            df_to_append["longtail_threshold"] = self.longtail_threshold
            df_to_append["num_longtail_tokens"] = self.longtail_mask.sum().item()

        # For group format, add information about which neurons are in the group
        if self.is_grouped:
            df_to_append["group_neurons"] = str(self.group_to_indices[group_name])
            df_to_append["num_neurons_in_group"] = len(self.group_to_indices[group_name])

        return df_to_append

    def run(self) -> dict[int, pd.DataFrame]:
        """Run the ablation experiment."""
        # Sample random sequence indices
        random_sequence_indices = np.random.choice(self.entropy_df.batch.unique(), self.k, replace=False)

        logger.info(f"ablate_components: ablate with k = {self.k}, long-tail threshold = {self.longtail_threshold}")

        pbar = tqdm.tqdm(total=self.k, file=sys.stdout)

        # Filter entropy_df for the random sequences
        filtered_entropy_df = self.entropy_df[self.entropy_df.batch.isin(random_sequence_indices)].copy()

        results = {}

        # Process each batch
        for batch_n in filtered_entropy_df.batch.unique():
            tok_seq = self.tokenized_data["tokens"][batch_n]

            # Get unaltered logits
            self.model.reset_hooks()
            inp = tok_seq.unsqueeze(0).to(self.device)
            logits, cache = self.model.run_with_cache(inp)
            logprobs = logits[0, :, :].log_softmax(dim=-1)

            res_stream = cache[utils.get_act_name("resid_post", self.layer_idx)][0]

            # Get the entropy_df entries for the current sequence
            batch_data = filtered_entropy_df[filtered_entropy_df.batch == batch_n]
            assert len(batch_data) == len(tok_seq), (
                f"len(batch_data) = {len(batch_data)}, len(tok_seq) = {len(tok_seq)}"
            )

            # Get the value of the logits projected onto the unigram direction
            unigram_projection_values = logits @ self.unigram_direction
            unigram_projection_values = unigram_projection_values.squeeze()

            # Get the log of unigram distribution for KL divergence calculation
            log_unigram_distrib = self.unigram_distrib.log()

            # Calculate KL divergence before ablation
            kl_divergence_before = (
                kl_div(logprobs, log_unigram_distrib, reduction="none", log_target=True).sum(axis=-1).cpu().numpy()
            )

            # Process each group
            batch_results = []
            for group_name in self.group_names:
                # Compute activation deltas
                activation_deltas = self._compute_activation_deltas(group_name, cache)

                # Compute residual stream deltas
                res_deltas = self._compute_residual_deltas(group_name, activation_deltas, res_stream)

                # Initialize result containers
                chunk_results = {
                    "loss_post_ablation": [],
                    "entropy_post_ablation": [],
                    "loss_post_ablation_with_frozen_unigram": [],
                    "entropy_post_ablation_with_frozen_unigram": [],
                    "kl_divergence_after": [],
                    "kl_divergence_after_frozen_unigram": [],
                }

                # Process in chunks for memory efficiency
                for i in range(0, res_stream.shape[0], self.chunk_size):
                    chunk_end = min(i + self.chunk_size, res_stream.shape[0])

                    # Process this chunk
                    chunk_result = self._process_chunk(
                        group_name,
                        res_stream,
                        res_deltas,
                        unigram_projection_values,
                        log_unigram_distrib,
                        inp,
                        logits,
                        i,
                        chunk_end,
                    )

                    # Append chunk results
                    for key, value in chunk_result.items():
                        chunk_results[key].append(value)

                # Concatenate results
                group_metrics = {key: np.concatenate(value, axis=0) for key, value in chunk_results.items()}

                # Create result dataframe for this group
                result_df = self._create_result_dataframe(batch_data, group_name, group_metrics, kl_divergence_before)

                batch_results.append(result_df)

            # Combine all group results for this batch
            results[batch_n] = pd.concat(batch_results) if batch_results else None

            # Clean up to save memory
            del res_stream
            del cache
            torch.cuda.empty_cache()

            pbar.update(1)

        return results


#######################################################################################################
# Fuctions applied in the main scripts
#######################################################################################################


@dataclass
class SearchResult:
    """Result of a neuron group search."""

    neurons: list[int]
    mediation: float
    kl_divergence: float

    def __repr__(self) -> str:
        return f"SearchResult(neurons={self.neurons}, mediation={self.mediation:.4f}, kl={self.kl_divergence:.4f})"


class NeuronGroupSearch:
    """Class for finding optimal neuron groups using various search strategies."""

    def __init__(
        self,
        neurons: list[int],
        evaluation_fn: Callable[[list[int]], tuple[float, float]],
        target_size: int,
        individual_mediation: list[float] | None = None,
        individual_kl: list[float] | None = None,
    ):
        """Initialize the neuron group search."""
        self.neurons = neurons
        self.evaluation_fn = evaluation_fn
        self.target_size = min(target_size, len(neurons))

        # Compute individual scores if not provided
        if individual_mediation is None or individual_kl is None:
            self._compute_individual_scores()
        else:
            self.individual_mediation = individual_mediation
            self.individual_kl = individual_kl

        # Derived information
        self.composite_scores = [
            self.individual_mediation[i] + 0.01 * self.individual_kl[i] for i in range(len(self.neurons))
        ]

    def _compute_individual_scores(self) -> None:
        """Compute mediation and KL scores for individual neurons."""
        self.individual_mediation = []
        self.individual_kl = []

        for i, neuron in enumerate(self.neurons):
            mediation, kl = self.evaluation_fn([neuron])
            self.individual_mediation.append(mediation)
            self.individual_kl.append(kl)

    def _evaluate_group(self, group: list[int]) -> tuple[float, float]:
        """Evaluate a neuron group using the evaluation function."""
        return self.evaluation_fn(group)

    def _get_sorted_neurons(self) -> list[tuple[int, float]]:
        """Get neurons sorted by their composite scores (mediation prioritized)."""
        scores = [(i, self.composite_scores[i]) for i in range(len(self.neurons))]
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def progressive_beam_search(self, beam_width: int = 20) -> SearchResult:
        """Find the best neuron group using progressive beam search."""
        # Get individual neuron scores
        sorted_indices = self._get_sorted_neurons()

        # Initialize beam with top individual neurons
        current_beam = []
        for i in range(min(beam_width, len(sorted_indices))):
            idx, _ = sorted_indices[i]
            neuron = self.neurons[idx]
            mediation, kl = self.individual_mediation[idx], self.individual_kl[idx]
            current_beam.append(({neuron}, mediation, kl))

        # Expand to larger groups
        for size in range(2, self.target_size + 1):
            candidates = []

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

            # Sort candidates by mediation first, then KL
            candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)

            # Keep top beam_width candidates
            current_beam = candidates[:beam_width]

        # Return the best group of the target size
        best_group = max(current_beam, key=lambda x: (x[1], x[2]))
        return SearchResult(neurons=list(best_group[0]), mediation=best_group[1], kl_divergence=best_group[2])

    def hierarchical_cluster_search(self, n_clusters: int = 5, expansion_factor: int = 3) -> SearchResult:
        """Find the best neuron group using hierarchical clustering-based search."""
        # Create feature vectors for clustering using both heuristics
        features = np.array([(self.individual_mediation[i], self.individual_kl[i]) for i in range(len(self.neurons))])

        # Normalize features
        if features.std(axis=0).min() > 0:  # Avoid division by zero
            features = (features - features.mean(axis=0)) / features.std(axis=0)

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
            # Sort neurons in this cluster by mediation and KL
            sorted_cluster = sorted(
                [
                    (self.neurons[idx], self.individual_mediation[idx], self.individual_kl[idx])
                    for idx in cluster_neurons
                ],
                key=lambda x: (x[1], x[2]),
                reverse=True,
            )

            # Take top representatives from each cluster
            representatives.extend([n for n, _, _ in sorted_cluster[:expansion_factor]])

        # If we have fewer representatives than target_size, use all representatives
        if len(representatives) <= self.target_size:
            mediation, kl = self._evaluate_group(representatives)
            return SearchResult(neurons=representatives, mediation=mediation, kl_divergence=kl)

        # Try various combinations to find the best group
        # Start with greedy selection based on individual scores
        sorted_reps = sorted(
            [
                (n, self.individual_mediation[self.neurons.index(n)], self.individual_kl[self.neurons.index(n)])
                for n in representatives
            ],
            key=lambda x: (x[1], x[2]),
            reverse=True,
        )

        # Start with top scoring neurons
        current_group = [n for n, _, _ in sorted_reps[: min(5, self.target_size)]]

        # Complete the group
        while len(current_group) < self.target_size:
            best_candidate = None
            best_candidate_scores = (-float("inf"), -float("inf"))

            for n, _, _ in [(n, m, k) for n, m, k in sorted_reps if n not in current_group]:
                test_group = current_group + [n]
                mediation, kl = self._evaluate_group(test_group)

                if (mediation, kl) > best_candidate_scores:
                    best_candidate = n
                    best_candidate_scores = (mediation, kl)

            if best_candidate is not None:
                current_group.append(best_candidate)
            else:
                break

        # Evaluate the final group
        mediation, kl = self._evaluate_group(current_group)
        best_group = current_group
        best_scores = (mediation, kl)

        # Refine the group with local search
        for _ in range(min(10, len(best_group))):
            for i, n in enumerate(best_group):
                # Try replacing each neuron in the group
                for candidate in [neuron for neuron in self.neurons if neuron not in best_group]:
                    test_group = [x for x in best_group if x != n] + [candidate]
                    mediation, kl = self._evaluate_group(test_group)

                    if (mediation, kl) > best_scores:
                        best_group = test_group
                        best_scores = (mediation, kl)

        return SearchResult(neurons=best_group, mediation=best_scores[0], kl_divergence=best_scores[1])

    def iterative_pruning(self) -> SearchResult:
        """Find the best neuron group using iterative pruning."""
        # Start with all neurons
        current_group = self.neurons.copy()

        # Iteratively remove neurons until we reach target_size
        while len(current_group) > self.target_size:
            worst_neuron = None
            best_remaining_scores = (-float("inf"), -float("inf"))

            # Try removing each neuron and keep the best resulting group
            for n in current_group:
                test_group = [x for x in current_group if x != n]
                mediation, kl = self._evaluate_group(test_group)

                # Determine if this is the best group after removal
                if (mediation, kl) > best_remaining_scores:
                    best_remaining_scores = (mediation, kl)
                    worst_neuron = n

            # Remove the worst neuron
            if worst_neuron is not None:
                current_group.remove(worst_neuron)
            else:
                # This shouldn't happen, but break to avoid infinite loop
                break

        # Evaluate final group
        mediation, kl = self._evaluate_group(current_group)

        return SearchResult(neurons=current_group, mediation=mediation, kl_divergence=kl)

    def importance_weighted_sampling(self, n_iterations: int = 100, learning_rate: float = 0.1) -> SearchResult:
        """Find the best neuron group using importance weighted sampling."""
        # Initialize weights from composite scores
        weights = np.array(self.composite_scores)

        # Ensure all weights are positive
        min_weight = min(weights)
        weights = weights - min_weight + 0.000001

        # Normalize to probabilities
        weights = weights / np.sum(weights)

        best_group = None
        best_scores = (-float("inf"), -float("inf"))

        for _ in range(n_iterations):
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

        return SearchResult(neurons=best_group, mediation=best_scores[0], kl_divergence=best_scores[1])

    def hybrid_search(self, n_init: int = 3, n_refinement: int = 50) -> SearchResult:
        """Find the best neuron group using a hybrid approach."""
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

        # Refine the best group
        current_group = best_group.copy()
        current_scores = best_scores

        for _ in range(n_refinement):
            improved = False

            # Try replacing each neuron in the group
            for i, current_neuron in enumerate(current_group):
                for candidate_neuron in [n for n in self.neurons if n not in current_group]:
                    # Create a new group with one neuron replaced
                    test_group = current_group.copy()
                    test_group[i] = candidate_neuron

                    # Evaluate the new group
                    mediation, kl = self._evaluate_group(test_group)

                    # Update if better
                    if (mediation, kl) > current_scores:
                        current_group = test_group
                        current_scores = (mediation, kl)
                        improved = True

            # Early stopping if no improvement
            if not improved:
                break

        return SearchResult(neurons=current_group, mediation=current_scores[0], kl_divergence=current_scores[1])

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
        # Find the best method (prioritizing mediation, then KL)
        best_method = max(results.items(), key=lambda x: (x[1].mediation, x[1].kl_divergence))
        return best_method
