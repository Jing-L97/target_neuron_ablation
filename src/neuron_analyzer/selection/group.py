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
import tqdm
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


class GroupModelAblationAnalyzer:
    """Class for analyzing neural network components through group ablations."""

    def __init__(
        self,
        model,
        unigram_distrib,
        tokenized_data,
        entropy_df,
        neuron_groups: list[list[str]],  # List of neuron groups, each group is a list of neuron names
        group_names: list[str] | None = None,  # Optional names for each group
        device: str = "cuda",
        k: int = 10,
        chunk_size: int = 20,
        ablation_mode: str = "mean",
        longtail_threshold: float = 0.001,
    ):
        """Initialize the GroupModelAblationAnalyzer."""
        self.model = model
        self.unigram_distrib = unigram_distrib
        self.tokenized_data = tokenized_data
        self.entropy_df = entropy_df
        self.device = device
        self.k = k
        self.chunk_size = chunk_size
        self.ablation_mode = ablation_mode
        self.longtail_threshold = longtail_threshold
        self.results = {}
        self.log_unigram_distrib = self.unigram_distrib.log()

        # Store neuron groups and set up group names
        self.neuron_groups = neuron_groups
        if group_names is None:
            self.group_names = [f"group_{i}" for i in range(len(neuron_groups))]
        else:
            self.group_names = group_names

        # This will be our components to ablate
        self.components_to_ablate = self.group_names

        # Validate that all neurons in each group are from the same layer
        for i, group in enumerate(neuron_groups):
            layers = [int(neuron.split(".")[0]) for neuron in group]
            if len(set(layers)) > 1:
                raise ValueError(f"Group {i} contains neurons from different layers: {layers}")

        # Store the layer for each group
        self.group_layers = [int(group[0].split(".")[0]) for group in neuron_groups]

        # Map each group to its neuron indices
        self.group_neuron_indices = [[int(neuron.split(".")[1]) for neuron in group] for group in neuron_groups]

        # Check if there's a component_name column which might indicate the neuron name
        if "component_name" in self.entropy_df.columns:
            logger.info("Found 'component_name' column - using this for neuron identification")
            self.has_component_name = True
        else:
            self.has_component_name = False

        # Check if there's a single activation column
        if "activation" in self.entropy_df.columns:
            logger.info("Found single 'activation' column - will use with component_name")
            self.has_single_activation = True
        else:
            self.has_single_activation = False
            # Check for other possible activation columns
            activation_cols = [col for col in self.entropy_df.columns if "_activation" in str(col)]
            if not activation_cols and not self.has_single_activation:
                raise ValueError("Cannot find any activation columns in the DataFrame")

        # Check the required columns for processing
        required_columns = ["batch"]
        for col in required_columns:
            if col not in self.entropy_df.columns:
                raise ValueError(f"Required column {col} not found in entropy_df")

    def build_vector(self) -> None:
        """Build frequency-related vectors for ablation analysis."""
        if self.ablation_mode == "longtail":
            # Create long-tail token mask (1 for long-tail tokens, 0 for common tokens)
            self.longtail_mask = (self.unigram_distrib < self.longtail_threshold).float()
            logger.info(
                f"Number of long-tail tokens: {self.longtail_mask.sum().item()} out of {len(self.longtail_mask)}"
            )

            # Create token frequency vector focusing on long-tail tokens only
            # Original token frequency vector from the unigram distribution
            full_unigram_direction_vocab = self.unigram_distrib.log() - self.unigram_distrib.log().mean()
            full_unigram_direction_vocab /= full_unigram_direction_vocab.norm()

            # This makes the vector only consider contributions from long-tail tokens
            self.unigram_direction_vocab = full_unigram_direction_vocab * self.longtail_mask
            # Re-normalize to keep it a unit vector
            if self.unigram_direction_vocab.norm() > 0:
                self.unigram_direction_vocab /= self.unigram_direction_vocab.norm()
        else:
            # Standard frequency direction for regular mean ablation
            self.unigram_direction_vocab = self.unigram_distrib.log() - self.unigram_distrib.log().mean()
            self.unigram_direction_vocab /= self.unigram_direction_vocab.norm()
            self.longtail_mask = None

    def project_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Get the value of logits projected onto the unigram direction."""
        unigram_projection_values = logits @ self.unigram_direction_vocab
        return unigram_projection_values.squeeze()

    def build_group_activation_means(self, filtered_entropy_df: pd.DataFrame) -> list:
        """Build mean activation values for each neuron in each group."""
        # For each group, compute the mean activation of each neuron in the group
        group_activation_means = []
        for group in self.neuron_groups:
            # Create a list to store means for each neuron in the group
            neuron_means = []
            for neuron in group:
                # Filter rows where component_name matches this neuron
                neuron_rows = filtered_entropy_df[filtered_entropy_df["component_name"] == neuron]
                # Get the mean activation for this neuron
                mean_activation = neuron_rows["activation"].mean()
                neuron_means.append(mean_activation)
            # Convert to tensor and add to group means
            group_means = torch.tensor(neuron_means)
            group_activation_means.append(group_means)
        return group_activation_means

    def adjust_vectors_3dim(self, v: torch.Tensor, u: torch.Tensor, target_values: torch.Tensor) -> torch.Tensor:
        """Adjusts a batch of vectors v such that their projections along the unit vector u equal the target values."""
        current_projections = (v @ u.unsqueeze(-1)).squeeze(-1)  # Current projections of v onto u
        delta = target_values - current_projections  # Differences needed to reach the target projections
        adjusted_v = v + delta.unsqueeze(-1) * u  # Adjust v by the deltas along the direction of u
        return adjusted_v

    def project_neurons(
        self,
        logits: torch.Tensor,
        unigram_projection_values: torch.Tensor,
        ablated_logits_chunk: torch.Tensor,
        res_deltas_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Project neurons based on ablation mode and adjust vectors."""
        # If we're in long-tail mode, apply the mask
        if self.ablation_mode == "longtail":
            # Get the original logits to preserve for common tokens
            original_logits = logits.repeat(res_deltas_chunk.shape[0], 1, 1)
            # Create a binary mask for the vocabulary dimension
            # 1 for long-tail tokens (to be modified), 0 for common tokens (to keep original)
            vocab_mask = self.longtail_mask.unsqueeze(0).unsqueeze(0)  # Shape: 1 x 1 x vocab_size
            # Apply the mask: use original_logits where mask is 0, use ablated_logits where mask is 1
            ablated_logits_chunk = (1 - vocab_mask) * original_logits + vocab_mask * ablated_logits_chunk

        # Adjust vectors to maintain unigram projections
        ablated_logits_with_frozen_unigram_chunk = self.adjust_vectors_3dim(
            ablated_logits_chunk, self.unigram_direction_vocab, unigram_projection_values
        )
        return ablated_logits_with_frozen_unigram_chunk

    def get_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Calculate entropy from logits."""
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(-1)
        return entropy

    def kl_div(
        self,
        input_logprobs: torch.Tensor,
        target_logprobs: torch.Tensor,
        reduction: str = "none",
        log_target: bool = True,
    ) -> torch.Tensor:
        """Calculate KL divergence between distributions."""
        if log_target:
            kl = torch.exp(input_logprobs) * (input_logprobs - target_logprobs)
        else:
            kl = torch.exp(input_logprobs) * (input_logprobs - torch.log(target_logprobs))

        if reduction == "none":
            return kl
        if reduction == "sum":
            return kl.sum()
        if reduction == "mean":
            return kl.mean()
        raise ValueError(f"Unsupported reduction: {reduction}")

    def compute_kl(
        self,
        ablated_logits_with_frozen_unigram_chunk: torch.Tensor,
        abl_logprobs: torch.Tensor,
        kl_divergence_after: list,
        kl_divergence_after_frozen_unigram: list,
    ) -> tuple[list, list]:
        """Compute KL divergence metrics for ablated distributions."""
        # compute KL divergence between the distribution ablated with frozen unigram and the og distribution
        abl_logprobs_with_frozen_unigram = ablated_logits_with_frozen_unigram_chunk.log_softmax(dim=-1)

        # compute KL divergence between the ablated distribution and the distribution from the unigram direction
        kl_divergence_after_chunk = (
            self.kl_div(
                abl_logprobs, self.log_unigram_distrib.expand_as(abl_logprobs), reduction="none", log_target=True
            )
            .sum(axis=-1)
            .cpu()
            .numpy()
        )

        del abl_logprobs
        kl_divergence_after.append(kl_divergence_after_chunk)

        if self.ablation_mode == "longtail":
            # For long-tail mode, compute KL divergence with focus on the long-tail tokens
            masked_logprobs = abl_logprobs_with_frozen_unigram.clone()
            masked_logprobs = masked_logprobs + (1 - self.longtail_mask).unsqueeze(0).unsqueeze(0) * -1e10
            masked_logprobs = torch.nn.functional.log_softmax(masked_logprobs, dim=-1)
            kl_divergence_after_frozen_unigram_chunk = (
                self.kl_div(
                    masked_logprobs,
                    self.log_unigram_distrib.expand_as(masked_logprobs),
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
                self.kl_div(
                    abl_logprobs_with_frozen_unigram,
                    self.log_unigram_distrib.expand_as(abl_logprobs_with_frozen_unigram),
                    reduction="none",
                    log_target=True,
                )
                .sum(axis=-1)
                .cpu()
                .numpy()
            )

        del abl_logprobs_with_frozen_unigram
        kl_divergence_after_frozen_unigram.append(kl_divergence_after_frozen_unigram_chunk)

        del ablated_logits_with_frozen_unigram_chunk
        return kl_divergence_after, kl_divergence_after_frozen_unigram

    def mean_ablate_components(self) -> dict[int, pd.DataFrame]:
        """Perform mean ablation on specified neuron groups."""
        # Sample a set of random batch indices
        try:
            random_sequence_indices = np.random.choice(self.entropy_df.batch.unique(), self.k, replace=False)
        except ValueError as e:
            logger.error(f"Error sampling batch indices: {e}")
            # If there are fewer batches than k, use all available batches
            random_sequence_indices = self.entropy_df.batch.unique()
            logger.info(f"Using all available batches: {len(random_sequence_indices)}")

        logger.info(f"ablate_components: ablate with k = {self.k}, long-tail threshold = {self.longtail_threshold}")

        pbar = tqdm.tqdm(total=self.k, file=sys.stdout)

        # new_entropy_df with only the random sequences
        filtered_entropy_df = self.entropy_df[self.entropy_df.batch.isin(random_sequence_indices)].copy()

        # Build group activation means
        try:
            group_activation_means = self.build_group_activation_means(filtered_entropy_df)
        except Exception as e:
            logger.error(f"Error building group activation means: {e}")
            raise

        # Build frequency vectors
        self.build_vector()

        for batch_n in filtered_entropy_df.batch.unique():
            tok_seq = self.tokenized_data["tokens"][batch_n]

            # get unaltered logits
            self.model.reset_hooks()
            inp = tok_seq.unsqueeze(0).to(self.device)
            logits, cache = self.model.run_with_cache(inp)
            logprobs = logits[0, :, :].log_softmax(dim=-1)

            # get the entropy_df entries for the current sequence
            rows = filtered_entropy_df[filtered_entropy_df.batch == batch_n]
            assert len(rows) == len(tok_seq), f"len(rows) = {len(rows)}, len(tok_seq) = {len(tok_seq)}"
            # get the value of the logits projected onto the b_U direction
            unigram_projection_values = self.project_logits(logits)

            # Process each group
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

            logger.info("Finish computing KL")

            """
            # Process each neuron group
            for group_idx, group_name in enumerate(self.group_names):
                logger.info(f"Processing group {group_idx}")
                layer_idx = self.group_layers[group_idx]
                neuron_indices = self.group_neuron_indices[group_idx]

                logger.info("Finish computing neuron indices")

                # Get activations for all neurons in this group
                res_stream = cache[utils.get_act_name("resid_post", layer_idx)][0]
                previous_activations = cache[utils.get_act_name("post", layer_idx)][0, :, neuron_indices]

                logger.info("Finish computing activation")

                # Get mean activations for this group
                group_mean = group_activation_means[group_idx].to(previous_activations.device)

                # Compute activation deltas for all neurons in the group
                activation_deltas = group_mean - previous_activations  # Shape: [seq_len, num_neurons_in_group]

                logger.info("Finish computing activation delta")
                # Apply W_out for all neurons in the group
                # activation_deltas: [seq_len, num_neurons_in_group]
                # W_out: [layer, neurons, d_model]
                # We want to get the effect of all neurons in the group on the residual stream

                # Get the W_out for all neurons in this group
                w_out_group = self.model.W_out[layer_idx, neuron_indices, :]  # [num_neurons_in_group, d_model]

                # Compute residual stream deltas for all neurons together
                # [seq_len, num_neurons_in_group, 1] * [num_neurons_in_group, d_model]
                # -> [seq_len, num_neurons_in_group, d_model]
                res_deltas = activation_deltas.unsqueeze(-1) * w_out_group

                # Sum across neurons to get the combined effect
                # [seq_len, num_neurons_in_group, d_model] -> [seq_len, d_model]
                res_deltas_sum = res_deltas.sum(dim=1)

                # Add a batch dimension for processing
                res_deltas_sum = res_deltas_sum.unsqueeze(0)  # [1, seq_len, d_model]
                # Process in chunks - here we only have one batch since we're doing the combined effect
                updated_res_stream = res_stream + res_deltas_sum[0]  # [seq_len, d_model]
                updated_res_stream = updated_res_stream.unsqueeze(0)  # Add batch dim: [1, seq_len, d_model]

                # Apply layer normalization
                updated_res_stream = self.model.ln_final(updated_res_stream)

                logger.info("Finish computing res stream")

                # Project to logit space
                ablated_logits = updated_res_stream @ self.model.W_U + self.model.b_U

                # Project neurons for frequency adjustment if needed
                ablated_logits_with_frozen_unigram = self.project_neurons(
                    logits,
                    unigram_projection_values,
                    ablated_logits,
                    res_deltas_sum,
                )

                # Compute loss
                loss_post_ablation_batch = self.model.loss_fn(ablated_logits, inp, per_token=True).cpu()
                loss_post_ablation_batch = np.concatenate(
                    (loss_post_ablation_batch, np.zeros((loss_post_ablation_batch.shape[0], 1))), axis=1
                )
                loss_post_ablation.append(loss_post_ablation_batch)

                logger.info("Finish computing ablated loss")

                # Compute entropy
                entropy_post_ablation_batch = self.get_entropy(ablated_logits)
                entropy_post_ablation.append(entropy_post_ablation_batch.cpu())

                # Process frozen unigram results
                loss_post_ablation_with_frozen_unigram_batch = self.model.loss_fn(
                    ablated_logits_with_frozen_unigram, inp, per_token=True
                ).cpu()
                loss_post_ablation_with_frozen_unigram_batch = np.concatenate(
                    (
                        loss_post_ablation_with_frozen_unigram_batch,
                        np.zeros((loss_post_ablation_with_frozen_unigram_batch.shape[0], 1)),
                    ),
                    axis=1,
                )
                loss_post_ablation_with_frozen_unigram.append(loss_post_ablation_with_frozen_unigram_batch)

                logger.info("Finish computing ablated loss by projection")

                # Compute entropy for frozen unigram
                entropy_post_ablation_with_frozen_unigram_batch = self.get_entropy(ablated_logits_with_frozen_unigram)
                entropy_post_ablation_with_frozen_unigram.append(entropy_post_ablation_with_frozen_unigram_batch.cpu())

                # Calculate KL divergence
                abl_logprobs = ablated_logits.log_softmax(dim=-1)
                kl_divergence_after_batch, kl_divergence_after_frozen_unigram_batch = self.compute_kl(
                    ablated_logits_with_frozen_unigram,
                    abl_logprobs,
                    [],  # Empty list as we're not appending to existing lists
                    [],  # Empty list as we're not appending to existing lists
                )
                kl_divergence_after.append(
                    kl_divergence_after_batch[0]
                )  # First element since compute_kl returns a list
                kl_divergence_after_frozen_unigram.append(kl_divergence_after_frozen_unigram_batch[0])

                logger.info("Finish computing ablated loss")

                # Clean up
                del ablated_logits, ablated_logits_with_frozen_unigram, abl_logprobs

            # Process results and prepare dataframe
            batch_results = self.process_group_batch_results(
                batch_n,
                filtered_entropy_df,
                loss_post_ablation,
                loss_post_ablation_with_frozen_unigram,
                entropy_post_ablation,
                entropy_post_ablation_with_frozen_unigram,
                kl_divergence_before,
                kl_divergence_after,
                kl_divergence_after_frozen_unigram,
            )

            self.results[batch_n] = batch_results
            pbar.update(1)

            # Clean up to avoid memory issues
            del logits, cache, logprobs
            torch.cuda.empty_cache()
            """
        return self.results

    def process_group_batch_results(
        self,
        batch_n: int,
        filtered_entropy_df: pd.DataFrame,
        loss_post_ablation: list,
        loss_post_ablation_with_frozen_unigram: list,
        entropy_post_ablation: list,
        entropy_post_ablation_with_frozen_unigram: list,
        kl_divergence_before: np.ndarray,
        kl_divergence_after: list,
        kl_divergence_after_frozen_unigram: list,
    ) -> pd.DataFrame:
        """Process results for a batch and create a dataframe for group ablation."""
        final_df = None

        for i, group_name in enumerate(self.group_names):
            df_to_append = filtered_entropy_df[filtered_entropy_df.batch == batch_n].copy()

            # Add group information
            df_to_append["group_name"] = group_name
            df_to_append["neuron_group"] = str(self.neuron_groups[i])  # Store the neurons in this group
            df_to_append["layer"] = self.group_layers[i]

            # Add ablation metrics
            df_to_append["loss_post_ablation"] = loss_post_ablation[i]
            df_to_append["loss_post_ablation_with_frozen_unigram"] = loss_post_ablation_with_frozen_unigram[i]
            df_to_append["entropy_post_ablation"] = entropy_post_ablation[i]
            df_to_append["entropy_post_ablation_with_frozen_unigram"] = entropy_post_ablation_with_frozen_unigram[i]
            df_to_append["kl_divergence_before"] = kl_divergence_before
            df_to_append["kl_divergence_after"] = kl_divergence_after[i]
            df_to_append["kl_divergence_after_frozen_unigram"] = kl_divergence_after_frozen_unigram[i]

            # Add ablation information
            df_to_append["ablation_mode"] = self.ablation_mode
            if self.ablation_mode == "longtail":
                df_to_append["longtail_threshold"] = self.longtail_threshold
                df_to_append["num_longtail_tokens"] = self.longtail_mask.sum().item()

            final_df = df_to_append if final_df is None else pd.concat([final_df, df_to_append])

        return final_df


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
