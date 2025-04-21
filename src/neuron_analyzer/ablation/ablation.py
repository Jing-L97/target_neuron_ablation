import sys

sys.path.append("../")
import logging
import typing as t
from warnings import simplefilter

import numpy as np
import pandas as pd
import torch
import tqdm
from transformer_lens import utils

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# Type variable for generic typing
T = t.TypeVar("T")


class ModelAblationAnalyzer:
    """Class for analyzing neural network components through ablations."""

    def __init__(
        self,
        model,
        unigram_distrib,
        tokenized_data,
        entropy_df,
        components_to_ablate,
        device: str,
        k: int = 10,
        chunk_size: int = 20,
        ablation_mode: str = "mean",  # "mean" or "longtail"
        longtail_threshold: float = 0.001,  # Threshold for long-tail tokens
    ):
        """Initialize the ModelAblationAnalyzer."""
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
        self.components_to_ablate = components_to_ablate
        self.longtail_mask = None
        self.unigram_direction_vocab = None

    def build_vector(self) -> None:
        """Build frequency-related vectors for ablation analysis."""
        if "longtail" in self.ablation_mode:
            # Create long-tail token mask (1 for long-tail tokens, 0 for common tokens)
            self.longtail_mask = (self.unigram_distrib < self.longtail_threshold).float()
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

    def adjust_vectors_3dim(self, v: torch.Tensor, u: torch.Tensor, target_values: torch.Tensor) -> torch.Tensor:
        """Adjusts a batch of vectors v such that their projections along the unit vector u equal the target values.

        v: A 3D tensor of shape (n, m, d), representing the batch of vectors to be adjusted
        u: A 1D unit tensor of shape (d,), representing the direction along which the adjustment is made
        target_values: A 2D tensor of shape (n, m), representing the desired projection values of the vectors in v along u
        """
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
        if "longtail" in self.ablation_mode:
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

        if "longtail" in self.ablation_mode:
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

    def process_batch_results(
        self,
        batch_n: int,
        filtered_entropy_df: pd.DataFrame,
        loss_post_ablation: np.ndarray,
        loss_post_ablation_with_frozen_unigram: np.ndarray,
        entropy_post_ablation: np.ndarray,
        entropy_post_ablation_with_frozen_unigram: np.ndarray,
        kl_divergence_before: np.ndarray,
        kl_divergence_after: np.ndarray,
        kl_divergence_after_frozen_unigram: np.ndarray,
    ) -> pd.DataFrame:
        """Process results for a batch and create a dataframe."""
        final_df = None

        for i, component_name in enumerate(self.components_to_ablate):
            df_to_append = filtered_entropy_df[filtered_entropy_df.batch == batch_n].copy()

            # drop all the columns that are not the component_name
            df_to_append = df_to_append.drop(
                columns=[f"{neuron}_activation" for neuron in self.components_to_ablate if neuron != component_name]
            )

            # rename the component_name column to 'activation'
            df_to_append = df_to_append.rename(columns={f"{component_name}_activation": "activation"})

            df_to_append["component_name"] = component_name
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

    def mean_ablate_components(self) -> dict[int, pd.DataFrame]:
        """Perform mean ablation on specified components."""
        # Sample a set of random batch indices
        random_sequence_indices = np.random.choice(self.entropy_df.batch.unique(), self.k, replace=False)

        logger.info(f"ablate_components: ablate with k = {self.k}, long-tail threshold = {self.longtail_threshold}")

        pbar = tqdm.tqdm(total=self.k, file=sys.stdout)

        # new_entropy_df with only the random sequences
        filtered_entropy_df = self.entropy_df[self.entropy_df.batch.isin(random_sequence_indices)].copy()

        activation_mean_values = torch.tensor(
            self.entropy_df[[f"{component_name}_activation" for component_name in self.components_to_ablate]].mean()
        )

        # Build frequency vectors
        self.build_vector()

        # get neuron indices
        self.neuron_indices = [int(neuron_name.split(".")[1]) for neuron_name in self.components_to_ablate]

        # get layer indices
        layer_indices = [int(neuron_name.split(".")[0]) for neuron_name in self.components_to_ablate]
        self.layer_idx = layer_indices[0]

        for batch_n in filtered_entropy_df.batch.unique():
            tok_seq = self.tokenized_data["tokens"][batch_n]

            # get unaltered logits
            self.model.reset_hooks()
            inp = tok_seq.unsqueeze(0).to(self.device)
            logits, cache = self.model.run_with_cache(inp)
            logprobs = logits[0, :, :].log_softmax(dim=-1)

            res_stream = cache[utils.get_act_name("resid_post", self.layer_idx)][0]

            # get the entropy_df entries for the current sequence
            rows = filtered_entropy_df[filtered_entropy_df.batch == batch_n]
            assert len(rows) == len(tok_seq), f"len(rows) = {len(rows)}, len(tok_seq) = {len(tok_seq)}"

            # get the value of the logits projected onto the b_U direction
            unigram_projection_values = self.project_logits(logits)
            previous_activation = cache[utils.get_act_name("post", self.layer_idx)][0, :, self.neuron_indices]
            del cache
            activation_deltas = activation_mean_values.to(previous_activation.device) - previous_activation
            # activation deltas is seq_n x n_neurons

            # multiple deltas by W_out
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
                # apply ln_final
                updated_res_stream_chunk = self.model.ln_final(updated_res_stream_chunk)

                # Project to logit space
                ablated_logits_chunk = updated_res_stream_chunk @ self.model.W_U + self.model.b_U

                ablated_logits_with_frozen_unigram_chunk = self.project_neurons(
                    logits,
                    unigram_projection_values,
                    ablated_logits_chunk,
                    res_deltas_chunk,
                )

                # compute loss for the chunk
                loss_post_ablation_chunk = self.model.loss_fn(
                    ablated_logits_chunk, inp.repeat(res_deltas_chunk.shape[0], 1), per_token=True
                ).cpu()
                loss_post_ablation_chunk = np.concatenate(
                    (loss_post_ablation_chunk, np.zeros((loss_post_ablation_chunk.shape[0], 1))), axis=1
                )
                loss_post_ablation.append(loss_post_ablation_chunk)

                # compute entropy for the chunk
                entropy_post_ablation_chunk = self.get_entropy(ablated_logits_chunk)
                entropy_post_ablation.append(entropy_post_ablation_chunk.cpu())

                abl_logprobs = ablated_logits_chunk.log_softmax(dim=-1)

                del ablated_logits_chunk

                # compute loss for ablated_logits_with_frozen_unigram_chunk
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

                # compute entropy for ablated_logits_with_frozen_unigram_chunk
                entropy_post_ablation_with_frozen_unigram_chunk = self.get_entropy(
                    ablated_logits_with_frozen_unigram_chunk
                )
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
            entropy_post_ablation_with_frozen_unigram = np.concatenate(
                entropy_post_ablation_with_frozen_unigram, axis=0
            )

            kl_divergence_after = np.concatenate(kl_divergence_after, axis=0)
            kl_divergence_after_frozen_unigram = np.concatenate(kl_divergence_after_frozen_unigram, axis=0)

            del res_deltas
            torch.cuda.empty_cache()  # Empty the cache

            # Process results and prepare dataframe
            batch_results = self.process_batch_results(
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

        return self.results
