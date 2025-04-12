import sys

sys.path.append("../")
import logging
from warnings import simplefilter

import numpy as np
import pandas as pd
import torch
import tqdm
from torch.nn.functional import kl_div
from transformer_lens import utils

from neuron_analyzer.ablation.abl_util import get_entropy

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def adjust_vectors_3dim(v, u, target_values):
    """Adjusts a batch of vectors v such that their projections along the unit vector u equal the target values.

    Parameters
    ----------
    - v: A 3D tensor of shape (n, m, d), representing the batch of vectors to be adjusted.
    - u: A 1D unit tensor of shape (d,), representing the direction along which the adjustment is made.
    - target_values: A 2D tensor of shape (n, m), representing the desired projection values of the vectors in v along u.

    Returns
    -------
    - adjusted_v: The adjusted batch of vectors such that their projections along u are equal to the target values.

    """
    current_projections = (v @ u.unsqueeze(-1)).squeeze(-1)  # Current projections of v onto u
    delta = target_values - current_projections  # Differences needed to reach the target projections
    adjusted_v = v + delta.unsqueeze(-1) * u  # Adjust v by the deltas along the direction of u
    return adjusted_v


def mean_ablate_components(
    components_to_ablate=None,
    unigram_distrib=None,
    tokenized_data=None,
    entropy_df=None,
    model=None,
    k=10,
    device="mps",
    chunk_size=20,
    ablation_mode="mean",  # "mean" or "longtail"
    longtail_threshold=0.001,
):  # Threshold for long-tail tokens
    # sample a set of random batch indices
    random_sequence_indices = np.random.choice(entropy_df.batch.unique(), k, replace=False)

    logger.info(f"ablate_components: ablate with k = {k}, long-tail threshold = {longtail_threshold}")

    pbar = tqdm.tqdm(total=k, file=sys.stdout)

    # new_entropy_df with only the random sequences
    filtered_entropy_df = entropy_df[entropy_df.batch.isin(random_sequence_indices)].copy()

    results = {}
    final_df = None

    activation_mean_values = torch.tensor(
        entropy_df[[f"{component_name}_activation" for component_name in components_to_ablate]].mean()
    )

    # This section is now handled in the ablation_mode conditional above

    if ablation_mode == "longtail":
        # Create long-tail token mask (1 for long-tail tokens, 0 for common tokens)
        longtail_mask = (unigram_distrib < longtail_threshold).float()
        logger.info(f"Number of long-tail tokens: {longtail_mask.sum().item()} out of {len(longtail_mask)}")

        # Create token frequency vector focusing on long-tail tokens only
        # Original token frequency vector from the unigram distribution
        full_unigram_direction_vocab = unigram_distrib.log() - unigram_distrib.log().mean()
        full_unigram_direction_vocab /= full_unigram_direction_vocab.norm()

        # Modified token frequency vector that zeros out common tokens
        # This makes the vector only consider contributions from long-tail tokens
        longtail_unigram_direction_vocab = full_unigram_direction_vocab * longtail_mask
        # Re-normalize to keep it a unit vector
        if longtail_unigram_direction_vocab.norm() > 0:
            longtail_unigram_direction_vocab /= longtail_unigram_direction_vocab.norm()
    else:
        # Standard frequency direction for regular mean ablation
        unigram_direction_vocab = unigram_distrib.log() - unigram_distrib.log().mean()
        unigram_direction_vocab /= unigram_direction_vocab.norm()

    # get neuron indices
    neuron_indices = [int(neuron_name.split(".")[1]) for neuron_name in components_to_ablate]

    # get layer indices
    layer_indices = [int(neuron_name.split(".")[0]) for neuron_name in components_to_ablate]
    layer_idx = layer_indices[0]

    for batch_n in filtered_entropy_df.batch.unique():
        tok_seq = tokenized_data["tokens"][batch_n]

        # get unaltered logits
        model.reset_hooks()
        inp = tok_seq.unsqueeze(0).to(device)
        logits, cache = model.run_with_cache(inp)
        logprobs = logits[0, :, :].log_softmax(dim=-1)

        res_stream = cache[utils.get_act_name("resid_post", layer_idx)][0]

        # get the entropy_df entries for the current sequence
        rows = filtered_entropy_df[filtered_entropy_df.batch == batch_n]
        assert len(rows) == len(tok_seq), f"len(rows) = {len(rows)}, len(tok_seq) = {len(tok_seq)}"

        # get the value of the logits projected onto the b_U direction
        if ablation_mode == "longtail":
            # For long-tail mode, project onto our modified frequency direction
            unigram_projection_values = logits @ longtail_unigram_direction_vocab
        else:
            # Regular projection for standard mean ablation
            unigram_projection_values = logits @ unigram_direction_vocab
        unigram_projection_values = unigram_projection_values.squeeze()
        previous_activation = cache[utils.get_act_name("post", layer_idx)][0, :, neuron_indices]
        del cache
        activation_deltas = activation_mean_values.to(previous_activation.device) - previous_activation
        # activation deltas is seq_n x n_neurons

        # multiple deltas by W_out
        res_deltas = activation_deltas.unsqueeze(-1) * model.W_out[layer_idx, neuron_indices, :]
        res_deltas = res_deltas.permute(1, 0, 2)

        loss_post_ablation = []
        entropy_post_ablation = []

        loss_post_ablation_with_frozen_unigram = []
        entropy_post_ablation_with_frozen_unigram = []

        kl_divergence_after = []
        kl_divergence_after_frozen_unigram = []

        log_unigram_distrib = unigram_distrib.log()

        kl_divergence_before = (
            kl_div(logprobs, log_unigram_distrib, reduction="none", log_target=True).sum(axis=-1).cpu().numpy()
        )

        for i in range(0, res_deltas.shape[0], chunk_size):
            res_deltas_chunk = res_deltas[i : i + chunk_size]
            updated_res_stream_chunk = res_stream.repeat(res_deltas_chunk.shape[0], 1, 1) + res_deltas_chunk
            # apply ln_final
            updated_res_stream_chunk = model.ln_final(updated_res_stream_chunk)

            # Project to logit space
            ablated_logits_chunk = updated_res_stream_chunk @ model.W_U + model.b_U

            # If we're in long-tail mode, apply the mask
            if ablation_mode == "longtail":
                # Get the original logits to preserve for common tokens
                original_logits = logits.repeat(res_deltas_chunk.shape[0], 1, 1)

                # Create a binary mask for the vocabulary dimension
                # 1 for long-tail tokens (to be modified), 0 for common tokens (to keep original)
                vocab_mask = longtail_mask.unsqueeze(0).unsqueeze(0)  # Shape: 1 x 1 x vocab_size

                # Apply the mask: use original_logits where mask is 0, use ablated_logits where mask is 1
                ablated_logits_chunk = (1 - vocab_mask) * original_logits + vocab_mask * ablated_logits_chunk

            # Adjust vectors to maintain unigram projections
            if ablation_mode == "longtail":
                # Use the long-tail-specific unigram direction
                ablated_logits_with_frozen_unigram_chunk = adjust_vectors_3dim(
                    ablated_logits_chunk, longtail_unigram_direction_vocab, unigram_projection_values
                )
            else:
                # Use the standard unigram direction
                ablated_logits_with_frozen_unigram_chunk = adjust_vectors_3dim(
                    ablated_logits_chunk, unigram_direction_vocab, unigram_projection_values
                )

            # compute loss for the chunk
            loss_post_ablation_chunk = model.loss_fn(
                ablated_logits_chunk, inp.repeat(res_deltas_chunk.shape[0], 1), per_token=True
            ).cpu()
            loss_post_ablation_chunk = np.concatenate(
                (loss_post_ablation_chunk, np.zeros((loss_post_ablation_chunk.shape[0], 1))), axis=1
            )
            loss_post_ablation.append(loss_post_ablation_chunk)

            # compute entropy for the chunk
            entropy_post_ablation_chunk = get_entropy(ablated_logits_chunk)
            entropy_post_ablation.append(entropy_post_ablation_chunk.cpu())

            abl_logprobs = ablated_logits_chunk.log_softmax(dim=-1)

            del ablated_logits_chunk

            # compute loss for ablated_logits_with_frozen_unigram_chunk
            loss_post_ablation_with_frozen_unigram_chunk = model.loss_fn(
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
            entropy_post_ablation_with_frozen_unigram_chunk = get_entropy(ablated_logits_with_frozen_unigram_chunk)
            entropy_post_ablation_with_frozen_unigram.append(entropy_post_ablation_with_frozen_unigram_chunk.cpu())

            # compute KL divergence between the distribution ablated with frozen unigram and the og distribution
            abl_logprobs_with_frozen_unigram = ablated_logits_with_frozen_unigram_chunk.log_softmax(dim=-1)

            # compute KL divergence between the ablated distribution and the distribution from the unigram direction
            kl_divergence_after_chunk = (
                kl_div(abl_logprobs, log_unigram_distrib.expand_as(abl_logprobs), reduction="none", log_target=True)
                .sum(axis=-1)
                .cpu()
                .numpy()
            )

            del abl_logprobs
            kl_divergence_after.append(kl_divergence_after_chunk)

            if ablation_mode == "longtail":
                # For long-tail mode, compute KL divergence with focus on the long-tail tokens
                masked_logprobs = abl_logprobs_with_frozen_unigram.clone()
                masked_logprobs = masked_logprobs + (1 - longtail_mask).unsqueeze(0).unsqueeze(0) * -1e10
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
            del abl_logprobs_with_frozen_unigram
            kl_divergence_after_frozen_unigram.append(kl_divergence_after_frozen_unigram_chunk)

            del ablated_logits_with_frozen_unigram_chunk
        # Concatenate results
        loss_post_ablation = np.concatenate(loss_post_ablation, axis=0)
        entropy_post_ablation = np.concatenate(entropy_post_ablation, axis=0)

        loss_post_ablation_with_frozen_unigram = np.concatenate(loss_post_ablation_with_frozen_unigram, axis=0)
        entropy_post_ablation_with_frozen_unigram = np.concatenate(entropy_post_ablation_with_frozen_unigram, axis=0)

        kl_divergence_after = np.concatenate(kl_divergence_after, axis=0)
        kl_divergence_after_frozen_unigram = np.concatenate(kl_divergence_after_frozen_unigram, axis=0)

        del res_deltas
        torch.cuda.empty_cache()  # Empty the cache

        # Process results as before
        for i, component_name in enumerate(components_to_ablate):
            df_to_append = filtered_entropy_df[filtered_entropy_df.batch == batch_n].copy()

            # drop all the columns that are not the component_name
            df_to_append = df_to_append.drop(
                columns=[f"{neuron}_activation" for neuron in components_to_ablate if neuron != component_name]
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
            df_to_append["ablation_mode"] = ablation_mode
            if ablation_mode == "longtail":
                df_to_append["longtail_threshold"] = longtail_threshold
                df_to_append["num_longtail_tokens"] = longtail_mask.sum().item()

            final_df = df_to_append if final_df is None else pd.concat([final_df, df_to_append])

        results[batch_n] = final_df
        final_df = None

        pbar.update(1)
    return results
