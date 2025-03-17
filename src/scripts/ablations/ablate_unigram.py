# %%
import argparse
import os
import sys

sys.path.append("../")
import logging
from pathlib import Path
from warnings import simplefilter

import hydra
import neel.utils as nutils
import numpy as np
import pandas as pd
import torch
import tqdm
import transformer_lens.utils as utils
from datasets import load_dataset
from omegaconf import DictConfig
from scipy import stats
from torch.nn.functional import kl_div

from neuron_analyzer import settings
from neuron_analyzer.ablations import (
    filter_entropy_activation_df,
    get_entropy,
    get_entropy_activation_df,
    get_pile_unigram_distribution,
    load_model_from_tl_name,
)
from neuron_analyzer.surprisal import StepConfig

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for step range."""
    parser = argparse.ArgumentParser(description="Extract word surprisal across different training steps.")
    parser.add_argument("-s", "--start", type=int, default=0, help="Start index of step range")
    parser.add_argument("-e", "--end", type=int, default=155, help="End index of step range")
    parser.add_argument(
        "-c", "--config", 
        type=str,
        default="config_unigram_ablations.yaml",
        help="Name of the configuration file to use (without .yaml extension)"
    )
    return parser.parse_args()


def adjust_vectors_3dim(v, u, target_values):
    """
    Adjusts a batch of vectors v such that their projections along the unit vector u equal the target values.

    Parameters:
    - v: A 3D tensor of shape (n, m, d), representing the batch of vectors to be adjusted.
    - u: A 1D unit tensor of shape (d,), representing the direction along which the adjustment is made.
    - target_values: A 2D tensor of shape (n, m), representing the desired projection values of the vectors in v along u.

    Returns:
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
            df_to_append[f"loss_post_ablation"] = loss_post_ablation[i]
            df_to_append[f"loss_post_ablation_with_frozen_unigram"] = loss_post_ablation_with_frozen_unigram[i]
            df_to_append[f"entropy_post_ablation"] = entropy_post_ablation[i]
            df_to_append[f"entropy_post_ablation_with_frozen_unigram"] = entropy_post_ablation_with_frozen_unigram[i]
            df_to_append[f"kl_divergence_before"] = kl_divergence_before
            df_to_append[f"kl_divergence_after"] = kl_divergence_after[i]
            df_to_append[f"kl_divergence_after_frozen_unigram"] = kl_divergence_after_frozen_unigram[i]
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


def calculate_zipf_threshold(unigram_distrib, head_portion=0.3, error_threshold=0.5):
    """Calculate long-tail threshold based on deviation from Zipf's law."""
    
    
    # Convert to numpy if tensor
    if torch.is_tensor(unigram_distrib):
        unigram_probs = unigram_distrib.cpu().numpy()
    else:
        unigram_probs = unigram_distrib
    
    # Make sure it's normalized
    if abs(unigram_probs.sum() - 1.0) > 1e-5:
        unigram_probs = unigram_probs / unigram_probs.sum()
    
    # Sort by frequency (descending)
    sorted_indices = np.argsort(-unigram_probs)
    sorted_probs = unigram_probs[sorted_indices]
    ranks = np.arange(1, len(sorted_probs) + 1)
    
    # Filter out zeros for log calculation
    nonzero_mask = sorted_probs > 0
    nonzero_ranks = ranks[nonzero_mask]
    nonzero_probs = sorted_probs[nonzero_mask]
    
    log_ranks = np.log(nonzero_ranks)
    log_probs = np.log(nonzero_probs)
    
    # Fit Zipf's law to the head of the distribution
    head_cutoff = int(len(nonzero_ranks) * head_portion)
    fit = stats.linregress(log_ranks[:head_cutoff], log_probs[:head_cutoff])
    alpha = -fit.slope
    
    # Predict token frequencies based on the fit
    predicted_log_probs = fit.slope * log_ranks + fit.intercept
    predicted_probs = np.exp(predicted_log_probs)
    
    # Calculate relative error between actual and predicted
    relative_error = np.abs(nonzero_probs - predicted_probs) / predicted_probs
    
    # Find where error exceeds the threshold
    deviation_indices = np.where(relative_error > error_threshold)[0]
    
    if len(deviation_indices) > 0:
        first_major_deviation = deviation_indices[0]
        threshold = nonzero_probs[first_major_deviation]
        deviation_rank = nonzero_ranks[first_major_deviation]
    else:
        # Fallback if no major deviation found
        threshold = np.percentile(nonzero_probs, 70)
        deviation_rank = None
    
    # Calculate percentage of tokens considered long-tail
    long_tail_count = (unigram_probs < threshold).sum()
    long_tail_percentage = long_tail_count / len(unigram_probs) * 100
    
    stats_dict = {
        'threshold': threshold,
        'alpha': alpha,
        'zipf_fit_slope': fit.slope,
        'zipf_fit_intercept': fit.intercept,
        'zipf_fit_r_value': fit.rvalue,
        'long_tail_token_count': long_tail_count,
        'long_tail_percentage': long_tail_percentage,
        'deviation_rank': deviation_rank
    }
    
    logger.info(f'The longtail threhsold is {long_tail_percentage}')
    return threshold, stats_dict


def process_single_step(args: DictConfig, step: int, save_path: Path) -> None:
    """Process a single step with the given configuration."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_grad_enabled(False)

    os.chdir(args.chdir)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(settings.PATH.dataset_root / "src" / "unigram" / args.hf_token_path, "r") as f:
        hf_token = f.read()

    # Load model and tokenizer for specific step
    model, tokenizer = load_model_from_tl_name(
        args.model,
        args.device,
        step=step,
        cache_dir=settings.PATH.model_dir,
        hf_token=hf_token
    )
    model = model.to(args.device)
    model.eval()

    # Load and process dataset
    data = load_dataset(args.dataset, split='train')
    first_1k = data.select([i for i in range(args.data_range_start, args.data_range_end)])
    tokenized_data = utils.tokenize_and_concatenate(
        first_1k,
        tokenizer,
        max_length=256,
        column_name='text'
    )
    tokenized_data = tokenized_data.shuffle(args.seed)
    token_df = nutils.make_token_df(tokenized_data['tokens'], model=model)

    logger.info("Finished tokenizing data")

    # Setup neuron indices
    entropy_neuron_layer = model.cfg.n_layers - 1
    if args.neuron_range is not None:
        start, end = map(int, args.neuron_range.split('-'))
        all_neuron_indices = list(range(start, end))
    else:
        all_neuron_indices = list(range(0, model.cfg.d_mlp))

    all_neurons = [f"{entropy_neuron_layer}.{i}" for i in all_neuron_indices]

    logger.info("Loaded all the neurons")

    if args.dry_run:
        all_neurons = all_neurons[:10]

    # Load unigram distribution
    if 'pythia' in args.model:
        logger.info('Loading unigram distribution for pythia...')
        unigram_distrib = get_pile_unigram_distribution(
            device=args.device,
            file_path=settings.PATH.dataset_root / "src/unigram/pythia-unigrams.npy"
        )
    elif 'gpt' in args.model:
        logger.info('Loading unigram distribution for gpt2...')
        unigram_distrib = get_pile_unigram_distribution(
            device=args.device,
            file_path=settings.PATH.dataset_root / "src/unigram/gpt2-small-unigrams_openwebtext-2M_rows_500000.npy",
            pad_to_match_W_U=False
        )
    else:
        raise Exception(f'No unigram distribution for {args.model}')

    # Compute entropy and activation for each neuron
    entropy_dim_layer = model.cfg.n_layers - 1
    entropy_df = get_entropy_activation_df(
        all_neurons,
        tokenized_data,
        token_df,
        model,
        batch_size=args.batch_size,
        device=args.device,
        cache_residuals=False,
        cache_pre_activations=False,
        compute_kl_from_bu=False,
        residuals_layer=entropy_dim_layer,
        residuals_dict={},
    )
    logger.info("finished computing all the entropy")

    # Ablate the dimensions
    model.set_use_attn_result(False)
    
    # Determine ablation mode from args
    ablation_mode = getattr(args, 'ablation_mode', 'mean')  # Default to 'mean' if not specified
    
    # For longtail mode, calculate threshold based on Zipf's law if auto_threshold is enabled
    if ablation_mode == "longtail":
        auto_threshold = getattr(args, 'auto_threshold', False)
        
        if auto_threshold:
            # Calculate threshold based on Zipf's law
            head_portion = getattr(args, 'zipf_head_portion', 0.3)
            error_threshold = getattr(args, 'zipf_error_threshold', 0.5)
            generate_plot = getattr(args, 'zipf_plot', True)
            
            logger.info(f"Calculating long-tail threshold using Zipf's law (head_portion={head_portion}, error_threshold={error_threshold})")
            longtail_threshold, threshold_stats = calculate_zipf_threshold(
                unigram_distrib,
                head_portion=head_portion,
                error_threshold=error_threshold
            )
            
            logger.info(f"Zipf analysis results: alpha={threshold_stats['alpha']:.3f}, threshold={longtail_threshold:.8f}")
            logger.info(f"Long-tail tokens: {threshold_stats['long_tail_token_count']} ({threshold_stats['long_tail_percentage']:.2f}% of vocabulary)")
            
            # Save threshold statistics
            stats_df = pd.DataFrame([threshold_stats])
            stats_path = save_path.parent / f"zipf_threshold_stats_step{step}.csv"
            stats_df.to_csv(stats_path, index=False)
            logger.info(f"Saved threshold statistics to {stats_path}")
        else:
            # Use the provided threshold
            longtail_threshold = getattr(args, 'longtail_threshold', 0.001)
            logger.info(f"Using provided long-tail threshold: {longtail_threshold}")
    else:
        # Not in longtail mode, use default threshold
        longtail_threshold = getattr(args, 'longtail_threshold', 0.001)
    
    logger.info(f"Using ablation mode: {ablation_mode}")
    if ablation_mode == "longtail":
        logger.info(f"Long-tail threshold: {longtail_threshold}")
        # Calculate how many tokens are below this threshold
        long_tail_count = (unigram_distrib.cpu().numpy() < longtail_threshold).sum()
        vocab_size = len(unigram_distrib)
        logger.info(f"Long-tail tokens: {long_tail_count} ({long_tail_count/vocab_size*100:.2f}% of vocabulary)")
    
    results = mean_ablate_components(
        components_to_ablate=all_neurons,
        tokenized_data=tokenized_data,
        entropy_df=entropy_df,
        model=model,
        k=args.k,
        device=args.device,
        unigram_distrib=unigram_distrib,
        ablation_mode=ablation_mode,
        longtail_threshold=longtail_threshold
    )
    logger.info("finished ablations!")

    # Process and save results
    final_df = pd.concat(results.values())
    final_df = filter_entropy_activation_df(
        final_df.reset_index(),
        model_name=args.model,
        tokenizer=tokenizer,
        start_pos=3,
        end_pos=-1
    )

    # Add threshold information to the dataframe
    if ablation_mode == "longtail":
        final_df['longtail_threshold'] = longtail_threshold
        if 'auto_threshold' in locals() and auto_threshold:
            for key, value in threshold_stats.items():
                if key != 'threshold':  # Already included as longtail_threshold
                    final_df[f'zipf_{key}'] = value

    # Save results
    final_df = final_df.reset_index(drop=True)
    
    # Include threshold in filename if automatically determined
    if ablation_mode == "longtail" and 'auto_threshold' in locals() and auto_threshold:
        threshold_str = f"_thresh{longtail_threshold:.8f}"
    else:
        threshold_str = ""
    
    output_path = save_path / f"k{args.k}_{ablation_mode}{threshold_str}.feather"
    final_df.to_feather(output_path)
    logger.info(f"Saved results for step {step} to {output_path}")


def main():
    """Main entry point that handles both CLI args and Hydra config."""
    # Parse command line arguments
    cli_args = parse_args()
    
    # Initialize Hydra programmatically with the config name from CLI
    config_name = cli_args.config
    config_path = "conf"
    
    # Set up Hydra's context
    with hydra.initialize(version_base=None, config_path=config_path):
        # Load the config with the name from CLI
        hydra_args = hydra.compose(config_name=config_name)
        logger.info(f"Using configuration: {config_name}")
        
        
        
        # Initialize configuration with all Pythia checkpoints
        steps_config = StepConfig()
        logger.info(f"Processing steps {cli_args.start} to {cli_args.end}")
        
        # Process each step in range
        for step in steps_config.steps[cli_args.start : cli_args.end]:
            # Calculate the base directory for saving results
            base_save_dir = (
                settings.PATH.result_dir
                / hydra_args.output_dir
                / hydra_args.ablation_mode
                / hydra_args.model
                / str(step)
                / str(hydra_args.data_range_end)
            )
            
            # Create save_path as a directory
            save_path = base_save_dir
            
            # Check for existing files with pattern matching expected output
            file_exists = False
            if base_save_dir.exists():
                for file in base_save_dir.glob(f"k{hydra_args.k}*.feather"):
                    file_exists = True
                    break
            
            if file_exists:
                logger.info(f"Files for step {step} already exist. Skip!")
                continue
            else:
                logger.info(f"Processing step {step}")
                try:
                    process_single_step(hydra_args, step, save_path)
                except Exception as e:
                    logger.error(f"Error processing step {step}: {str(e)}")
                    continue

if __name__ == '__main__':
    logger.info(f'Current directory: {os.getcwd()}')
    main()