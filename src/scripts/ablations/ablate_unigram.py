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
import transformer_lens.utils as utils
from datasets import load_dataset
from omegaconf import DictConfig

from neuron_analyzer import settings
from neuron_analyzer.abl_util import (
    filter_entropy_activation_df,
    get_entropy_activation_df,
    get_pile_unigram_distribution,
    load_model_from_tl_name,
)
from neuron_analyzer.ablation import mean_ablate_components
from neuron_analyzer.freq import ZipfThresholdAnalyzer
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
        "-c",
        "--config",
        type=str,
        default="config_unigram_ablations.yaml",
        help="Name of the configuration file to use (without .yaml extension)",
    )
    return parser.parse_args()



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
        args.model, args.device, step=step, cache_dir=settings.PATH.model_dir, hf_token=hf_token
    )
    model = model.to(args.device)
    model.eval()

    # Load and process dataset
    data = load_dataset(args.dataset, split="train")
    first_1k = data.select([i for i in range(args.data_range_start, args.data_range_end)])
    tokenized_data = utils.tokenize_and_concatenate(first_1k, tokenizer, max_length=256, column_name="text")
    tokenized_data = tokenized_data.shuffle(args.seed)
    token_df = nutils.make_token_df(tokenized_data["tokens"], model=model)

    logger.info("Finished tokenizing data")

    # Setup neuron indices
    entropy_neuron_layer = model.cfg.n_layers - 1
    if args.neuron_range is not None:
        start, end = map(int, args.neuron_range.split("-"))
        all_neuron_indices = list(range(start, end))
    else:
        all_neuron_indices = list(range(0, model.cfg.d_mlp))

    all_neurons = [f"{entropy_neuron_layer}.{i}" for i in all_neuron_indices]

    logger.info("Loaded all the neurons")

    if args.dry_run:
        all_neurons = all_neurons[:10]

    # Load unigram distribution
    if "pythia" in args.model:
        logger.info("Loading unigram distribution for pythia...")
        unigram_distrib = get_pile_unigram_distribution(
            device=args.device, file_path=settings.PATH.dataset_root / "src/unigram/pythia-unigrams.npy"
        )
    elif "gpt" in args.model:
        logger.info("Loading unigram distribution for gpt2...")
        unigram_distrib = get_pile_unigram_distribution(
            device=args.device,
            file_path=settings.PATH.dataset_root / "src/unigram/gpt2-small-unigrams_openwebtext-2M_rows_500000.npy",
            pad_to_match_W_U=False,
        )
    else:
        raise Exception(f"No unigram distribution for {args.model}")

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
    ablation_mode = getattr(args, "ablation_mode", "mean")  # Default to 'mean' if not specified

    # For longtail mode, calculate threshold based on Zipf's law if auto_threshold is enabled
    if ablation_mode == "longtail":
        window_size = 2000
        analyzer = ZipfThresholdAnalyzer(unigram_distrib, window_size = window_size)
        threshold_stats = analyzer.analyze_zipf_anomalies(verbose=False)
        longtail_threshold = threshold_stats["elbow_info"]["elbow_probability"]
        logger.info(f"Calculating long-tail threshold using Zipf's law with window size {window_size}.")

        if step == 0:
            # Save threshold statistics only for the first step
            stats_df = pd.DataFrame([threshold_stats])
            stats_path = save_path.parent / "zipf_threshold_stats.csv"
            stats_df.to_csv(stats_path, index=False)
            logger.info(f"Saved threshold statistics to {stats_path}")

    else:
        # Not in longtail mode, use default threshold
        longtail_threshold = None

    logger.info(f"Using ablation mode: {ablation_mode}")
    if ablation_mode == "longtail":
        logger.info(f"Long-tail threshold: {longtail_threshold}")
        # Calculate how many tokens are below this threshold
        long_tail_count = (unigram_distrib.cpu().numpy() < longtail_threshold).sum()
        vocab_size = len(unigram_distrib)
        logger.info(f"Long-tail tokens: {long_tail_count} ({long_tail_count / vocab_size * 100:.2f}% of vocabulary)")

    results = mean_ablate_components(
        components_to_ablate=all_neurons,
        tokenized_data=tokenized_data,
        entropy_df=entropy_df,
        model=model,
        k=args.k,
        device=args.device,
        unigram_distrib=unigram_distrib,
        ablation_mode=ablation_mode,
        longtail_threshold=longtail_threshold,
    )
    logger.info("finished ablations!")

    # Process and save results
    final_df = pd.concat(results.values())
    final_df = filter_entropy_activation_df(
        final_df.reset_index(), model_name=args.model, tokenizer=tokenizer, start_pos=3, end_pos=-1
    )

    # Add threshold information to the dataframe
    if ablation_mode == "longtail":
        final_df["longtail_threshold"] = longtail_threshold

        for key, value in threshold_stats.items():
            if key != "threshold":  # Already included as longtail_threshold
                final_df[f"zipf_{key}"] = value

    # Save results
    final_df = final_df.reset_index(drop=True)

    output_path = save_path / f"k{args.k}_{ablation_mode}.feather"
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


if __name__ == "__main__":
    logger.info(f"Current directory: {os.getcwd()}")
    main()
