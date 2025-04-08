# %%
import argparse
import os
import sys

sys.path.append("../")
import logging
import typing as t
from pathlib import Path
from warnings import simplefilter

import hydra
import neel.utils as nutils
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from transformer_lens import utils

from neuron_analyzer import settings
from neuron_analyzer.abl_util import (
    filter_entropy_activation_df,
    get_entropy_activation_df,
    get_pile_unigram_distribution,
    load_model_from_tl_name,
)
from neuron_analyzer.ablation import mean_ablate_components
from neuron_analyzer.analysis.freq import ZipfThresholdAnalyzer
from neuron_analyzer.model_util import StepConfig

T = t.TypeVar("T")

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for step range."""
    parser = argparse.ArgumentParser(description="Extract word surprisal across different training steps.")
    parser.add_argument("--start", type=int, default=0, help="Start index of step range")
    parser.add_argument("--end", type=int, default=155, help="End index of step range")
    parser.add_argument(
        "--config",
        type=str,
        default="config_unigram_ablations.yaml",
        help="Name of the configuration file to use (without .yaml extension)",
    )
    return parser.parse_args()


class NeuronAblationProcessor:
    """Class to handle neural network ablation processing."""

    def __init__(self, args: DictConfig, logger: logging.Logger | None = None):
        """Initialize the ablation processor with configuration."""
        # Set up logger
        self.logger = logger or logging.getLogger(__name__)

        # Initialize parameters from args
        self.args = args
        self.seed: int = args.seed
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize random seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.set_grad_enabled(False)

        # Change directory if specified
        if hasattr(args, "chdir") and args.chdir:
            os.chdir(args.chdir)

    def load_unigram(self) -> torch.Tensor:
        """Load unigram distribution based on model type."""
        # Load unigram distribution
        if "pythia" in self.args.model:
            self.logger.info("Loading unigram distribution for pythia...")
            unigram_distrib = get_pile_unigram_distribution(
                device=self.args.device, file_path=settings.PATH.unigram_dir / "pythia-unigrams.npy"
            )
        elif "gpt" in self.args.model:
            self.logger.info("Loading unigram distribution for gpt2...")
            unigram_distrib = get_pile_unigram_distribution(
                device=self.args.device,
                file_path=settings.PATH.unigram_dir / "gpt2-small-unigrams_openwebtext-2M_rows_500000.npy",
                pad_to_match_W_U=False,
            )
        else:
            raise Exception(f"No unigram distribution for {self.args.model}")

        return unigram_distrib

    def get_tail_threshold(self, unigram_distrib, save_path: Path) -> tuple[float | None, dict | None]:
        """Calculate threshold for long-tail ablation mode."""
        if self.args.ablation_mode == "longtail":
            if unigram_distrib is None:
                self.load_unigram()

            window_size = 2000
            analyzer = ZipfThresholdAnalyzer(unigram_distrib, window_size=window_size)
            threshold_stats = analyzer.analyze_zipf_anomalies(verbose=False)
            longtail_threshold = threshold_stats["elbow_info"]["elbow_probability"]
            self.logger.info(f"Calculating long-tail threshold using Zipf's law with window size {window_size}.")

            # Save threshold statistics only for the first step
            stats_df = pd.DataFrame([threshold_stats])
            stats_path = save_path / "zipf_threshold_stats.csv"
            stats_df.to_csv(stats_path, index=False)
            self.logger.info(f"Saved threshold statistics to {stats_path}")

            return longtail_threshold, threshold_stats
        # Not in longtail mode, use default threshold
        return None, None

    def process_single_step(
        self, step: int, unigram_distrib, longtail_threshold, threshold_stats, save_path: Path
    ) -> None:
        """Process a single step with the given configuration."""
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Load model and tokenizer for specific step
        model, tokenizer = self.load_model_and_tokenizer(step)

        self.logger.info("Finished loading model and tokenizer")
        # Load and process dataset
        data = load_dataset(self.args.dataset, split="train")
        first_1k = data.select([i for i in range(self.args.data_range_start, self.args.data_range_end)])
        tokenized_data = utils.tokenize_and_concatenate(first_1k, tokenizer, max_length=256, column_name="text")
        tokenized_data = tokenized_data.shuffle(self.args.seed)
        token_df = nutils.make_token_df(tokenized_data["tokens"], model=model)

        self.logger.info("Finished tokenizing data")

        # Setup neuron indices
        entropy_neuron_layer = model.cfg.n_layers - 1
        if self.args.neuron_range is not None:
            start, end = map(int, self.args.neuron_range.split("-"))
            all_neuron_indices = list(range(start, end))
        else:
            all_neuron_indices = list(range(model.cfg.d_mlp))

        all_neurons = [f"{entropy_neuron_layer}.{i}" for i in all_neuron_indices]
        self.logger.info("Loaded all the neurons")

        if self.args.dry_run:
            all_neurons = all_neurons[:10]

        # Compute entropy and activation for each neuron
        entropy_dim_layer = model.cfg.n_layers - 1
        entropy_df = get_entropy_activation_df(
            all_neurons,
            tokenized_data,
            token_df,
            model,
            batch_size=self.args.batch_size,
            device=self.device,
            cache_residuals=False,
            cache_pre_activations=False,
            compute_kl_from_bu=False,
            residuals_layer=entropy_dim_layer,
            residuals_dict={},
        )
        self.logger.info("finished computing all the entropy")

        # Ablate the dimensions
        model.set_use_attn_result(False)

        if self.args.ablation_mode == "longtail" and longtail_threshold is not None:
            self.logger.info(f"Long-tail threshold: {longtail_threshold}")
            # Calculate how many tokens are below this threshold
            long_tail_count = (unigram_distrib.cpu().numpy() < longtail_threshold).sum()
            vocab_size = len(unigram_distrib)
            self.logger.info(
                f"Long-tail tokens: {long_tail_count} ({long_tail_count / vocab_size * 100:.2f}% of vocabulary)"
            )

        results = mean_ablate_components(
            components_to_ablate=all_neurons,
            tokenized_data=tokenized_data,
            entropy_df=entropy_df,
            model=model,
            k=self.args.k,
            device=self.device,
            unigram_distrib=unigram_distrib,
            ablation_mode=self.args.ablation_mode,
            longtail_threshold=longtail_threshold,
        )
        self.logger.info("finished ablations!")

        # Process and save results
        self._save_results(results, tokenizer, step, save_path)

    def load_model_and_tokenizer(self, step: int) -> tuple[t.Any, t.Any]:
        """Load model and tokenizer for processing."""
        # Load HF token
        with open(settings.PATH.unigram_dir / self.args.hf_token_path) as f:
            hf_token = f.read()

        # Load model and tokenizer
        model, tokenizer = load_model_from_tl_name(
            self.args.model, self.device, step=step, cache_dir=settings.PATH.model_dir, hf_token=hf_token
        )
        model = model.to(self.device)
        model.eval()

        return model, tokenizer

    def _save_results(
        self,
        results: dict,
        tokenizer,
        step: int,
        save_path: Path,
    ) -> None:
        """Process and save ablation results."""
        final_df = pd.concat(results.values())
        final_df = filter_entropy_activation_df(
            final_df.reset_index(), model_name=self.args.model, tokenizer=tokenizer, start_pos=3, end_pos=-1
        )

        # Save results
        final_df = final_df.reset_index(drop=True)
        output_path = save_path / f"k{self.args.k}.feather"
        final_df.to_feather(output_path)
        self.logger.info(f"Saved results for step {step} to {output_path}")


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

        # intialize the process class
        base_save_dir = settings.PATH.result_dir / hydra_args.output_dir / hydra_args.ablation_mode / hydra_args.model
        base_save_dir.mkdir(parents=True, exist_ok=True)
        abalation_processor = NeuronAblationProcessor(args=hydra_args, logger=logger)
        unigram_distrib = abalation_processor.load_unigram()
        longtail_threshold, threshold_stats = abalation_processor.get_tail_threshold(
            unigram_distrib, save_path=base_save_dir
        )

        # Process each step in range
        for step in steps_config.steps[cli_args.start : cli_args.end]:
            # Create save_path as a directory
            save_path = base_save_dir / str(step) / str(hydra_args.data_range_end)
            save_path.mkdir(parents=True, exist_ok=True)
            # Check for existing files with pattern matching expected output
            if (save_path / f"k{hydra_args.k}.feather").is_file():
                logger.info(f"Files for step {step} already exist. Skip!")
                continue
            logger.info(f"Processing step {step}")
            try:
                abalation_processor.process_single_step(
                    step, unigram_distrib, longtail_threshold, threshold_stats, save_path
                )

            except Exception as e:
                logger.error(f"Error processing step {step}: {e!s}")
                continue


if __name__ == "__main__":
    logger.info(f"Current directory: {os.getcwd()}")
    main()
