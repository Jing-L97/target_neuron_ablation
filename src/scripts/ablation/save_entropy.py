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
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from neuron_analyzer import settings
from neuron_analyzer.ablation.abl_util import (
    get_entropy_activation_df,
)
from neuron_analyzer.model_util import ModelHandler, StepConfig

T = t.TypeVar("T")

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for step range."""
    parser = argparse.ArgumentParser(description="Extract word surprisal across different training steps.")
    parser.add_argument("--interval", type=int, default=10, help="Checkpoint interval sampling")
    parser.add_argument(
        "--config_name",
        type=str,
        default="config_unigram_ablations_410.yaml",
        help="Name of the configuration file to use",
    )
    parser.add_argument("--config_path", type=str, default="conf", help="Relative dir to config file")
    parser.add_argument("--start", type=int, default=14, help="Start index of step range")
    parser.add_argument("--end", type=int, default=142, help="End index of step range")
    parser.add_argument("--debug", action="store_true", help="Compute the first few 5 lines if enabled")
    parser.add_argument("--resume", action="store_true", help="Resume from the existing checkpoint")
    return parser.parse_args()


#######################################################################################################
# Fucntions applied in the main scripts
#######################################################################################################


class NeuronAblationProcessor:
    """Class to handle neural network ablation processing."""

    def __init__(self, args: DictConfig, device, logger: logging.Logger | None = None):
        """Initialize the ablation processor with configuration."""
        # Set up logger
        self.logger = logger or logging.getLogger(__name__)

        # Initialize parameters from args
        self.args = args
        self.seed: int = args.seed
        self.device: str = device

        # Initialize random seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.set_grad_enabled(False)

        # Change directory if specified
        if hasattr(args, "chdir") and args.chdir:
            os.chdir(args.chdir)

    def process_single_step(self, step: int, save_path: Path) -> None:
        """Process a single step with the given configuration."""
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Load model and tokenizer for specific step
        model_handler = ModelHandler()
        # Load model and tokenizer for specific step
        model, _ = model_handler.load_model_and_tokenizer(
            step=step,
            model_name=self.args.model,
            hf_token_path=settings.PATH.unigram_dir / "hf_token.txt",
            device=self.device,
        )

        logger.info("Finished loading model and tokenizer")

        # Load and process dataset
        tokenized_data, token_df = model_handler.tokenize_data(
            dataset=self.args.dataset,
            data_range_start=self.args.data_range_start,
            data_range_end=self.args.data_range_end,
            seed=self.args.seed,
            get_df=True,
        )

        logger.info("Finished tokenizing data")

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
        self.logger.info("Finished computing all the entropy")
        entropy_df.to_csv(save_path / "entropy_df.csv")

    def get_save_dir(self):
        """Get the savepath based on current configurations."""
        if self.args.ablation_mode == "longtail" and self.args.apply_elbow:
            ablation_name = "longtail_elbow"
        if self.args.ablation_mode == "longtail" and not self.args.apply_elbow:
            ablation_name = f"longtail_{self.args.tail_threshold}"
        else:
            ablation_name = self.args.ablation_mode
        base_save_dir = settings.PATH.result_dir / self.args.output_dir / ablation_name / self.args.model
        base_save_dir.mkdir(parents=True, exist_ok=True)
        return base_save_dir


#######################################################################################################
# Entry point of the script
#######################################################################################################


def main():
    """Main entry point that handles both CLI args and Hydra config."""
    # Parse command line arguments
    cli_args = parse_args()
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Set up Hydra's context
    with hydra.initialize(version_base=None, config_path=cli_args.config_path):
        # Load the config with the name from CLI
        hydra_args = hydra.compose(config_name=cli_args.config_name)
        logger.info(f"Using configuration: {cli_args.config_name}")

        # Initialize step configurations
        steps_config = StepConfig(
            debug=cli_args.debug,
            interval=cli_args.interval,
            start_idx=cli_args.start,
            end_idx=cli_args.end,
        )
        # intialize the process class
        abalation_processor = NeuronAblationProcessor(args=hydra_args, device=device, logger=logger)
        base_save_dir = abalation_processor.get_save_dir()
        for step in steps_config.steps:
            # Create save_path as a directory
            save_path = base_save_dir / str(step) / str(hydra_args.data_range_end)
            save_path.mkdir(parents=True, exist_ok=True)
            # Check for existing files with pattern matching expected output
            if cli_args.resume and (save_path / "entropy_df.csv").is_file():
                logger.info(save_path / "entropy_df.csv")
                logger.info(f"Files for step {step} already exist. Skip!")
                continue
            logger.info(f"Processing step {step}")
            try:
                abalation_processor.process_single_step(step, save_path)
            except Exception as e:
                logger.error(f"Error processing step {step}: {e!s}")
                continue


if __name__ == "__main__":
    logger.info(f"Current directory: {os.getcwd()}")
    main()
