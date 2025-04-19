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
from neuron_analyzer.ablation.abl_util import (
    filter_entropy_activation_df,
    load_model_from_tl_name,
)
from neuron_analyzer.analysis.freq import ZipfThresholdAnalyzer
from neuron_analyzer.load_util import JsonProcessor, load_unigram
from neuron_analyzer.model_util import StepConfig
from neuron_analyzer.selection.group import GroupModelAblationAnalyzer

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
        default="config_unigram_ablations_70.yaml",
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

    def get_tail_threshold_stat(self, unigram_distrib, save_path: Path) -> tuple[float | None, dict | None]:
        """Calculate threshold for long-tail ablation mode."""
        if self.args.ablation_mode == "longtail":
            analyzer = ZipfThresholdAnalyzer(
                unigram_distrib=unigram_distrib,
                window_size=self.args.window_size,
                tail_threshold=self.args.tail_threshold,
                apply_elbow=self.args.apply_elbow,
            )
            longtail_threshold, threshold_stats = analyzer.get_tail_threshold()
            JsonProcessor.save_json(threshold_stats, save_path / "zipf_threshold_stats.json")
            self.logger.info(f"Saved threshold statistics to {save_path}/zipf_threshold_stats.json")
            return longtail_threshold
        # Not in longtail mode, use default threshold
        return None

    def process_single_step(self, step: int, unigram_distrib, longtail_threshold, save_path: Path) -> None:
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
        entropy_df = pd.read_feather(save_path / f"k{self.args.k}.feather")
        self.logger.info("Finished computing all the entropy")

        # Ablate the dimensions
        model.set_use_attn_result(False)
        neuron_groups = [["5.110", "5.96", "5.838", "5.1622", "5.587"]]
        analyzer = GroupModelAblationAnalyzer(
            neuron_groups=neuron_groups,
            model=model,
            device=self.device,
            unigram_distrib=unigram_distrib,
            tokenized_data=tokenized_data,
            entropy_df=entropy_df,
            k=self.args.k,
            ablation_mode=self.args.ablation_mode,
            longtail_threshold=longtail_threshold,
        )

        results = analyzer.mean_ablate_components()

        self.logger.info("Finished ablations!")

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
        output_path = save_path / f"k{self.args.k}_group.feather"
        final_df.to_feather(output_path)
        self.logger.info(f"Saved results for step {step} to {output_path}")

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
        unigram_distrib, _ = load_unigram(model_name=hydra_args.model, device=device)
        longtail_threshold = abalation_processor.get_tail_threshold_stat(unigram_distrib, save_path=base_save_dir)

        # Process each step in range
        for step in steps_config.steps:
            # Create save_path as a directory
            save_path = base_save_dir / str(step) / str(hydra_args.data_range_end)
            save_path.mkdir(parents=True, exist_ok=True)
            # Check for existing files with pattern matching expected output
            if cli_args.resume and (save_path / f"k{hydra_args.k}_group.feather").is_file():
                logger.info(f"Files for step {step} already exist. Skip!")
                continue
            logger.info(f"Processing step {step}")

            abalation_processor.process_single_step(step, unigram_distrib, longtail_threshold, save_path)


if __name__ == "__main__":
    logger.info(f"Current directory: {os.getcwd()}")
    main()
