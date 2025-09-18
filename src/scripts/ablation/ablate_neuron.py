# %%
import argparse
import os
import sys

sys.path.append("../")
import logging
import typing as t
from warnings import simplefilter

import hydra
import pandas as pd
import torch

from neuron_analyzer import settings
from neuron_analyzer.ablation.ablation import NeuronAblationProcessor
from neuron_analyzer.load_util import cleanup, load_unigram
from neuron_analyzer.model_util import StepConfig

T = t.TypeVar("T")

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for step range."""
    parser = argparse.ArgumentParser(description="Extract word surprisal across different training steps.")
    parser.add_argument("--interval", type=int, default=20, help="Checkpoint interval sampling")
    parser.add_argument("--layer_num", type=int, default=1, help="Last n MLP layer")
    parser.add_argument(
        "--config_name",
        type=str,
        default="config_unigram_ablations_70M.yaml",
        help="Name of the configuration file to use",
    )
    parser.add_argument("--config_path", type=str, default="conf", help="Relative dir to config file")
    parser.add_argument("--start", type=int, default=14, help="Start index of step range")
    parser.add_argument("--end", type=int, default=130, help="End index of step range")
    parser.add_argument("--last_N_step", type=int, default=0, help="whether to further select least n steps")
    parser.add_argument("--debug", action="store_true", help="Compute the first few 5 lines if enabled")
    parser.add_argument("--resume", action="store_true", help="Resume from the existing checkpoint")
    return parser.parse_args()


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
            debug=cli_args.debug, interval=cli_args.interval, start_idx=cli_args.start, end_idx=cli_args.end
        )
        # Process each step in range
        step_lst = steps_config.steps[(-1 * cli_args.last_N_step) :]
        logger.info(f"Further filtering results in {len(step_lst)} step(s)")

        # intialize the process class; compute this as it has not been saved yet
        abalation_processor = NeuronAblationProcessor(
            args=hydra_args, device=device, logger=logger, debug=cli_args.debug
        )
        base_save_dir = abalation_processor.get_save_dir()

        unigram_distrib, _ = load_unigram(
            model_name=hydra_args.model, device=device, dtype=settings.get_dtype(hydra_args.model)
        )
        min_freq, max_freq = abalation_processor.get_tail_threshold_stat(unigram_distrib)

        for step in step_lst:
            # Create save_path as a directory
            save_path = base_save_dir / str(step) / str(hydra_args.data_range_end)
            save_path.mkdir(parents=True, exist_ok=True)
            # Check for existing files with pattern matching expected output
            if cli_args.resume and (save_path / f"k{hydra_args.k}.feather").is_file():
                logger.info(f"Files for step {step} already exist. Skip!")
                continue
            logger.info(f"Processing step {step}")
            try:
                abalation_processor.process_single_step(step, unigram_distrib, min_freq, max_freq, save_path)

            except Exception as e:
                logger.error(f"Error processing step {step}: {e!s}")
                continue

            cleanup()


if __name__ == "__main__":
    logger.info(f"Current directory: {os.getcwd()}")
    main()
