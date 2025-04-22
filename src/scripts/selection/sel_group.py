#!/usr/bin/env python
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neuron_analyzer import settings
from neuron_analyzer.load_util import JsonProcessor, load_unigram
from neuron_analyzer.model_util import ModelHandler, NeuronLoader
from neuron_analyzer.selection.group import GroupModelAblationAnalyzer, NeuronGroupSearch, get_heuristics

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Seach neuron groups across different training steps.")

    parser.add_argument("-m", "--model", type=str, default="EleutherAI/pythia-70m-deduped", help="Target model name")
    parser.add_argument("--vector", choices=["mean", "longtail", "longtail_50"], default="longtail_50")
    parser.add_argument(
        "--effect", type=str, choices=["boost", "suppress"], default="suppress", help="boost or suppress long-tail prob"
    )
    parser.add_argument(
        "--heuristic", type=str, choices=["KL", "prob"], default="prob", help="heuristic besides mediation effect"
    )
    parser.add_argument("--top_n", type=int, default=10, help="use_bos_only if enabled")
    parser.add_argument("--data_range_start", type=int, default=0, help="the selected datarange")
    parser.add_argument("--data_range_end", type=int, default=500, help="the selected datarange")
    parser.add_argument("--k", type=int, default=10, help="use_bos_only if enabled")
    parser.add_argument("--resume", action="store_true", help="Whether to resume from exisitng file")
    parser.add_argument("--debug", action="store_true", help="Compute the first 500 lines if enabled")
    parser.add_argument("--seed", type=int, default=42, help="random seed to select neurons")
    parser.add_argument("--dataset", type=str, default="stas/c4-en-10k", help="random seed to select neurons")
    return parser.parse_args()


#######################################################################################################
# Fucntions applied in the main scripts
#######################################################################################################


def load_neuron_index(args) -> list[int]:
    """Load neuron indices and delta loss"""
    # assume that file has already been built
    neuron_path = (
        settings.PATH.neuron_dir
        / "neuron"
        / args.vector
        / args.model
        / args.heuristic
        / args.effect
        / f"{args.data_range_end}_all.csv"
    )
    if not neuron_path.is_file():
        logger.error("Please run the individual neuron selection first!")
    # read and parse the input .csv file
    neuron_loader = NeuronLoader(file_path=neuron_path, top_n=args.top_n)
    step_neuron, layer_num = neuron_loader.load_neuron_dict(
        key_col="step",
        value_col="top_neurons",
        random_base=False,
    )
    step_stat = neuron_loader.load_neuron_stat(key_col="step", value_col=args.heuristic)
    return layer_num, step_neuron, step_stat


def load_tail_threshold_stat(args) -> tuple[float | None, dict | None]:
    """Load longtail threshold from the jason file."""
    data = JsonProcessor.load_json(settings.PATH.ablation_dir / args.vector / args.model / "zipf_threshold_stats.json")
    return data["threshold_info"]["probability"]


def set_path(args) -> Path:
    """Set save and cache path."""
    save_path = settings.PATH.neuron_dir / "group" / args.vector / args.model / args.heuristic / args.effect
    save_path.mkdir(parents=True, exist_ok=True)
    return save_path


def sort_path(abl_path: Path) -> list[Path]:
    """Get the sorted directory by steps."""
    # Get all step directories and sort them by the number after "step" in descending order
    step_dirs = []
    for step in abl_path.iterdir():
        if step.is_dir():
            # Extract the number after "step" prefix
            step_num = int(step.name)  # Remove "step" prefix and convert to integer
            step_dirs.append((step, step_num))
    # Sort directories by step number in descending order
    step_dirs.sort(key=lambda x: x[1], reverse=True)
    return step_dirs


def resume_results(resume, save_path: Path, step_dirs: list[Path]):
    """Resume results from the existing directory."""
    # resume file and update the steps
    if resume and save_path.is_file():
        # load json file
        final_results = JsonProcessor.load_json(save_path)
        # get the generated ckpts
        completed_results = list(final_results.keys())
        # remove the existing files
        step_dirs = [p for p in step_dirs if p not in completed_results]
        logger.info(f"Resume {len(step_dirs) - len(completed_results)} states from {save_path}.")
        return final_results, step_dirs
    return {}, step_dirs


class NeuronGroupSelector:
    """Class to run experiments of neuron group search."""

    def __init__(self, args, device: str, layer_num: int, step_neuron: dict, step_stat: dict, save_dir: Path):
        """Initialize the ablation processor with configuration."""
        # Initialize parameters from args
        self.args = args
        self.device = device
        self.layer_num = layer_num
        self.step_neuron = step_neuron
        self.step_stat = step_stat
        self.save_dir = save_dir

        # Initialize random seeds
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.set_grad_enabled(False)

    def process_single_step(self, step: int, unigram_distrib, longtail_threshold) -> None:
        """Process a single step with the given configuration."""
        # initlize the model handler class
        model_handler = ModelHandler()
        # Load model and tokenizer for specific step
        model, tokenizer = model_handler.load_model_and_tokenizer(
            step=step,
            model_name=self.args.model,
            hf_token_path=settings.PATH.unigram_dir / "hf_token.txt",
            device=self.device,
        )

        # Load and process dataset
        tokenized_data = model_handler.tokenize_data(
            dataset=self.args.dataset,
            data_range_start=self.args.data_range_start,
            data_range_end=self.args.data_range_end,
            seed=self.args.seed,
        )

        # load the single neuron and heuristic list
        neuron_lst, stat_lst = self.step_neuron[int(step)], self.step_stat[int(step)]
        # initilize the eval
        entropy_df = pd.read_csv(
            settings.PATH.ablation_dir
            / self.args.vector
            / self.args.model
            / str(step)
            / str(self.args.data_range_end)
            / "entropy_df.csv"
        )
        if self.args.debug:
            row_num = 1_000_000
            entropy_df = entropy_df.head(row_num)
            logger.info(f"Enter debugging mode. Apply {row_num} rows.")

        group_evaluator = GroupModelAblationAnalyzer(
            model=model,
            device=self.device,
            tokenized_data=tokenized_data,
            unigram_distrib=unigram_distrib,
            entropy_df=entropy_df,
            layer_idx=self.layer_num,
            k=self.args.k,
            ablation_mode=self.args.vector,
            longtail_threshold=longtail_threshold,
        )

        self.cache_dir = self._get_save_path(step)
        # initialize the neuron group search
        maxmize_heuristic = get_heuristics(self.args.effect)
        logger.info(f"Select {self.args.effect} neurons. Maximize heuristics: {maxmize_heuristic}.")
        search = NeuronGroupSearch(
            neurons=neuron_lst,
            individual_delta_loss=stat_lst,
            evaluator=group_evaluator,
            target_size=self.args.top_n,
            cache_dir=self.cache_dir,
            maximize=maxmize_heuristic,
        )
        # Get the best result using all methods
        results = search.get_best_result()
        logger.info(f"Method with highest heuristic: {results['best']}")
        logger.info(f"Method with target length: {results['target_size']}")
        return results

    def _get_save_path(self, step) -> Path:
        """Get the savepath based on current configurations."""
        cache_dir = self.save_dir / "cache" / str(self.args.top_n) / str(step)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir


#######################################################################################################
# Entry point of the script
#######################################################################################################


def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # configure save_dir
    save_dir = set_path(args)
    save_path = save_dir / f"{args.data_range_end}_{args.top_n}.json"
    # load neuron
    layer_num, step_neuron, step_stat = load_neuron_index(args)
    unigram_distrib, _ = load_unigram(model_name=args.model, device=device)
    longtail_threshold = load_tail_threshold_stat(args)
    # initilize the selector class
    group_selector = NeuronGroupSelector(
        args=args, device=device, layer_num=layer_num, step_neuron=step_neuron, step_stat=step_stat, save_dir=save_dir
    )
    # loop over different steps
    abl_path = settings.PATH.result_dir / "ablations" / args.vector / args.model
    # order the steps in descending way
    step_dirs = sort_path(abl_path)
    final_results, step_dirs = resume_results(args.resume, save_path, step_dirs)
    # Process steps in the sorted order
    for _, step in step_dirs:
        try:
            results = group_selector.process_single_step(step, unigram_distrib, longtail_threshold)
            final_results[step] = results
            # save the intermediate checkpoints
            JsonProcessor.save_json(final_results, save_path)
            logger.info(f"Save the results to {save_path}")
        except:
            logger.info(f"Something wrong with {step}")


if __name__ == "__main__":
    main()
