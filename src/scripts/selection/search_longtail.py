#!/usr/bin/env python
import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

from neuron_analyzer import settings
from neuron_analyzer.ablation.ablation import NeuronAblationProcessor
from neuron_analyzer.analysis.freq import UnigramAnalyzer
from neuron_analyzer.load_util import load_unigram
from neuron_analyzer.selection.neuron import NeuronSelector

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Select neurons based on single neuron heuristics only on ceonverged step."
    )

    parser.add_argument("-m", "--model", type=str, default="EleutherAI/pythia-70m-deduped", help="Target model name")
    parser.add_argument(
        "--vector",
        type=str,
        default="longtail_50",
        choices=["mean", "longtail_elbow", "longtail_50", "longtail_elbow_20"],
        help="boost or suppress long-tail prob",
    )
    parser.add_argument(
        "--effect", type=str, choices=["boost", "suppress"], default="suppress", help="boost or suppress long-tail prob"
    )
    parser.add_argument(
        "--heuristic", type=str, choices=["KL", "prob"], default="prob", help="heuristic besides mediation effect"
    )
    parser.add_argument(
        "--sel_freq",
        type=str,
        choices=["longtail", "common", None],
        default="longtail",
        help="freq by common or not",
    )

    parser.add_argument(
        "--max_freq",
        default=20,
        help="the proportion of selected max freq",
    )
    parser.add_argument(
        "--min_freq",
        default="elbow",
        help="the proportion of selected min freq",
    )

    parser.add_argument(
        "--step_mode", type=str, choices=["single", "multi"], default="multi", help="whether to compute multi steps"
    )
    parser.add_argument("--sel_by_med", action="store_true", help="whether to select by mediation effect")
    parser.add_argument("--debug", action="store_true", help="Compute the first 500 lines if enabled")
    parser.add_argument("--top_n", type=int, default=-1, help="The top n neurons to be selected")
    parser.add_argument("--stat_file", type=str, default="freq_elbow_20.json", help="stat filename")
    parser.add_argument("--tokenizer_name", type=str, default="EleutherAI/pythia-410m", help="Unigram tokenizer name")
    parser.add_argument("--data_range_end", type=int, default=500, help="the selected datarange")
    parser.add_argument("--k", type=int, default=10, help="use_bos_only if enabled")
    parser.add_argument("--seed", type=int, default=42, help="use_bos_only if enabled")
    parser.add_argument("--window_size", type=int, default=2000, help="use_bos_only if enabled")
    return parser.parse_args()


#######################################################################################################
# Fucntions applied in the main scripts
#######################################################################################################


def set_path(args) -> Path:
    """Set the saving path based on differnt configurations."""
    top_n_name = "all" if args.top_n == -1 else args.top_n
    filename = f"{args.data_range_end}_{top_n_name}.debug" if args.debug else f"{args.data_range_end}_{top_n_name}.csv"
    save_path = (
        settings.PATH.result_dir
        / "selection"
        / "neuron"
        / f"{args.min_freq}_{args.max_freq}"
        / args.model
        / args.heuristic
        / args.effect
        / filename
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    threshold_path = settings.PATH.freq_dir / args.model / args.stat_file
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    return save_path, threshold_path


def load_unigram_analyzer(args):
    """Load the target unigram analyzer based on whether filtering by freq."""
    if not args.sel_freq:
        return None

    return UnigramAnalyzer(device="cpu", model_name=args.model)


def filter_single(args, abl_path: Path, save_path: Path, threshold_path: Path) -> None:
    """Sort results of single ckpt."""
    feather_path = abl_path / str(args.data_range_end) / f"k{args.k}.feather"
    # initilize the class
    neuron_selector = NeuronSelector(
        feather_path=feather_path,
        debug=args.debug,
        top_n=args.top_n,
        step=-1,
        threshold_path=threshold_path,
        sel_freq=args.sel_freq,
        sel_by_med=args.sel_by_med,
        unigram_analyzer=load_unigram_analyzer(args),
    )
    neuron_df = neuron_selector.run_pipeline(heuristic=args.heuristic, effect=args.effect)
    # assign col headers
    neuron_df.to_csv(save_path)
    logger.info(f"Save file to {save_path}")


def filter_multi(args, neuron_df: pd.DataFrame, step: Path, abl_path: Path, threshold_path: Path) -> pd.DataFrame:
    """Sort results of single ckpt."""
    feather_path = abl_path / str(step.name) / str(args.data_range_end) / f"k{args.k}.feather"

    if feather_path.is_file():
        # initilize the class
        neuron_selector = NeuronSelector(
            feather_path=feather_path,
            debug=args.debug,
            top_n=args.top_n,
            step=step.name,
            threshold_path=threshold_path,
            sel_freq=args.sel_freq,
            sel_by_med=args.sel_by_med,
            unigram_analyzer=load_unigram_analyzer(args),
        )
        frame = neuron_selector.run_pipeline(heuristic=args.heuristic, effect=args.effect)
        neuron_df = pd.concat([neuron_df, frame])
    return neuron_df


#######################################################################################################
# Entry point of the script
#######################################################################################################


def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()
    # loop over different steps
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    abl_path = settings.PATH.result_dir / "ablations" / args.vector / args.model
    save_path, threshold_path = set_path(args)
    if save_path.is_file():
        logger.info(f"{save_path} already exists, skip!")

    # generate freq file
    abalation_processor = NeuronAblationProcessor(args=args, device=device, logger=logger)
    # base_save_dir = abalation_processor.get_save_dir()
    unigram_distrib, _ = load_unigram(model_name=args.model, device=device, dtype=settings.get_dtype(args.model))
    min_freq, max_freq = abalation_processor.get_tail_threshold_stat(unigram_distrib, save_path=threshold_path)

    if args.step_mode == "single":
        filter_single(args, abl_path, save_path, threshold_path)

    if args.step_mode == "multi":
        neuron_df = pd.DataFrame()
        # check whether the target file has been created
        for step in abl_path.iterdir():
            neuron_df = filter_multi(args, neuron_df, step, abl_path, threshold_path)
        # assign col headers
        neuron_df.to_csv(save_path)
        logger.info(f"Save file to {save_path}")


if __name__ == "__main__":
    main()
