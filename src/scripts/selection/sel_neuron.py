#!/usr/bin/env python
import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

from neuron_analyzer import settings
from neuron_analyzer.selection.neuron import NeuronSelector

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Select neurons based on single neuron heuristics.")

    parser.add_argument("-m", "--model", type=str, default="EleutherAI/pythia-70m-deduped", help="Target model name")
    parser.add_argument(
        "--vector",
        type=str,
        default="longtail_50",
        choices=["mean", "longtail_elbow", "longtail_50"],
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
        choices=["longtail_50", "common", None],
        default="longtail_50",
        help="freq by common or not",
    )
    parser.add_argument("--sel_by_med", type=bool, default=False, help="whether to select by mediation effect")
    parser.add_argument("--debug", action="store_true", help="Compute the first 500 lines if enabled")
    parser.add_argument("--top_n", type=int, default=10, help="The top n neurons to be selected")
    parser.add_argument("--stat_file", type=str, default="zipf_threshold_stats.json", help="stat filename")
    parser.add_argument("--tokenizer_name", type=str, default="EleutherAI/pythia-410m", help="Unigram tokenizer name")
    parser.add_argument("--data_range_end", type=int, default=500, help="the selected datarange")
    parser.add_argument("--k", type=int, default=10, help="use_bos_only if enabled")
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
        / args.sel_freq
        / args.model
        / args.heuristic
        / args.effect
        / filename
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    return save_path


#######################################################################################################
# Entry point of the script
#######################################################################################################


def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # loop over different steps
    abl_path = settings.PATH.result_dir / "ablations" / args.vector / args.model
    save_path = set_path(args)
    if save_path.is_file():
        logger.info(f"{save_path} already exists, skip!")
    else:
        neuron_df = pd.DataFrame()
        # check whether the target file has been created
        for step in abl_path.iterdir():
            feather_path = abl_path / str(step) / str(args.data_range_end) / f"k{args.k}.feather"
            if feather_path.is_file():
                # initilize the class
                # TODO: add unigram analyzer for intilization
                neuron_selector = NeuronSelector(
                    feather_path=feather_path,
                    debug=args.debug,
                    top_n=args.top_n,
                    step=step.name,
                    threshold_path=abl_path / args.stat_file,
                    sel_freq=args.sel_freq,
                    sel_by_med=args.sel_by_med,
                )
                frame = neuron_selector.run_pipeline(heuristic=args.heuristic, effect=args.effect)
                neuron_df = pd.concat([neuron_df, frame])
        # assign col headers
        neuron_df.to_csv(save_path)
        logger.info(f"Save file to {save_path}")


if __name__ == "__main__":
    main()
