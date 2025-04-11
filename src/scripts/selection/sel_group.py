#!/usr/bin/env python
import argparse
import logging

import pandas as pd

from neuron_analyzer import settings

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract word surprisal across different training steps.")

    parser.add_argument("-m", "--model", type=str, default="EleutherAI/pythia-70m-deduped", help="Target model name")
    parser.add_argument("--vector", choices=["mean", "longtail"], default="longtail")
    parser.add_argument(
        "--effect", type=str, choices=["boost", "suppress"], default="suppress", help="boost or suppress long-tail prob"
    )
    parser.add_argument("--top_n", type=int, default=10, help="use_bos_only if enabled")
    parser.add_argument("--data_range_end", type=int, default=500, help="the selected datarange")
    parser.add_argument("--k", type=int, default=10, help="use_bos_only if enabled")
    return parser.parse_args()


# TODO: cache the group activation as we want to save computation

# filter tokens or select from .feather file

# save intermediate results


def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()

    # loop over different steps
    abl_path = settings.PATH.result_dir / "ablations" / args.vector / args.model
    save_path = (
        settings.PATH.result_dir
        / "token_freq"
        / args.effect
        / args.vector
        / args.model
        / f"{args.data_range_end}_{args.top_n}.csv"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.is_file():
        logger.info(f"{save_path} already exists, skip!")
    else:
        neuron_df = pd.DataFrame()
        # check whether the target file has been created
        for step in abl_path.iterdir():
            feather_path = abl_path / str(step) / str(args.data_range_end) / f"k{args.k}.feather"
            frame = select_top_token_frequency_neurons(feather_path, args.top_n, step.name, args.effect)
            neuron_df = pd.concat([neuron_df, frame])
        # assign col headers
        neuron_df.to_csv(save_path)
        logger.info(f"Save file to {save_path}")


if __name__ == "__main__":
    main()
