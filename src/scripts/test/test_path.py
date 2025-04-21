#!/usr/bin/env python
import argparse
import logging
from pathlib import Path

from neuron_analyzer import settings

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
    parser.add_argument("--debug", action="store_true", help="Compute the first 500 lines if enabled")
    parser.add_argument("--seed", type=int, default=42, help="random seed to select neurons")
    parser.add_argument("--dataset", type=str, default="stas/c4-en-10k", help="random seed to select neurons")
    return parser.parse_args()


#######################################################################################################
# Fucntions applied in the main scripts
#######################################################################################################


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


#######################################################################################################
# Entry point of the script
#######################################################################################################


def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()
    # loop over different steps
    abl_path = settings.PATH.result_dir / "ablations" / args.vector / args.model
    # order the steps in descending way
    step_dirs = sort_path(abl_path)
    print(step_dirs)
    # Process steps in the sorted order
    for step, _ in step_dirs:
        print(step)


if __name__ == "__main__":
    main()
