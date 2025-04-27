#!/usr/bin/env python
import argparse
import logging
from pathlib import Path

import torch

from neuron_analyzer import settings
from neuron_analyzer.classify.sel_threshold import (
    GlobalThresholdOptimizer,
    NeuronFeatureExtractor,
    SingleThresholdCalculator,
)
from neuron_analyzer.load_util import StepPathProcessor

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Select neurons based on single neuron heuristics.")

    parser.add_argument("-m", "--model", type=str, default="EleutherAI/pythia-70m-deduped", help="Target model name")
    parser.add_argument("--vector", type=str, default="longtail_50", choices=["mean", "longtail_elbow", "longtail_50"])
    parser.add_argument(
        "--heuristic", type=str, choices=["KL", "prob"], default="prob", help="heuristic besides mediation effect"
    )
    parser.add_argument("--sel_longtail", type=bool, default=True, help="whether to filter by longtail token")
    parser.add_argument("--sel_by_med", type=bool, default=False, help="whether to select by mediation effect")
    parser.add_argument("--filename", type=str, default="global_threshold_results.json", help="Target stat filename")
    parser.add_argument("--resume", action="store_true", help="Whether to resume from exisitng file")
    parser.add_argument("--debug", action="store_true", help="Compute the first 500 lines if enabled")
    parser.add_argument("--top_n", type=int, default=10, help="palceholder to intilize the class")
    parser.add_argument("--stat_file", type=str, default="zipf_threshold_stats.json", help="stat filename")
    parser.add_argument("--tokenizer_name", type=str, default="EleutherAI/pythia-410m", help="Unigram tokenizer name")
    parser.add_argument("--data_range_end", type=int, default=500, help="the selected datarange")
    parser.add_argument("--k", type=int, default=10, help="use_bos_only if enabled")
    return parser.parse_args()


#######################################################################################################
# Functions applied in the main scripts
#######################################################################################################


def configure_path(args):
    """Configure save path based on the setting."""
    save_heuristic = f"{args.heuristic}_med" if args.sel_by_med else args.heuristic
    save_path = settings.PATH.classify_dir / "data" / args.vector / args.model / save_heuristic
    abl_path = settings.PATH.result_dir / "ablations" / args.vector / args.model
    save_path.mkdir(parents=True, exist_ok=True)
    return save_path, abl_path


def select_threshold(
    args, device: str, abl_path: Path, out_dir: Path, save_path: Path, stat_results: dict, step_dirs: list
) -> dict:
    """Extract optimal threshold across multiple steps."""
    for step in step_dirs:
        # Initialize the extractor
        extractor = NeuronFeatureExtractor(
            args=args, abl_path=abl_path, step_path=step[0], step_num=str(step[1]), device=device, out_dir=out_dir
        )
        data = extractor.run_pipeline()
        calculator = SingleThresholdCalculator(args=args, data=data, step_num=str(step[1]), out_dir=out_dir)
        threshold_result = calculator.run_pipeline()
        stat_results[step[1]] = threshold_result
    logger.info(f"Loaded threhsold stats of {len(stat_results)} steps.")
    optimize_global = GlobalThresholdOptimizer(
        args=args, out_dir=out_dir, stat_results=stat_results, save_path=save_path
    )
    optimize_global.run_pipeline()


#######################################################################################################
# Entry point of the script
#######################################################################################################


def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # loop over different steps
    out_dir, abl_path = configure_path(args)
    # initilize with the step dir
    step_processor = StepPathProcessor(abl_path)
    save_path = out_dir / args.filename
    stat_results, step_dirs = step_processor.resume_results(args.resume, save_path)
    # Process multiple steps
    select_threshold(args, device, abl_path, out_dir, save_path, stat_results, step_dirs)


if __name__ == "__main__":
    main()
