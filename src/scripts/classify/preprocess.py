#!/usr/bin/env python
import argparse
import logging

from neuron_analyzer import settings
from neuron_analyzer.classify.preprocess import LabelAnnotator, get_threshold
from neuron_analyzer.load_util import JsonProcessor, StepPathProcessor

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train classifier to seperate different neurons.")

    parser.add_argument("-m", "--model", type=str, default="EleutherAI/pythia-410m-deduped", help="Target model name")
    parser.add_argument("--vector", type=str, default="longtail_50", choices=["mean", "longtail_elbow", "longtail_50"])
    parser.add_argument(
        "--heuristic", type=str, choices=["KL", "prob"], default="prob", help="heuristic besides mediation effect"
    )
    parser.add_argument(
        "--threshold_mode",
        type=str,
        default="unified",
        choices=["binary", "unified", "triclass"],
        help="which optimal threshold to choose from",
    )
    parser.add_argument(
        "--class_num",
        type=int,
        default=2,
        help="how many classes to classify",
    )
    parser.add_argument("--sel_by_med", type=bool, default=False, help="whether to select by mediation effect")
    parser.add_argument("--resume", action="store_true", help="Whether to resume from exisitng file")
    parser.add_argument("--debug", action="store_true", help="Compute the first 500 lines if enabled")
    parser.add_argument("--data_range_end", type=int, default=500, help="the selected datarange")
    return parser.parse_args()


#######################################################################################################
# Functions applied in the main scripts
#######################################################################################################


def configure_path(args):
    """Configure save path based on the setting."""
    save_heuristic = f"{args.heuristic}_med" if args.sel_by_med else args.heuristic
    data_path = settings.PATH.classify_dir / "data" / args.vector / args.model / save_heuristic
    model_path = settings.PATH.classify_dir / "model" / args.vector / args.model / save_heuristic
    eval_path = settings.PATH.classify_dir / "eval" / args.vector / args.model / save_heuristic
    model_path.mkdir(parents=True, exist_ok=True)
    eval_path.mkdir(parents=True, exist_ok=True)
    return (
        data_path,
        model_path,
        eval_path,
    )


def run_pipeline(args, data_path, model_path, eval_path, step_dirs: list, results_all: dict) -> dict:
    """Extract optimal threshold across multiple steps."""
    # load threshold
    threshold = get_threshold(
        data_path=data_path / "global_threshold_results.json", threshold_mode=f"{args.threshold_mode}_recommendation"
    )
    logger.info(f"{args.threshold_mode} threshold has been loaded: {threshold}")
    results_all = {}
    for step in step_dirs:
        # Load data preprocessor
        # try:
        summary = classify_neuron(args, data_path, model_path, eval_path, step, threshold)
        results_all[step[1]] = summary
    # except:
    # logger.info(f"Something wrong with step {step[1]}")
    JsonProcessor.save_json(results_all, eval_path / "results_summary.json")
    return results_all


def classify_neuron(args, data_path, model_path, eval_path, step, threshold):
    """Train and evaluate classifier from single step."""
    data_loader = LabelAnnotator(
        threshold_mode=args.threshold_mode,
        data_dir=data_path / str(step[1]) / str(args.data_range_end),
        resume=args.resume,
        threshold=threshold,
    )
    data_loader.run_pipeline()


#######################################################################################################
# Entry point of the script
#######################################################################################################


def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()
    # loop over different steps
    data_path, model_path, eval_path = configure_path(args)
    # initilize with the step dir
    step_processor = StepPathProcessor(data_path)
    results_all, step_dirs = step_processor.resume_results(args.resume, eval_path / "global_threshold_results.json")
    run_pipeline(args, data_path, model_path, eval_path, step_dirs, results_all)


if __name__ == "__main__":
    main()
