#!/usr/bin/env python
import argparse
import logging
from pathlib import Path

from neuron_analyzer import settings
from neuron_analyzer.classify.analyses import NeuronHypothesisTester
from neuron_analyzer.classify.classifier import NeuronClassifier
from neuron_analyzer.classify.preprocess import LabelAnnotator, get_threshold
from neuron_analyzer.load_util import JsonProcessor, StepPathProcessor

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train classifier to seperate different neurons.")

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


def run_pipeline(args, data_dir: Path, step_dirs: list, output_dir: Path) -> dict:
    """Extract optimal threshold across multiple steps."""
    # load threshold
    threshold = get_threshold(data_path, args.threshold_mode)
    results_all = {}
    for step in step_dirs:
        # Load data preprocessor
        try:
            data_loader = LabelAnnotator(
                resume=args.resume, threshold=threshold, threshold_mode=args.threshold_mode, data_dir=data_dir
            )
            X, y, neuron_indices, metadata = data_loader.run_pipeline()
            # train and evlauate the classifiers
            classifier = NeuronClassifier(
                X=X,
                y=y,
                neuron_indices=neuron_indices,
                metadata=metadata,
                classification_mode=args.classification_mode,
            )
            classifier_results = classifier.run_all_analyses(test_size=0.2)
            classifier.save_results(output_dir / "classifier_results.json")
            # initial analyses on the results
            hypthesis_summary = NeuronHypothesisTester(
                classifier_results=classifier_results, out_path=output_dir / "seperation_results.json"
            )
            summary = hypthesis_summary.run_pipeline()
            results_all[step[1]] = summary
        except:
            logger.info(f"Something wrong with step {step[1]}")

    JsonProcessor.save_json(results_all, output_dir / "results_summary.json")
    return results_all


#######################################################################################################
# Entry point of the script
#######################################################################################################


def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()
    # loop over different steps
    out_dir, abl_path = configure_path(args)
    # initilize with the step dir
    step_processor = StepPathProcessor(abl_path)
    save_path = out_dir / args.filename
    stat_results, step_dirs = step_processor.resume_results(args.resume, save_path)
    # Process multiple steps
    run_pipeline(args, data_dir, step_dirs, output_dir)


if __name__ == "__main__":
    main()
