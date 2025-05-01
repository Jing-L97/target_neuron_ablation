#!/usr/bin/env python
import argparse
import logging

from neuron_analyzer import settings
from neuron_analyzer.analysis.geometry_util import NeuronGroupAnalyzer
from neuron_analyzer.classify.analyses import NeuronHypothesisTester
from neuron_analyzer.classify.classifier import NeuronClassifier
from neuron_analyzer.classify.preprocess import (
    DataLoader,
    FeatureLoader,
    FixedLabeler,
    ThresholdLabeler,
    get_threshold,
)
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
    parser.add_argument(
        "--group_type", type=str, choices=["individual", "group"], default="individual", help="different neuron groups"
    )
    parser.add_argument(
        "--threshold_mode",
        type=str,
        default="unified",
        choices=["binary", "unified", "triclass"],
        help="which optimal threshold to choose from",
    )
    parser.add_argument(
        "--label_type",
        type=str,
        default="fixed",
        choices=["fixed", "threshold"],
        help="selected label type",
    )
    parser.add_argument(
        "--class_num",
        type=int,
        default=2,
        help="how many classes to classify",
    )
    parser.add_argument("--load_stat", type=bool, default=True, help="Whether to load from existing index")
    parser.add_argument(
        "--exclude_random", type=bool, default=True, help="Whether to include the random neuron indices"
    )
    parser.add_argument("--sel_by_med", type=bool, default=False, help="whether to select by mediation effect")
    parser.add_argument("--top_n", type=int, default=100, help="The top n neurons to be selected")
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
    threshold = load_threshold(args, data_path)
    classification_condition = get_classification_condition(args)
    results_all = {}
    for step in step_dirs:
        try:
            # load data
            X, y, neuron_indices = load_data(args, data_path, step, threshold, classification_condition)
            # train classifier
            summary = classify_neuron(args, X, y, neuron_indices, model_path, eval_path, step, classification_condition)
            results_all[step[1]] = summary
        except:
            logger.info(f"Something wrong with  step {step[1]}")
    JsonProcessor.save_json(results_all, eval_path / "results_summary.json")
    return results_all


def load_data(args, data_path, step, threshold, classification_condition):
    """Train and evaluate classifier from single step."""
    # filter features with unequal length
    feature_loader = FeatureLoader(data_dir=data_path / str(step[1]) / str(args.data_range_end))
    data, fea = feature_loader.run_pipeline()
    logger.info("Features have been loaded.")
    # label data
    if args.label_type == "threshold":
        threshold_labeler = ThresholdLabeler(threshold=threshold, data=data)
        labels, neuron_indices = threshold_labeler.run_pipeline()
    if args.label_type == "fixed":
        fixed_labeler = FixedLabeler(data=data, class_indices=load_neuron_indices(args, step_path=step[0]))
        fea, labels, neuron_indices = fixed_labeler.run_pipeline()
    # integrate features and labels
    data_loader = DataLoader(
        X=fea,
        y=labels,
        neuron_indices=neuron_indices,
        out_path=data_path / str(step[1]) / str(args.data_range_end) / f"data_{classification_condition}.json",
        resume=args.resume,
    )
    X, y, neuron_indices = data_loader.run_pipeline()
    return X, y, neuron_indices


def classify_neuron(args, X, y, neuron_indices, model_path, eval_path, step, classification_condition):
    """Train and evaluate classifier from single step."""
    # train and evlauate the classifiers
    classifier = NeuronClassifier(
        X=X,
        y=y,
        model_path=model_path / str(step[1]) / str(args.data_range_end) / classification_condition,
        eval_path=eval_path / str(step[1]) / str(args.data_range_end) / classification_condition,
        neuron_indices=neuron_indices,
        class_num=args.class_num,
        test_size=0.2,
    )
    classifier_results = classifier.run_pipeline()
    # initial analyses on the results
    hypthesis_summary = NeuronHypothesisTester(
        classifier_results=classifier_results,
        out_path=eval_path
        / str(step[1])
        / str(args.data_range_end)
        / classification_condition
        / "seperation_analysis.json",
        resume=args.resume,
    )
    summary = hypthesis_summary.run_pipeline()
    return summary


def load_neuron_indices(args, step_path) -> dict:
    """Load neuron indices from the existing file."""
    neuron_analyzer = NeuronGroupAnalyzer(args=args, device="cpu", step_path=step_path)
    boost_neuron_indices, suppress_neuron_indices, random_indices = neuron_analyzer.load_neurons()
    return {
        "boost": boost_neuron_indices,
        "suppress": suppress_neuron_indices,
        "random": random_indices,
    }


def load_threshold(args, data_path) -> float:
    """Load threshold if needed."""
    if args.label_type == "threshold":
        return get_threshold(
            data_path=data_path / "global_threshold_results.json",
            threshold_mode=f"{args.threshold_mode}_recommendation",
        )
    return None


def get_classification_condition(args) -> str:
    """Get the notation for different conditions."""
    if args.label_type == "threshold":
        return args.threshold_mode
    if args.label_type == "fixed":
        return str(args.top_n)
    return "undefined"


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
