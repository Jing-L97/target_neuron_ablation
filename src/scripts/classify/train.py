#!/usr/bin/env python
import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

from neuron_analyzer import settings
from neuron_analyzer.analysis.geometry_util import NeuronGroupAnalyzer, get_group_name
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
        "--group_size", type=str, choices=["best", "target_size"], default="best", help="different group size"
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


class Trainer:
    """Class for running the entire neuron classification pipeline."""

    def __init__(self, args: Any, data_path: Path, model_path: Path, eval_path: Path, step_dirs: list[tuple[str, str]]):
        """Initialize the pipeline with all necessary parameters."""
        self.args = args
        self.data_path = data_path
        self.model_path = model_path
        self.eval_path = eval_path
        self.step_dirs = step_dirs

    def run_pipeline(self) -> dict[str, Any]:
        """Extract optimal threshold across multiple steps and run classification."""
        threshold = self._load_threshold()
        classification_condition = self._get_classification_condition()
        results_all = {}

        for step in self.step_dirs:
            try:
                # Load data
                X, y, neuron_indices = self._load_data(step, threshold, classification_condition)
                # configure the save path
                step_model_path, step_eval_path = self._configure_save_path(step, classification_condition)
                # Train classifier
                summary = self._classify_neuron(X, y, neuron_indices, step_model_path, step_eval_path)
                results_all[step[1]] = summary
            except Exception as e:
                logger.info(f"Error processing step {step[1]}: {e!s}")
                logger.debug("Exception details:", exc_info=True)

        # Save all results
        JsonProcessor.save_json(results_all, self.eval_path / "results_summary.json")
        return results_all

    def _load_data(
        self, step: tuple[str, str], threshold: float | None, classification_condition: str
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Load and prepare data for classification from a single step."""
        # Filter features with unequal length
        feature_loader = FeatureLoader(data_dir=self.data_path / str(step[1]) / str(self.args.data_range_end))
        data, fea = feature_loader.run_pipeline()
        logger.info("Features have been loaded.")

        # Label data
        if self.args.label_type == "threshold":
            threshold_labeler = ThresholdLabeler(threshold=threshold, data=data)
            labels, neuron_indices = threshold_labeler.run_pipeline()
        elif self.args.label_type == "fixed":
            fixed_labeler = FixedLabeler(data=data, class_indices=self._load_neuron_indices(step_path=step[0]))
            fea, labels, neuron_indices = fixed_labeler.run_pipeline()
        else:
            raise ValueError(f"Unsupported label_type: {self.args.label_type}")

        # Integrate features and labels
        data_loader = DataLoader(
            X=fea,
            y=labels,
            neuron_indices=neuron_indices,
            out_path=self.data_path
            / str(step[1])
            / str(self.args.data_range_end)
            / f"data_{classification_condition}.json",
        )
        X, y, neuron_indices = data_loader.run_pipeline()

        return X, y, neuron_indices

    def _classify_neuron(
        self,
        X: np.ndarray,
        y: np.ndarray,
        neuron_indices: list[str],
        step_model_path: Path,
        step_eval_path: Path,
    ) -> dict[str, Any]:
        """Train and evaluate classifier from single step."""
        # Train and evaluate the classifiers
        classifier = NeuronClassifier(
            X=X,
            y=y,
            model_path=step_model_path,
            eval_path=step_eval_path,
            neuron_indices=neuron_indices,
            class_num=self.args.class_num,
            test_size=0.2,
        )
        classifier_results = classifier.run_pipeline()

        # Initial analyses on the results
        hypothesis_tester = NeuronHypothesisTester(
            classifier_results=classifier_results,
            out_path=step_eval_path / "separation_analysis.json",
            resume=self.args.resume,
        )
        summary = hypothesis_tester.run_pipeline()

        return summary

    def _load_neuron_indices(self, step_path: str) -> dict[str, list[int]]:
        """Load neuron indices from the existing file."""
        neuron_analyzer = NeuronGroupAnalyzer(args=self.args, device="cpu", step_path=step_path)
        boost_neuron_indices, suppress_neuron_indices, random_indices = neuron_analyzer.load_neurons()

        return {
            "boost": boost_neuron_indices,
            "suppress": suppress_neuron_indices,
            "random": random_indices,
        }

    def _load_threshold(self) -> float | None:
        """Load threshold if needed based on label type."""
        if self.args.label_type == "threshold":
            return get_threshold(
                data_path=self.data_path / "global_threshold_results.json",
                threshold_mode=f"{self.args.threshold_mode}_recommendation",
            )
        return None

    def _get_classification_condition(self) -> str:
        """Get the notation string for different classification conditions."""
        if self.args.label_type == "threshold":
            return self.args.threshold_mode
        if self.args.label_type == "fixed":
            return str(self.args.top_n)
        return "undefined"

    def _configure_save_path(self, step: tuple[str, str], classification_condition: str) -> tuple[Path, Path]:
        """Configure dave path based on different conditions."""
        step_model_path = (
            self.model_path
            / str(step[1])
            / str(self.args.data_range_end)
            / classification_condition
            / str(self.args.class_num)
        )
        step_eval_path = (
            self.eval_path
            / str(step[1])
            / str(self.args.data_range_end)
            / classification_condition
            / str(self.args.class_num)
        )
        if self.args.label_type == "fixed":
            group_name = get_group_name(self.args)
            return step_model_path / group_name, step_eval_path / group_name
        return step_model_path, step_eval_path


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
    step_dirs = step_processor.sort_paths()
    data_path, model_path, eval_path = configure_path(args)
    trainer = Trainer(args, data_path, model_path, eval_path, step_dirs)
    trainer.run_pipeline()


if __name__ == "__main__":
    main()
