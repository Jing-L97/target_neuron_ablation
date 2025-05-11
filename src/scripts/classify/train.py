#!/usr/bin/env python
import argparse
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neuron_analyzer import settings
from neuron_analyzer.analysis.freq import UnigramAnalyzer
from neuron_analyzer.analysis.geometry_util import NeuronGroupAnalyzer, get_group_name
from neuron_analyzer.classify.analyses import NeuronHypothesisTester
from neuron_analyzer.classify.classifier import NeuronClassifier
from neuron_analyzer.classify.preprocess import (
    DataLoader,
    FeatureLoader,
    FixedLabeler,
    NeuronFeatureExtractor,
    ThresholdLabeler,
    get_threshold,
)
from neuron_analyzer.load_util import StepPathProcessor, load_unigram
from neuron_analyzer.model_util import NeuronLoader
from neuron_analyzer.selection.neuron import generate_random_indices

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Only show errors, not warnings

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
    parser.add_argument(
        "--sel_freq",
        type=str,
        choices=["longtail_50", "common"],
        default="longtail_50",
    )
    parser.add_argument("--load_stat", type=bool, default=True, help="Whether to load from existing index")
    parser.add_argument("--exclude_random", type=bool, default=True, help="Include all neuron indices if set True")
    parser.add_argument(
        "--index_type",
        type=str,
        default="extreme",
        choices=["baseline", "extreme", "random"],
        help="the index type labels",
    )
    parser.add_argument("--sel_by_med", type=bool, default=False, help="whether to select by mediation effect")
    parser.add_argument("--fea_dim", type=int, default=500, help="Number of tokens as the activation feature")
    parser.add_argument("--top_n", type=int, default=50, help="The top n neurons to be selected")
    parser.add_argument("--resume", action="store_true", help="Whether to resume from exisitng file")
    parser.add_argument("--debug", action="store_true", help="Compute the first 500 lines if enabled")
    parser.add_argument("--k", type=int, default=10, help="Number of chunks")
    parser.add_argument("--tokenizer_name", type=str, default="EleutherAI/pythia-410m", help="Unigram tokenizer name")
    parser.add_argument("--stat_file", type=str, default="zipf_threshold_stats.json", help="stat filename")
    parser.add_argument("--sel_longtail", type=bool, default=True, help="whether to filter by longtail token")
    parser.add_argument("--data_range_end", type=int, default=500, help="the selected datarange")
    return parser.parse_args()


#######################################################################################################
# Functions applied in the main scripts
#######################################################################################################


def configure_path(args):
    """Configure save path based on the setting."""
    save_heuristic = f"{args.heuristic}_med" if args.sel_by_med else args.heuristic
    path_suffix = Path(args.sel_freq) / args.model / str(args.fea_dim)
    data_path = settings.PATH.classify_dir / "data" / path_suffix
    model_path = settings.PATH.classify_dir / "model" / path_suffix
    eval_path = settings.PATH.classify_dir / "eval" / path_suffix
    feather_path = settings.PATH.result_dir / "ablations" / "mean" / args.model
    entropy_path = settings.PATH.result_dir / "ablations" / "longtail_50" / args.model
    data_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)
    eval_path.mkdir(parents=True, exist_ok=True)
    return data_path, model_path, eval_path, feather_path, entropy_path


class Trainer:
    """Class for running the entire neuron classification pipeline."""

    def __init__(
        self,
        args: Any,
        feather_path: Path,
        entropy_path: Path,
        data_path: Path,
        model_path: Path,
        eval_path: Path,
        step_dirs: list[tuple[str, str]],
    ):
        """Initialize the pipeline with all necessary parameters."""
        self.args = args
        self.data_path = data_path
        self.model_path = model_path
        self.eval_path = eval_path
        self.step_dirs = step_dirs
        self.feather_path = feather_path
        self.entropy_path = entropy_path
        self.run_baseline = True if self.args.index_type == "baseline" else False

    def run_pipeline(self) -> dict[str, Any]:
        """Extract optimal threshold across multiple steps and run classification."""
        threshold = self._load_threshold()
        classification_condition = self._get_classification_condition()
        # load neuron index file if needed
        if self.args.index_type == "extreme":
            self.neuron_file = self._load_neuron_file()

        for step in self.step_dirs:
            try:
                # Load data
                logger.info("----------Stage 1: Loading training data---------------------------")
                X, y, neuron_indices = self._load_data(step, threshold, classification_condition)
                # configure the save path
                step_model_path, step_eval_path = self._configure_save_path(step, classification_condition)
                # Train classifier
                logger.info("----------Stage 2: Training data for hyperplanes-------------------")
                _ = self._classify_neuron(X, y, neuron_indices, step_model_path, step_eval_path)
                logger.info(f"############################################ Finished step {step[1]}")

            except Exception as e:
                logger.info(f"Error processing step {step[1]}: {e!s}")

    def _load_data(
        self, step: tuple[str, str], threshold: float | None, classification_condition: str
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Load and prepare data for classification from a single step."""
        # load data
        data, fea = self._load_fea(step)

        # Label data
        if self.args.label_type == "threshold":
            logger.info("--Step 2: Labeling data from the selected threshold")
            threshold_labeler = ThresholdLabeler(threshold=threshold, data=data)
            labels, neuron_indices = threshold_labeler.run_pipeline()
        elif self.args.label_type == "fixed":
            logger.info("--Step 2: Labeling data from the prelabeled data")

            fixed_labeler = FixedLabeler(
                data=data,
                run_baseline=self.run_baseline,
                class_indices=self._load_neuron_indices(data=data, step_path=step[0], step=step[1]),
            )
            fea, labels, neuron_indices = fixed_labeler.run_pipeline()
        else:
            raise ValueError(f"Unsupported label_type: {self.args.label_type}")

        # Integrate features and labels
        logger.info("--Step 3: Building dataset for training")
        filename = (
            f"data_{classification_condition}_{self.args.index_type}.json"
            if self.args.label_type == "fixed"
            else f"data_{classification_condition}.json"
        )

        data_loader = DataLoader(
            X=fea,
            y=labels,
            neuron_indices=neuron_indices,
            out_path=self.data_path / str(step[1]) / str(self.args.data_range_end) / filename,
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
        # configure save path
        filename = (
            f"separation_analysis_{self.args.index_type}.json"
            if self.args.label_type == "fixed"
            else "separation_analysis.json"
        )
        # Train and evaluate the classifiers
        classifier = NeuronClassifier(
            X=X,
            y=y,
            model_path=step_model_path,
            eval_path=step_eval_path,
            neuron_indices=neuron_indices,
            class_num=self.args.class_num,
            test_size=0.2,
            index_type=self.args.index_type,
        )
        classifier_results = classifier.run_pipeline()

        # Initial analyses on the results
        hypothesis_tester = NeuronHypothesisTester(
            classifier_results=classifier_results,
            out_path=step_eval_path / filename,
            resume=self.args.resume,
        )
        summary = hypothesis_tester.run_pipeline()

        return summary

    def _load_neuron_indices(self, data: dict, step_path: str, step: str) -> dict[str, list[int]]:
        """Load neuron indices from the existing file."""
        neuron_analyzer = NeuronGroupAnalyzer(args=self.args, device="cpu", step_path=step_path)
        boost_neuron_indices, suppress_neuron_indices, random_indices = neuron_analyzer.load_neurons()
        group_size = max(len(boost_neuron_indices), len(suppress_neuron_indices))
        special_indices = set(boost_neuron_indices + suppress_neuron_indices + random_indices)

        if self.args.index_type == "extreme":
            baseline_indices = self._load_extreme_neurons(step)
            return {"boost": boost_neuron_indices, "suppress": suppress_neuron_indices, "random": baseline_indices}

        if self.args.index_type == "baseline":
            baseline_indices = generate_random_indices(
                all_neuron_indices=data["neuron_features"].keys(),
                special_indices=special_indices,
                group_size=group_size,
                num_random_groups=1,
            )
            # merge mulitple indices
            baseline_indices = [item for sublist in baseline_indices for item in sublist]
            return {"baseline": baseline_indices, "random": random_indices}
        return {
            "boost": boost_neuron_indices,
            "suppress": suppress_neuron_indices,
            "random": random_indices,
        }

    def _load_fea(self, step) -> tuple[dict, dict]:
        # intialize the unigram analyzer
        unigram_distrib, unigram_count = load_unigram(model_name=self.args.model, device="cpu")
        # intialize the unigram analyzer
        unigram_analyzer = UnigramAnalyzer(
            device="cpu", model_name=self.args.model, unigram_distrib=unigram_distrib, unigram_count=unigram_count
        )
        # initilize the selector class
        data_path = self.data_path / str(step[1]) / str(self.args.data_range_end) / "features.json"
        logger.info(f"Target data path is {data_path}")
        if data_path.is_file() and self.args.resume:
            logger.info(f"--Step 1: Resuming feature data from {data_path}")

        # rebuild the features if not setting resume
        else:
            logger.info("--Step 1:Building feature data from scratch.")
            feature_extractor = NeuronFeatureExtractor(
                args=self.args,
                feather_path=self.feather_path,
                entropy_path=self.entropy_path,
                step_path=step[0],
                unigram_analyzer=unigram_analyzer,
                out_dir=self.data_path / str(step[1]) / str(self.args.data_range_end),
                step_num=step[1],
                device="cpu",
            )
            feature_extractor.run_pipeline()

        # load features fromt ehsave path
        feature_loader = FeatureLoader(data_path=data_path)
        data, fea = feature_loader.run_pipeline()
        logger.info("Features have been loaded.")
        return data, fea

    def _load_threshold(self) -> float | None:
        """Load threshold if needed based on label type."""
        if self.args.label_type == "threshold":
            return get_threshold(
                data_path=self.data_path / "global_threshold_results.json",
                threshold_mode=f"{self.args.threshold_mode}_recommendation",
            )
        # return None
        return 0.0

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

    def _load_neuron_file(self) -> pd.DataFrame:
        """Load neuron file based on different config."""
        neuron_path = (
            settings.PATH.result_dir
            / "selection"
            / "neuron"
            / self.args.sel_freq
            / self.args.model
            / self.args.heuristic
            / "boost"
            / f"{self.args.data_range_end}_all.csv"
        )
        logger.info(f"Loading the extreme neuron indices from {neuron_path}.")
        return pd.read_csv(neuron_path)

    def _load_extreme_neurons(self, step: str) -> list[int]:
        """Load extreme neurons."""
        # initilize neuron class
        neuron_loader = NeuronLoader(top_n=-1)
        frame = self.neuron_file[self.neuron_file["step"] == int(step)]
        # convert neuron index format
        neuron_value = frame.head(1)["top_neurons"].item()
        special_neuron_indices, _ = neuron_loader.extract_neurons(neuron_value)
        return special_neuron_indices[(len(special_neuron_indices) - 2 * (self.args.top_n)) :]


#######################################################################################################
# Entry point of the script
#######################################################################################################


def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()
    # loop over different steps
    data_path, model_path, eval_path, feather_path, entropy_path = configure_path(args)
    # initilize with the step dir
    step_processor = StepPathProcessor(entropy_path)
    step_dirs = step_processor.sort_paths()
    trainer = Trainer(args, feather_path, entropy_path, data_path, model_path, eval_path, step_dirs)
    trainer.run_pipeline()


if __name__ == "__main__":
    main()
