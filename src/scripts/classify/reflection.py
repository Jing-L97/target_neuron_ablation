#!/usr/bin/env python
import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from neuron_analyzer import settings
from neuron_analyzer.analysis.geometry_util import get_group_name, get_last_layer
from neuron_analyzer.classify.reflection import SVMHyperplaneReflector
from neuron_analyzer.eval.surprisal import StepSurprisalExtractor
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
        "--classifier_type",
        type=str,
        default="svm_linear",
        choices=["svm_linear", "linear_svc"],
        help="selected classifier type",
    )
    parser.add_argument(
        "--class_num",
        type=int,
        default=2,
        help="how many classes to classify",
    )
    parser.add_argument("--load_stat", type=bool, default=True, help="Whether to load from existing index")
    parser.add_argument("--exclude_random", type=bool, default=True, help="Include all neuron indices if set True")
    parser.add_argument("--run_baseline", action="store_true", help="Whether to run baseline models")
    parser.add_argument("--sel_by_med", type=bool, default=False, help="whether to select by mediation effect")
    parser.add_argument("--top_n", type=int, default=10, help="The top n neurons to be selected")
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


class ReflectionAnalyzer:
    """Class for running the entire neuron classification pipeline."""

    def __init__(
        self,
        args: Any,
        device: str,
        data_path: Path,
        model_path: Path,
        eval_path: Path,
        step_dirs: list[tuple[str, str]],
    ):
        """Initialize the pipeline with all necessary parameters."""
        self.args = args
        self.device = device
        self.data_path = data_path
        self.model_path = model_path
        self.eval_path = eval_path
        self.step_dirs = step_dirs
        self.layer_num = get_last_layer(self.args.model)

    def run_pipeline(self) -> dict[str, Any]:
        """Extract optimal threshold across multiple steps and run classification."""
        for step in self.step_dirs:
            # try:
            # configure the save path
            step_data_path, step_model_path, step_eval_path = self._configure_save_path(step)
            # Load data
            neurons, activation_lst, input_string_lst, target_string_lst = self._load_data(step_data_path)

            # load model and tokenize target strings
            model, tokenizer = self._load_model_tokenizer(step[1])
            input_token_ids_list, target_token_ids_list = self._tokenize_strings(
                tokenizer, input_string_lst, target_string_lst
            )

            # intialize the analyzer
            reflector = SVMHyperplaneReflector(
                svm_checkpoint_path=step_model_path / f"{self.args.classifier_type}.joblib",
                model_name=self.args.model,
                device=self.device,
                model=model,
                tokenizer=tokenizer,
                layer_num=self.layer_num,
                save_path=step_eval_path / "reflection.json",
                step_num=step[1],
            )

            activations = [sub[1] for sub in activation_lst]

            reflector.run_reflection_analysis(
                tokenized_input=input_token_ids_list[1],
                target_token_ids=target_token_ids_list[1],
                neurons=neurons,
                activations=activations,
            )

        # except Exception as e:
        # logger.info(f"Error processing step {step[1]}: {e!s}")

    def _load_data(self, step_data_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Load and prepare data for reflection test."""
        # Load neuron indices and activations
        data = JsonProcessor.load_json(step_data_path / f"data_{self.args.top_n}.json")
        neurons, activation_lst = data["neuron_indices"], data["X"]
        # load input strings
        df = pd.read_feather(step_data_path / "entropy_df.feather")
        df = df.head(self.args.top_n)
        return neurons, activation_lst, df["context"].to_list(), df["str_tokens"].to_list()

    def _load_model_tokenizer(self, step: int):
        """Process a single step with the given configuration."""
        # initlize the model handler class
        model_cache_dir = settings.PATH.model_dir / self.args.model
        extractor = StepSurprisalExtractor(
            config=[],
            model_name=self.args.model,
            model_cache_dir=model_cache_dir,  # note here we use the relative path
            layer_num=self.layer_num,
            device=self.device,
        )
        return extractor.load_model_for_step(step)

    def _tokenize_strings(self, tokenizer, input_string_lst: list[str], target_string_lst: list[str]):
        """Tokenize the target strings and token."""
        input_token_ids_list = [tokenizer.encode(text, add_special_tokens=False) for text in input_string_lst]
        target_token_ids_list = [tokenizer.encode(text, add_special_tokens=False) for text in target_string_lst]
        return input_token_ids_list, target_token_ids_list

    def _configure_save_path(self, step: tuple[str, str]) -> tuple[Path, Path, Path]:
        """Configure dave path based on different conditions."""
        group_name = get_group_name(self.args)
        suffix_path = (
            Path(str(step[1]))
            / str(self.args.data_range_end)
            / str(self.args.top_n)
            / str(self.args.class_num)
            / group_name
        )
        step_model_path = self.model_path / suffix_path
        step_eval_path = self.eval_path / suffix_path
        step_data_path = self.data_path / str(step[1]) / str(self.args.data_range_end)
        return step_data_path, step_model_path, step_eval_path


#######################################################################################################
# Entry point of the script
#######################################################################################################


def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # loop over different steps
    data_path, model_path, eval_path = configure_path(args)
    # initilize with the step dir
    step_processor = StepPathProcessor(data_path)
    step_dirs = step_processor.sort_paths()
    data_path, model_path, eval_path = configure_path(args)
    trainer = ReflectionAnalyzer(
        args=args, device=device, data_path=data_path, model_path=model_path, eval_path=eval_path, step_dirs=step_dirs
    )
    trainer.run_pipeline()


if __name__ == "__main__":
    main()
