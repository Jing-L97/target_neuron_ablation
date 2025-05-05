#!/usr/bin/env python
import logging
from pathlib import Path

import pandas as pd
import torch

from neuron_analyzer import settings
from neuron_analyzer.load_util import JsonProcessor
from neuron_analyzer.model_util import NeuronLoader
from neuron_analyzer.selection.neuron import NeuronSelector

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

#######################################################
# Load last layer


def get_last_layer(model_name) -> int:
    """Get the last layer of the model name."""
    return 5 if "70" in model_name else 23


#######################################################
# Neuron group Selector class


class NeuronGroupAnalyzer:
    def __init__(
        self,
        args,
        device: str,
        unigram_analyzer=None,
        feather_path: Path = None,
        step_path: Path = None,
        abl_path: Path = None,
        neuron_dir: Path = None,
    ):
        self.args = args
        self.device = device
        self.unigram_analyzer = unigram_analyzer
        self.feather_path = feather_path
        self.step = step_path
        self.abl_path = abl_path
        self.neuron_dir = neuron_dir  # optional: only apply this when loading group neurons

    def run_pipeline(self) -> tuple[pd.DataFrame, list[int], list[int]]:
        """Run pipeline of the neuron selection."""
        # get activation data
        activation_data = self.load_activation_df()
        # get different neuron groups
        boost_neuron_indices, suppress_neuron_indices, random_neuron_indices = self.load_neurons(activation_data)
        if self.args.exclude_random:
            return activation_data, boost_neuron_indices, suppress_neuron_indices, random_neuron_indices
        return activation_data, boost_neuron_indices, suppress_neuron_indices, []

    def load_neurons(self, activation_data=None) -> tuple[list[int], list[int]]:
        """Load neurons in different conditions."""
        if self.args.load_stat:
            boost_neuron_indices, suppress_neuron_indices, random_neuron_indices = self.load_neuron_from_stat()
            logger.info(f"Loading the {self.args.group_type} neurons from stat")
            if self.args.exclude_random:
                logger.info("Including random neurons")
                return boost_neuron_indices, suppress_neuron_indices, random_neuron_indices
        else:
            if self.args.group_type == "group":
                boost_neuron_indices, suppress_neuron_indices = self.load_group_neuron()
                logger.info("Selecting from the group neurons")
            if self.args.group_type == "individual":
                boost_neuron_indices, suppress_neuron_indices = self.load_individual_neuron(activation_data)
                logger.info("Selecting from the individual neurons")
        return boost_neuron_indices, suppress_neuron_indices, []

    def load_activation_df(self) -> pd.DataFrame:
        """Filter neuron index for different interventions and return the grouped data."""
        # load selector
        self.neuron_selector = NeuronSelector(
            feather_path=self.feather_path,
            debug=self.args.debug,
            top_n=self.args.top_n,
            step=self.step.name,
            unigram_analyzer=self.unigram_analyzer,
            threshold_path=Path(self.abl_path) / self.args.stat_file,
            sel_by_med=self.args.sel_by_med,
            sel_freq=self.args.sel_freq,
        )
        # load feather dataframe
        activation_data = self.neuron_selector.load_and_filter_df()
        # intilize neuron loader
        self.neuron_loader = NeuronLoader(top_n=self.args.top_n)
        # convert the component name into target format
        activation_data["component_name"] = activation_data["component_name"].apply(self.neuron_loader.parse_neurons)
        return activation_data

    def load_neuron_from_stat(self) -> tuple[list[int], list[int]]:
        """Load neuron from existing activation stats."""
        neuron_dict = self._get_stat_file()
        if self.args.exclude_random:
            return (
                neuron_dict[self.step.name]["neuron_indices"]["boost"],
                neuron_dict[self.step.name]["neuron_indices"]["suppress"],
                neuron_dict[self.step.name]["neuron_indices"]["random_1"]
                + neuron_dict[self.step.name]["neuron_indices"]["random_2"],
            )
        return (
            neuron_dict[self.step.name]["neuron_indices"]["boost"],
            neuron_dict[self.step.name]["neuron_indices"]["suppress"],
            [],
        )

    def load_group_neuron(self) -> tuple[list[int], list[int]]:
        """Load selected group neurons."""
        boost_neuron_indices = self._get_group_neuron_index(self.neuron_dir, "boost")
        suppress_neuron_indices = self._get_group_neuron_index(self.neuron_dir, "suppress")
        return boost_neuron_indices, suppress_neuron_indices

    def load_individual_neuron(self, activation_data: pd.DataFrame) -> tuple[list[int], list[int]]:
        """Filter neuron index for different interventions and return the grouped data."""
        final_df = self.neuron_selector._prepare_dataframe(activation_data)
        # select neurons
        boost_neuron_indices, _ = self._get_individual_neuron_index(final_df, "boost")
        suppress_neuron_indices, _ = self._get_individual_neuron_index(final_df, "suppress")
        return boost_neuron_indices, suppress_neuron_indices

    def _get_stat_file(self):
        """Load exsiting stat file."""
        group_name, save_heuristic = get_dir_name(self.args)
        stat_path = (
            settings.PATH.direction_dir
            / group_name
            / "activation"
            / self.args.vector
            / self.args.model
            / save_heuristic
            / f"{self.args.data_range_end}_{self.args.top_n}.json"
        )
        neuron_dict = JsonProcessor.load_json(stat_path)
        return neuron_dict

    def _get_group_neuron_index(self, neuron_dir: Path, effect: str):
        """Filter neuron index for different interventions."""
        neuron_dict = JsonProcessor.load_json(
            neuron_dir / effect / f"{self.args.data_range_end}_{self.args.top_n}.json"
        )
        return neuron_dict[self.step.name][self.args.group_size]["neurons"]

    def _get_individual_neuron_index(self, activation_data, effect: str):
        """Filter neuron index for different interventions."""
        # select neurons
        if self.args.heuristic == "KL":
            frame = self.neuron_selector.select_by_KL(effect=effect, final_df=activation_data)
        elif self.args.heuristic == "prob":
            frame = self.neuron_selector.select_by_prob(effect=effect, final_df=activation_data)
        else:
            raise ValueError(f"Unknown heuristic: {self.args.heuristic}")

        # convert neuron index format
        neuron_value = frame.head(1)["top_neurons"].item()
        logger.info(f"The filtered neuron values are: {neuron_value}")
        special_neuron_indices, _ = self.neuron_loader.extract_neurons(neuron_value)
        return special_neuron_indices


#######################################################
# Function to integrate loading


def load_activation_indices(args, abl_path: Path, step_path: Path, step_num: str, neuron_dir: Path, device: str):
    """Initialize NeuronGroupAnalyzer class."""
    feather_path = abl_path / step_num / str(args.data_range_end) / f"k{args.k}.feather"
    if feather_path.is_file():
        group_analyzer = NeuronGroupAnalyzer(
            args=args,
            feather_path=feather_path,
            step_path=step_path,
            abl_path=abl_path,
            neuron_dir=neuron_dir,
            device=device,
        )
        activation_data, boost_neuron_indices, suppress_neuron_indices, random_indices = group_analyzer.run_pipeline()
        return activation_data, boost_neuron_indices, suppress_neuron_indices, random_indices, True
    return None, None, None, None, False


#######################################################
# Path helper


def get_dir_name(args) -> tuple[Path, Path]:
    """Get group name folder in different conditions."""
    return (
        f"{args.group_type}_{args.group_size}" if args.group_type == "group" else args.group_type,
        f"{args.heuristic}_med" if args.sel_by_med else args.heuristic,
    )


def get_group_name(args) -> Path:
    """Get group name in different conditions."""
    return f"{args.group_type}_{args.group_size}" if args.group_type == "group" else args.group_type


#######################################################
# Device manager


def get_device():
    if torch.cuda.is_available():
        return "cuda", True
    return "cpu", False
