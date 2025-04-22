#!/usr/bin/env python
import argparse
import logging
from pathlib import Path

import torch

from neuron_analyzer import settings
from neuron_analyzer.analysis.a_geometry import ActivationGeometricAnalyzer
from neuron_analyzer.load_util import JsonProcessor
from neuron_analyzer.model_util import NeuronLoader
from neuron_analyzer.selection.neuron import NeuronSelector

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze geometric features in activation space.")
    parser.add_argument("-m", "--model", type=str, default="EleutherAI/pythia-70m-deduped", help="Target model name")
    parser.add_argument("--vector", type=str, default="longtail_50", choices=["mean", "longtail_elbow", "longtail_50"])
    parser.add_argument("--heuristic", type=str, choices=["KL", "prob"], default="prob", help="selection heuristic")
    parser.add_argument("--sel_longtail", type=bool, default=True, help="whether to filter by longtail token")
    parser.add_argument("--sel_by_med", type=bool, default=False, help="whether to select by mediation effect")
    parser.add_argument("--debug", action="store_true", help="Compute the first 500 lines if enabled")
    parser.add_argument("--resume", action="store_true", help="Check existing file and resume when setting this")
    parser.add_argument("--top_n", type=int, default=10, help="The top n neurons to be selected")
    parser.add_argument("--stat_file", type=str, default="zipf_threshold_stats.json", help="stat filename")
    parser.add_argument("--tokenizer_name", type=str, default="EleutherAI/pythia-410m", help="Unigram tokenizer name")
    parser.add_argument("--data_range_end", type=int, default=500, help="the selected datarange")
    parser.add_argument("--k", type=int, default=10, help="use_bos_only if enabled")
    return parser.parse_args()


#######################################################################################################
# Fucntions applied in the main scripts
#######################################################################################################


class NeuronGroupAnalyzer:
    def __init__(self, args, device: str, feather_path: Path, step: Path, abl_path: Path):
        self.args = args
        self.device = device
        self.feather_path = feather_path
        self.step = step
        self.abl_path = abl_path

    def get_neuron_group(self):
        """Filter neuron index for different interventions and return the grouped data."""
        # adjust the debug
        # load selector
        neuron_selector = NeuronSelector(
            feather_path=self.feather_path,
            debug=self.args.debug,
            top_n=self.args.top_n,
            step=self.step.name,
            tokenizer_name=self.args.tokenizer_name,
            threshold_path=Path(self.abl_path) / self.args.stat_file,
            sel_longtail=self.args.sel_longtail,
            device=self.device,
            sel_by_med=self.args.sel_by_med,
        )
        # load feather dataframe
        activation_data = neuron_selector.load_and_filter_df()
        final_df = neuron_selector._prepare_dataframe(activation_data)
        # select neurons
        boost_neuron_indices, _ = self.get_neuron_index(neuron_selector, final_df, "boost")
        suppress_neuron_indices, neuron_loader = self.get_neuron_index(neuron_selector, final_df, "suppress")
        # convert the component name into target format
        activation_data["component_name"] = activation_data["component_name"].apply(neuron_loader.parse_neurons)
        return activation_data, boost_neuron_indices, suppress_neuron_indices

    def get_neuron_index(self, neuron_selector, activation_data, effect: str):
        """Filter neuron index for different interventions."""
        # select neurons
        if self.args.heuristic == "KL":
            frame = neuron_selector.select_by_KL(effect=effect, final_df=activation_data)
        elif self.args.heuristic == "prob":
            frame = neuron_selector.select_by_prob(effect=effect, final_df=activation_data)
        else:
            raise ValueError(f"Unknown heuristic: {self.args.heuristic}")

        # convert neuron index format
        neuron_loader = NeuronLoader(top_n=self.args.top_n)
        neuron_value = frame.head(1)["top_neurons"].item()
        logger.info(f"The filtered neuron values are: {neuron_value}")
        special_neuron_indices, _ = neuron_loader.extract_neurons(neuron_value)
        return special_neuron_indices, neuron_loader


def configure_save_path(args):
    """Configure save path based on the setting."""
    save_heuristic = f"{args.heuristic}_med" if args.sel_by_med else args.heuristic
    filename_suffix = ".debug" if args.debug else ".json"
    save_path = (
        settings.PATH.result_dir
        / "geometry"
        / "activation"
        / args.vector
        / args.model
        / save_heuristic
        / f"{args.data_range_end}_{args.top_n}{filename_suffix}"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    return save_path


def get_device():
    if torch.cuda.is_available():
        return "cuda", True
    return "cpu", False


#######################################################################################################
# Entry point of the script
#######################################################################################################


def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()
    device, use_mixed_precision = get_device()

    # loop over different steps
    abl_path = settings.PATH.result_dir / "ablations" / args.vector / args.model
    save_path = configure_save_path(args)
    if save_path.is_file() and args.resume:
        # load and update result json
        final_results = 
        logger.info(f"{save_path} already exists, skip!")
    else:
        final_results = {}
        for step in abl_path.iterdir():
            feather_path = abl_path / str(step) / str(args.data_range_end) / f"k{args.k}.feather"
            if feather_path.is_file():
                group_analyzer = NeuronGroupAnalyzer(
                    args=args, feather_path=feather_path, step=step, abl_path=abl_path, device=device
                )
                activation_data, boost_neuron_indices, suppress_neuron_indices = group_analyzer.get_neuron_group()
                # initilize the class
                geometry_analyzer = ActivationGeometricAnalyzer(
                    activation_data=activation_data,
                    boost_neuron_indices=boost_neuron_indices,
                    suppress_neuron_indices=suppress_neuron_indices,
                    activation_column="activation",
                    token_column="str_tokens",
                    context_column="context",
                    component_column="component_name",
                    num_random_groups=2,
                    device=device,
                    use_mixed_precision=use_mixed_precision,
                )
                results = geometry_analyzer.run_all_analyses()
                final_results[step.name] = results
                # assign col headers
                JsonProcessor.save_json(final_results, save_path)
                logger.info(f"Save file to {save_path}")


if __name__ == "__main__":
    main()
