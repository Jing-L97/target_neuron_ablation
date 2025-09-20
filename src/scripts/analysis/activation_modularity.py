#!/usr/bin/env python
import argparse
import logging
from pathlib import Path

import yaml

from neuron_analyzer import settings
from neuron_analyzer.analysis.a_modularity import AnalysisConfig, run_all_analyses
from neuron_analyzer.analysis.geometry_util import get_device, get_group_name, load_activation_indices
from neuron_analyzer.load_util import JsonProcessor, StepPathProcessor

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze geometric features in activation space.")
    parser.add_argument("-m", "--model", type=str, default="EleutherAI/pythia-70m-deduped", help="Target model name")
    parser.add_argument(
        "--vector", type=str, default="longtail_50", choices=["mean", "longtail_elbow", "longtail_0_50", "longtail_50"]
    )
    parser.add_argument("--heuristic", type=str, choices=["KL", "prob"], default="prob", help="selection heuristic")
    parser.add_argument(
        "--sel_freq",
        type=str,
        choices=["longtail_50", "common", None],
        default="longtail_50",
        help="freq by common or not",
    )
    parser.add_argument(
        "--group_type", type=str, choices=["individual", "group"], default="individual", help="different neuron groups"
    )
    parser.add_argument(
        "--group_size", type=str, choices=["best", "target_size"], default="best", help="different group size"
    )
    parser.add_argument(
        "--step_mode", type=str, choices=["single", "multi"], default="single", help="whether to compute multi steps"
    )
    parser.add_argument("--sel_longtail", type=bool, default=True, help="whether to filter by longtail token")
    parser.add_argument("--sel_by_med", type=bool, default=False, help="whether to select by mediation effect")
    parser.add_argument("--load_stat", action="store_true", help="Whether to load from existing index")
    parser.add_argument("--exclude_random", action="store_true", help="Whether to exclude existing random")
    parser.add_argument("--debug", action="store_true", help="Compute the first 500 lines if enabled")
    parser.add_argument("--resume", action="store_true", help="Check existing file and resume when setting this")
    parser.add_argument("--top_n", type=int, default=50, help="The top n neurons to be selected")
    parser.add_argument("--max_freq", default=15, help="the proportion of selected max freq")
    parser.add_argument("--min_freq", default=0, help="the proportion of selected min freq")
    parser.add_argument("--tokenizer_name", type=str, default="EleutherAI/pythia-410m", help="Unigram tokenizer name")
    parser.add_argument("--data_range_end", type=int, default=500, help="the selected datarange")
    parser.add_argument("--k", type=int, default=10, help="use_bos_only if enabled")

    # New argument for config file path
    parser.add_argument(
        "--config_file", type=str, default=None, help="Path to YAML config file for AnalysisConfig parameters"
    )

    return parser.parse_args()


def load_analysis_config_from_yaml(config_file_path: str) -> dict:
    """Load AnalysisConfig parameters from YAML file."""
    if not config_file_path or not Path(config_file_path).exists():
        logger.warning(f"Config file {config_file_path} not found, using default AnalysisConfig")
        return {}

    with open(config_file_path) as file:
        config_data = yaml.safe_load(file)
        logger.info(f"Loaded AnalysisConfig parameters from {config_file_path}")
        return config_data.get("analysis_config", {})


#######################################################################################################
# Functions applied in the main scripts
#######################################################################################################


def configure_path(args, config_params):
    """Configure save path based on the setting."""
    save_heuristic = f"{args.heuristic}_med" if args.sel_by_med else args.heuristic
    filename_suffix = ".debug" if args.debug else ".json"
    group_name = get_group_name(args)
    stat_filename = (
        f"{args.data_range_end}_{args.top_n}_check_random{filename_suffix}"
        if args.exclude_random
        else f"{args.data_range_end}_{args.top_n}{filename_suffix}"
    )
    threshold_filename = (
        f"{args.data_range_end}_{args.top_n}_check_random{filename_suffix}"
        if args.exclude_random
        else f"{args.data_range_end}_{args.top_n}_threshold{filename_suffix}"
    )

    # Use config parameters for edge configuration
    edge_type = config_params.get("edge_construction_method", "mi")
    alg = config_params.get("algorithm", "spectral")

    # Determine threshold based on edge type
    if edge_type == "correlation":
        edge_threshold = config_params.get("correlation_threshold", float("-inf"))
    elif edge_type == "mi":
        edge_threshold = config_params.get("mi_threshold", float("-inf"))
    else:
        edge_threshold = float("-inf")  # default

    logger.info(f"The threshold is {edge_threshold}")

    save_path = (
        settings.PATH.direction_dir
        / group_name
        / "modularity"
        / args.vector
        / args.model
        / save_heuristic
        / f"{edge_type}_{alg}"
    )
    save_path.mkdir(parents=True, exist_ok=True)

    save_stat_path = save_path / stat_filename
    save_threshold_path = save_path / threshold_filename

    # if args.sel_freq == "common":  # select from all of the tokens
    #     threshold_path = settings.PATH.ablation_dir / "longtail_50" / args.model
    #     abl_path = settings.PATH.ablation_dir / "mean" / args.model
    # else:
    #     threshold_path = None
    #     abl_path = settings.PATH.ablation_dir / args.vector / args.model

    abl_path = settings.PATH.ablation_dir / args.vector / args.model
    threshold_path = settings.PATH.freq_dir / args.model / f"{args.min_freq}_{args.max_freq}.json"
    threshold_path.parent.mkdir(parents=True, exist_ok=True)

    # only set the path when loading the group neurons
    neuron_dir = settings.PATH.neuron_dir / "group" / args.vector / args.model / args.heuristic
    return save_stat_path, save_threshold_path, abl_path, neuron_dir, threshold_path


def create_analysis_config(config_params: dict) -> AnalysisConfig:
    """Create AnalysisConfig instance with parameters from config file."""
    analysis_config = AnalysisConfig()

    # Override with parameters from config file
    for key, value in config_params.items():
        if hasattr(analysis_config, key):
            setattr(analysis_config, key, value)
            logger.info(f"Set AnalysisConfig.{key} = {value}")
        else:
            logger.warning(f"Unknown AnalysisConfig parameter: {key}")

    return analysis_config


def analyze_single(
    args,
    abl_path: Path,
    save_stat_path: Path,
    neuron_dir: Path,
    threshold_path: Path,
    device: str,
    analysis_config: AnalysisConfig,
    **kwargs,
) -> None:
    """Analyze the activation space of the single step."""
    activation_data, boost_neuron_indices, suppress_neuron_indices, _, do_analysis = load_activation_indices(
        args=args,
        abl_path=abl_path,
        step_num="-1",
        neuron_dir=neuron_dir,
        threshold_path=threshold_path,
        device=device,
    )

    if do_analysis:
        # Override with any additional provided kwargs
        for key, value in kwargs.items():
            if hasattr(analysis_config, key):
                setattr(analysis_config, key, value)

        # Run analysis
        results = run_all_analyses(
            activation_data=activation_data,
            boost_neuron_indices=boost_neuron_indices,
            suppress_neuron_indices=suppress_neuron_indices,
            config=analysis_config,
        )

        final_results = {}
        final_results[str(-1)] = results
        # assign col headers
        JsonProcessor.save_json(final_results, save_stat_path)
        logger.info(f"Save stat file to {save_stat_path}")


def analyze_multi(
    args,
    abl_path: Path,
    save_stat_path: Path,
    neuron_dir: Path,
    threshold_path: Path,
    device: str,
    analysis_config: AnalysisConfig,
    **kwargs,
) -> None:
    """Analyze the activation space of the multi step."""
    step_processor = StepPathProcessor(abl_path)
    final_results, step_dirs = step_processor.resume_results(args.resume, save_stat_path, neuron_dir)

    for step in step_dirs:
        # try:
        activation_data, boost_neuron_indices, suppress_neuron_indices, random_indices, do_analysis = (
            load_activation_indices(args, abl_path, str(step[1]), neuron_dir, threshold_path, device)
        )
        logger.info(f"Finished loading boost neuron indices {boost_neuron_indices}")
        logger.info(f"Finished loading suppress neuron indices {suppress_neuron_indices}")

        if do_analysis:
            # Override with any additional provided kwargs
            for key, value in kwargs.items():
                if hasattr(analysis_config, key):
                    setattr(analysis_config, key, value)

            # Run analysis
            results = run_all_analyses(
                activation_data=activation_data,
                boost_neuron_indices=boost_neuron_indices,
                suppress_neuron_indices=suppress_neuron_indices,
                config=analysis_config,
            )

            final_results = {}
            final_results[str(-1)] = results
            # assign col headers
            JsonProcessor.save_json(final_results, save_stat_path)
            logger.info(f"Save stat file to {save_stat_path}")


#######################################################################################################
# Entry point of the script
#######################################################################################################


def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()
    device, use_mixed_precision = get_device()

    if args.exclude_random:
        logger.info("Exclude the existing random neurons")

    # Load AnalysisConfig parameters from YAML file
    config_params = load_analysis_config_from_yaml(args.config_file)
    analysis_config = create_analysis_config(config_params)

    # Configure paths
    save_stat_path, save_threshold_path, abl_path, neuron_dir, threshold_path = configure_path(args, config_params)

    if args.step_mode == "single":
        analyze_single(
            args=args,
            abl_path=abl_path,
            save_stat_path=save_stat_path,
            neuron_dir=neuron_dir,
            threshold_path=threshold_path,
            device=device,
            analysis_config=analysis_config,
        )
    elif args.step_mode == "multi":
        analyze_multi(
            args=args,
            abl_path=abl_path,
            save_stat_path=save_stat_path,
            neuron_dir=neuron_dir,
            threshold_path=threshold_path,
            device=device,
            analysis_config=analysis_config,
        )


if __name__ == "__main__":
    main()
