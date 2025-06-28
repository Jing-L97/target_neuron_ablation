#!/usr/bin/env python
import argparse
import logging
from pathlib import Path

from neuron_analyzer import settings
from neuron_analyzer.analysis.a_modularity import AnalysisConfig, run_all_analyses
from neuron_analyzer.analysis.geometry_util import get_device, get_group_name, load_activation_indices
from neuron_analyzer.load_util import JsonProcessor

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze geometric features in activation space.")
    parser.add_argument("-m", "--model", type=str, default="EleutherAI/pythia-70m-deduped", help="Target model name")
    parser.add_argument("--vector", type=str, default="longtail_50", choices=["mean", "longtail_elbow", "longtail_50"])
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
    parser.add_argument("--edge_type", type=str, choices=["correlation", "mi", "hybrid"], default="correlation")
    parser.add_argument("--edge_threshold", type=float, default=0.1)
    parser.add_argument("--apply_abs", action="store_true", help="whether to use absoluta value threshold")
    parser.add_argument("--sel_longtail", type=bool, default=True, help="whether to filter by longtail token")
    parser.add_argument("--sel_by_med", type=bool, default=False, help="whether to select by mediation effect")
    parser.add_argument("--load_stat", action="store_true", help="Whether to load from existing index")
    parser.add_argument("--exclude_random", action="store_true", help="Whether to exclude existing random")
    parser.add_argument("--debug", action="store_true", help="Compute the first 500 lines if enabled")
    parser.add_argument("--resume", action="store_true", help="Check existing file and resume when setting this")
    parser.add_argument("--top_n", type=int, default=50, help="The top n neurons to be selected")
    parser.add_argument("--stat_file", type=str, default="zipf_threshold_stats.json", help="stat filename")
    parser.add_argument("--tokenizer_name", type=str, default="EleutherAI/pythia-410m", help="Unigram tokenizer name")
    parser.add_argument("--data_range_end", type=int, default=500, help="the selected datarange")
    parser.add_argument("--k", type=int, default=10, help="use_bos_only if enabled")
    return parser.parse_args()


#######################################################################################################
# Fucntions applied in the main scripts
#######################################################################################################


def configure_path(args):
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
    # TODO: we revise this part for baseline experiment
    if args.apply_abs:
        edge_dir = f"{args.edge_type}_{args.edge_threshold}_abs"
    else:
        edge_dir = f"{args.edge_type}_{args.edge_threshold}"
    save_path = (
        settings.PATH.direction_dir / group_name / "modularity" / args.vector / args.model / save_heuristic / edge_dir
    )
    save_path.mkdir(parents=True, exist_ok=True)

    save_stat_path = save_path / stat_filename
    save_threshold_path = save_path / threshold_filename

    if args.sel_freq == "common":  # select from all of the tokens
        threshold_path = settings.PATH.ablation_dir / "longtail_50" / args.model
        abl_path = settings.PATH.ablation_dir / "mean" / args.model
    else:
        threshold_path = None
        abl_path = settings.PATH.ablation_dir / args.vector / args.model
    # only set the path when loading the group neurons
    neuron_dir = settings.PATH.neuron_dir / "group" / args.vector / args.model / args.heuristic
    return save_stat_path, save_threshold_path, abl_path, neuron_dir, threshold_path


def overdrive_class(args) -> AnalysisConfig:
    """Overdrive the class with new parameters."""
    if args.edge_type == "correlation":
        return AnalysisConfig(
            edge_construction_method=args.edge_type, correlation_threshold=args.edge_threshold, apply_abs=args.apply_abs
        )
    if args.edge_type == "mi":
        return AnalysisConfig(
            edge_construction_method=args.edge_type, mi_threshold=args.edge_threshold, apply_abs=args.apply_abs
        )
    return AnalysisConfig


def analyze_single(
    args,
    abl_path: Path,
    save_stat_path: Path,
    neuron_dir: Path,
    threshold_path: Path,
    device: str,
    **kwargs,
) -> None:
    """Analze the activation space of the single step."""
    activation_data, boost_neuron_indices, suppress_neuron_indices, random_indices, do_analysis = (
        load_activation_indices(
            args=args,
            abl_path=abl_path,
            step_num="-1",
            neuron_dir=neuron_dir,
            threshold_path=threshold_path,
            device=device,
        )
    )

    analysis_config = overdrive_class(args)

    if do_analysis:
        # initilize the class
        # Override with any provided kwargs
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
    **kwargs,
) -> None:
    """Analze the activation space of the single step."""
    activation_data, boost_neuron_indices, suppress_neuron_indices, random_indices, do_analysis = (
        load_activation_indices(
            args=args,
            abl_path=abl_path,
            step_num="-1",
            neuron_dir=neuron_dir,
            threshold_path=threshold_path,
            device=device,
        )
    )

    analysis_config = overdrive_class(args)

    if do_analysis:
        # initilize the class
        # Override with any provided kwargs
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
    # loop over different steps
    save_stat_path, save_threshold_path, abl_path, neuron_dir, threshold_path = configure_path(args)
    analysis_config = AnalysisConfig()

    if args.step_mode == "single":
        analyze_single(
            args=args,
            abl_path=abl_path,
            save_stat_path=save_stat_path,
            save_threshold_path=save_threshold_path,
            neuron_dir=neuron_dir,
            threshold_path=threshold_path,
            device=device,
            analysis_config=analysis_config,
        )
    if args.step_mode == "multi":
        analyze_multi(
            args=args,
            abl_path=abl_path,
            save_stat_path=save_stat_path,
            save_threshold_path=save_threshold_path,
            neuron_dir=neuron_dir,
            threshold_path=threshold_path,
            device=device,
            use_mixed_precision=use_mixed_precision,
        )


if __name__ == "__main__":
    main()
