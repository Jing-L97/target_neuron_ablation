#!/usr/bin/env python
import argparse
import logging
from pathlib import Path

from neuron_analyzer import settings
from neuron_analyzer.analysis.a_MI import MutualInformationAnalyzer
from neuron_analyzer.analysis.geometry_util import get_device, get_group_name, load_activation_indices
from neuron_analyzer.load_util import JsonProcessor, StepPathProcessor

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
    parser.add_argument("--sel_longtail", type=bool, default=True, help="whether to filter by longtail token")
    parser.add_argument("--sel_by_med", type=bool, default=False, help="whether to select by mediation effect")
    parser.add_argument("--load_stat", action="store_true", help="Whether to load from existing index")
    parser.add_argument("--exclude_random", action="store_true", help="Whether to exclude existing random")
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


def configure_path(args):
    """Configure save path based on the setting."""
    save_heuristic = f"{args.heuristic}_med" if args.sel_by_med else args.heuristic
    filename_suffix = ".debug" if args.debug else ".json"
    group_name = get_group_name(args)
    filename = (
        f"{args.data_range_end}_{args.top_n}_check_random{filename_suffix}"
        if args.exclude_random
        else f"{args.data_range_end}_{args.top_n}{filename_suffix}"
    )
    # TODO: we revise this part for baseline experiment

    save_path = settings.PATH.direction_dir / group_name / "MI" / args.vector / args.model / save_heuristic / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if args.sel_freq == "common":  # select from all of the tokens
        threshold_path = settings.PATH.ablation_dir / "longtail_50" / args.model
        abl_path = settings.PATH.ablation_dir / "mean" / args.model
    else:
        threshold_path = None
        abl_path = settings.PATH.ablation_dir / args.vector / args.model
    # only set the path when loading the group neurons
    neuron_dir = settings.PATH.neuron_dir / "group" / args.vector / args.model / args.heuristic
    return save_path, abl_path, neuron_dir, threshold_path


def analyze_single(
    args,
    abl_path: Path,
    save_path: Path,
    neuron_dir: Path,
    threshold_path: Path,
    device: str,
    use_mixed_precision: bool,
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

    if do_analysis:
        # initilize the class
        geometry_analyzer = MutualInformationAnalyzer(
            activation_data=activation_data,
            boost_neuron_indices=boost_neuron_indices,
            suppress_neuron_indices=suppress_neuron_indices,
            excluded_neuron_indices=random_indices,
            activation_column="activation",
            token_column="str_tokens",
            context_column="context",
            component_column="component_name",
            num_random_groups=2,
            device=device,
            use_mixed_precision=use_mixed_precision,
        )
        results = geometry_analyzer.run_all_analyses()
        final_results = {}
        final_results[str(-1)] = results
        # assign col headers
        JsonProcessor.save_json(final_results, save_path)
        logger.info(f"Save file to {save_path}")


def analyze_multi(
    args,
    abl_path: Path,
    save_path: Path,
    neuron_dir: Path,
    threshold_path: Path,
    device: str,
    use_mixed_precision: bool,
) -> None:
    """Analze the activation space of multiple steps."""
    # load and update result json
    step_processor = StepPathProcessor(abl_path)
    final_results, step_dirs = step_processor.resume_results(args.resume, save_path, neuron_dir)

    for step in step_dirs:
        # try:
        activation_data, boost_neuron_indices, suppress_neuron_indices, random_indices, do_analysis = (
            load_activation_indices(args, abl_path, str(step[1]), neuron_dir, threshold_path, device)
        )
        if do_analysis:
            # initilize the class
            geometry_analyzer = MutualInformationAnalyzer(
                activation_data=activation_data,
                boost_neuron_indices=boost_neuron_indices,
                suppress_neuron_indices=suppress_neuron_indices,
                excluded_neuron_indices=random_indices,
                activation_column="activation",
                token_column="str_tokens",
                context_column="context",
                component_column="component_name",
                num_random_groups=2,
                device=device,
                use_mixed_precision=use_mixed_precision,
            )
            results = geometry_analyzer.run_all_analyses()
            final_results[str(step[1])] = results
            # assign col headers
            JsonProcessor.save_json(final_results, save_path)
            logger.info(f"Save file to {save_path}")

    # except:
    #     logger.info(f"Something wrong with {step[1]}")


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
    save_path, abl_path, neuron_dir, threshold_path = configure_path(args)

    if args.step_mode == "single":
        analyze_single(
            args=args,
            abl_path=abl_path,
            save_path=save_path,
            neuron_dir=neuron_dir,
            threshold_path=threshold_path,
            device=device,
            use_mixed_precision=use_mixed_precision,
        )
    if args.step_mode == "multi":
        analyze_multi(
            args=args,
            abl_path=abl_path,
            save_path=save_path,
            neuron_dir=neuron_dir,
            threshold_path=threshold_path,
            device=device,
            use_mixed_precision=use_mixed_precision,
        )


if __name__ == "__main__":
    main()
