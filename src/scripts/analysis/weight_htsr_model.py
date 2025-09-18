#!/usr/bin/env python
import argparse
import logging
from pathlib import Path

from transformers import AutoModelForCausalLM

from neuron_analyzer import settings
from neuron_analyzer.analysis.geometry_util import (
    NeuronGroupAnalyzer,
    get_device,
    get_group_name,
    get_last_layer,
    load_activation_indices,
)
from neuron_analyzer.analysis.htsr import WeightSpaceHeavyTailedAnalyzer
from neuron_analyzer.load_util import JsonProcessor

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze geometric features in activation space.")
    parser.add_argument("-m", "--model", type=str, default="gpt2-xl", help="Target model name")
    parser.add_argument("--vector", type=str, default="longtail_50", choices=["mean", "longtail_elbow", "longtail_50"])
    parser.add_argument("--heuristic", type=str, choices=["KL", "prob"], default="prob", help="selection heuristic")
    parser.add_argument(
        "--group_type", type=str, choices=["individual", "group"], default="individual", help="different neuron groups"
    )
    parser.add_argument(
        "--group_size", type=str, choices=["best", "target_size"], default="best", help="different group size"
    )
    parser.add_argument(
        "--sel_freq",
        type=str,
        choices=["longtail_50", "common", None],
        default="longtail_50",
        help="freq by common or not",
    )
    parser.add_argument("--sel_by_med", type=bool, default=False, help="whether to select by mediation effect")
    parser.add_argument("--stat_file", type=str, default="zipf_threshold_stats.json", help="stat filename")
    parser.add_argument("--load_stat", action="store_true", help="Whether to load from existing index")
    parser.add_argument("--exclude_random", action="store_true", help="Whether to exclude existing random")
    parser.add_argument("--debug", action="store_true", help="Compute the first 500 lines if enabled")
    parser.add_argument("--resume", action="store_true", help="Check existing file and resume when setting this")
    parser.add_argument("--top_n", type=int, default=30, help="The top n neurons to be selected")
    parser.add_argument("--max_freq", default=50, help="the proportion of selected max freq")
    parser.add_argument("--min_freq", default=0, help="the proportion of selected min freq")
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

    save_path = settings.PATH.direction_dir / "htsr" / group_name / args.vector / args.model / save_heuristic / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    abl_path = settings.PATH.result_dir / "ablations" / args.vector / args.model
    # only set the path when loading the group neurons
    neuron_dir = settings.PATH.neuron_dir / "group" / args.vector / args.model / args.heuristic
    sel_dir = settings.PATH.neuron_dir / "neuron" / args.vector / args.model / args.heuristic / "boost"
    # threshold_path = settings.PATH.ablation_dir / "longtail_50" / args.model / f"{args.min_freq}_{args.max_freq}.json"

    threshold_path = settings.PATH.freq_dir / args.model / f"{args.min_freq}_{args.max_freq}.json"
    threshold_path.parent.mkdir(parents=True, exist_ok=True)

    return save_path, abl_path, neuron_dir, sel_dir, threshold_path


def load_model(model_dir: Path, device: str):
    """Load model from the given path."""
    model = AutoModelForCausalLM.from_pretrained(str(model_dir)).to(device)
    return model.eval()


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
    save_path, abl_path, neuron_dir, sel_dir, threshold_path = configure_path(args)

    # load and update result json
    # step_processor = StepPathProcessor(abl_path)
    # final_results, step_dirs = step_processor.resume_results(args.resume, save_path, neuron_dir)

    # Initialize extractor
    model_dir = settings.PATH.model_dir / args.model
    # model = load_model(model_dir, device)
    model = load_model(args.model, device)
    neuron_analyzer = NeuronGroupAnalyzer(
        args=args, abl_path=abl_path, step_num="-1", neuron_dir=neuron_dir, device=device
    )

    _, boost_neuron_indices, suppress_neuron_indices, random_indices, _ = load_activation_indices(
        args=args,
        abl_path=abl_path,
        step_num="-1",
        neuron_dir=neuron_dir,
        threshold_path=threshold_path,
        device=device,
    )

    # initilize the class
    geometry_analyzer = WeightSpaceHeavyTailedAnalyzer(
        model=model,
        model_name=args.model,
        boost_neuron_indices=boost_neuron_indices,
        suppress_neuron_indices=suppress_neuron_indices,
        excluded_neuron_indices=random_indices,
        num_random_groups=2,
        layer_num=get_last_layer(sel_dir),
        use_mixed_precision=use_mixed_precision,
        device=device,
    )
    results = geometry_analyzer.run_all_analyses()

    # save the results
    final_results = {}
    final_results["-1"] = results
    JsonProcessor.save_json(final_results, save_path)
    logger.info(f"Save file to {save_path}")


if __name__ == "__main__":
    main()
