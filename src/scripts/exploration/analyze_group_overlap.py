#!/usr/bin/env python
import argparse
import logging

from neuron_analyzer import settings
from neuron_analyzer.load_util import JsonProcessor

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze neuron index overlapping.")
    parser.add_argument("-m", "--model", type=str, default="EleutherAI/pythia-70m-deduped", help="Target model name")
    parser.add_argument("--vector", type=str, default="longtail_50", choices=["mean", "longtail_elbow", "longtail_50"])
    parser.add_argument("--heuristic", type=str, choices=["KL", "prob"], default="prob", help="selection heuristic")
    parser.add_argument("--top_n", type=int, default=10, help="The top n neurons to be selected")
    parser.add_argument("--stat_file", type=str, default="zipf_threshold_stats.json", help="stat filename")
    parser.add_argument("--data_range_end", type=int, default=500, help="the selected datarange")
    parser.add_argument("--k", type=int, default=10, help="use_bos_only if enabled")
    return parser.parse_args()


#######################################################################################################
# Fucntions applied in the main scripts
#######################################################################################################


def load_dict(args, neuron_dir, top_n):
    """Load the differnet conditions."""
    suppress_dict = JsonProcessor.load_json(neuron_dir / "suppress" / f"{args.data_range_end}_{top_n}.json")
    boost_dict = JsonProcessor.load_json(neuron_dir / "boost" / f"{args.data_range_end}_{top_n}.json")
    return suppress_dict, boost_dict


def get_overlap_stat(suppress_dict: dict, boost_dict: dict, group_type: str) -> dict:
    """Compare the ovelapping between different conditions."""
    overlap_steps = list(set(suppress_dict.keys()) & set(boost_dict.keys()))
    step_stat = {}
    for step in overlap_steps:
        overlap_index = set(suppress_dict[step][group_type]["neurons"]) & set(boost_dict[step][group_type]["neurons"])
        overlap_rate = len(overlap_index) / min(len(suppress_dict), len(boost_dict))
        if overlap_rate > 0:
            step_stat[step] = [overlap_index, overlap_rate]
            logger.info(f"Found overlapping indices in step {step}")
    return step_stat


#######################################################################################################
# Entry point of the script
#######################################################################################################


def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()
    index_dict = {}
    top_n_lst = [10, 50, 100]
    neuron_dir = settings.PATH.neuron_dir / "group" / args.vector / args.model / args.heuristic
    for top_n in top_n_lst:
        logger.info(f"Processing top {top_n} neurons")
        suppress_dict, boost_dict = load_dict(args, neuron_dir, top_n)
        step_stat_best = get_overlap_stat(suppress_dict, boost_dict, "best")
        step_stat_target = get_overlap_stat(suppress_dict, boost_dict, "target_size")
        index_dict[top_n] = [{"best": step_stat_best, "target_size": step_stat_target}]
        JsonProcessor.save_json(index_dict, neuron_dir / "overlap_stat.json")
        logger.info(f"Saving the results to {neuron_dir}/overlap_stat.json")


if __name__ == "__main__":
    main()
