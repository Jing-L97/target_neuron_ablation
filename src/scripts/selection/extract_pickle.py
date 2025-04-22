#!/usr/bin/env python3
"""Script to recover and analyze search results from NeuronGroupSearch pickle files."""

import argparse
import logging
import pickle
from pathlib import Path

from neuron_analyzer import settings
from neuron_analyzer.load_util import JsonProcessor
from neuron_analyzer.selection.group import get_heuristics

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Seach neuron groups across different training steps.")

    parser.add_argument("-m", "--model", type=str, default="EleutherAI/pythia-70m-deduped", help="Target model name")
    parser.add_argument("--vector", choices=["mean", "longtail", "longtail_50"], default="longtail_50")
    parser.add_argument(
        "--effect", type=str, choices=["boost", "suppress"], default="suppress", help="boost or suppress long-tail prob"
    )
    parser.add_argument(
        "--heuristic", type=str, choices=["KL", "prob"], default="prob", help="heuristic besides mediation effect"
    )
    parser.add_argument("--top_n", type=int, default=10, help="use_bos_only if enabled")
    parser.add_argument("--data_range_end", type=int, default=500, help="the selected datarange")
    parser.add_argument("--resume", action="store_true", help="Whether to resume from exisitng file")
    return parser.parse_args()


#######################################################################################################
# Fucntions applied in the main scripts
#######################################################################################################


class SearchStateLoader:
    """Manages loading and processing of search state data."""

    def __init__(self, cache_dir: Path):
        """Initialize the SearchStateManager with a cache directory."""
        self.cache_dir = cache_dir

    def load_search_state(self, method: str) -> dict | None:
        """Load a search state from a pickle file."""
        path = self.cache_dir / f"{method}_search_state.pkl"

        if not path.exists():
            print(f"No state file found for method {method} in {path}")
            return None

        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading pickle file for {method}: {e}")
            return None

    def get_search_state(self, method: str) -> list[dict] | None:
        """Get the best and target size results for a method."""
        result = self.load_search_state(method)
        if result and result.get("completed"):
            return [result["best_result"], result["target_size_result"]]
        return None

    def get_all_states(self, method_lst: list[str]) -> dict[str, list[dict]]:
        """Get search states for multiple methods."""
        state_all = {}
        for method in method_lst:
            state = self.get_search_state(method)
            if state:
                state_all[method] = state
        return state_all

    def get_best_result(self, method_lst: list[str], effect: str) -> dict:
        """Get the best method and its results based on delta_loss."""
        results = self.get_all_states(method_lst)
        # set whether to maximize heuristic based on effect
        maximize = get_heuristics(effect)
        # Comparison function based on maximization goal
        if maximize:
            best_method_name = max(results.keys(), key=lambda x: results[x][0]["delta_loss"])
            target_method_name = max(results.keys(), key=lambda x: results[x][1]["delta_loss"])
        else:
            best_method_name = min(results.keys(), key=lambda x: results[x][0]["delta_loss"])
            target_method_name = min(results.keys(), key=lambda x: results[x][1]["delta_loss"])
        return {"best": results[best_method_name][0], "target_size": results[target_method_name][1], "total": results}


#######################################################################################################
# Entry point of the script
#######################################################################################################


def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()

    # configure save_dir
    method_lst = ["progressive_beam", "hierarchical_cluster", "iterative_pruning", "importance_weighted", "hybrid"]
    save_dir = settings.PATH.neuron_dir / "group" / args.vector / args.model / args.heuristic / args.effect
    save_path = save_dir / f"{args.data_range_end}_{args.top_n}.json"
    cache_dir = save_dir / "cache" / str(args.top_n)
    # loop over the cache directory
    final_results = {}
    for step in cache_dir.iterdir():
        try:
            # initilize the state loader
            search_loader = SearchStateLoader(cache_dir=step)
            results = search_loader.get_best_result(method_lst=method_lst, effect=args.effect)
            final_results[step.name] = results
            # save the intermediate checkpoints
            JsonProcessor.save_json(final_results, save_path)
            logger.info(f"Save the results to {save_path}")
        except:
            logger.info(f"Something wrong with {step}")


if __name__ == "__main__":
    main()
