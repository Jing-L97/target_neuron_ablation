#!/usr/bin/env python
import argparse
import logging
from pathlib import Path

from neuron_analyzer import settings
from neuron_analyzer.analysis.geometry_util import NeuronGroupAnalyzer, get_device
from neuron_analyzer.analysis.htsr import WeightSpaceHeavyTailedAnalyzer
from neuron_analyzer.load_util import JsonProcessor, StepPathProcessor
from neuron_analyzer.model_util import ModelHandler

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
        "--group_type", type=str, choices=["individual", "group"], default="individual", help="different neuron groups"
    )
    parser.add_argument(
        "--group_size", type=str, choices=["best", "target_size"], default="best", help="different group size"
    )
    parser.add_argument("--sel_longtail", type=bool, default=True, help="whether to filter by longtail token")
    parser.add_argument("--sel_by_med", type=bool, default=False, help="whether to select by mediation effect")
    parser.add_argument("--load_stat", action="store_true", help="Whether to load from existing index")
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
    group_name = f"{args.group_type}_{args.group_size}" if args.group_type == "group" else args.group_type
    save_path = (
        settings.PATH.direction_dir
        / "htsr"
        / group_name
        / "activation"
        / args.vector
        / args.model
        / save_heuristic
        / f"{args.data_range_end}_{args.top_n}{filename_suffix}"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    abl_path = settings.PATH.result_dir / "ablations" / args.vector / args.model
    # only set the path when loading the group neurons
    neuron_dir = settings.PATH.neuron_dir / "group" / args.vector / args.model / args.heuristic
    return save_path, abl_path, neuron_dir


def get_layer_num(model: str) -> int:
    return 5 if "70" in model else 23


def run_analysis(args, device, step, abl_path: Path) -> dict:
    """Run the analysis pipeline."""
    neuron_analyzer = NeuronGroupAnalyzer(
        args,
        device=device,
        step_path=step[0],
        abl_path=abl_path,
    )
    boost_neuron_indices, suppress_neuron_indices = neuron_analyzer.load_neurons()
    model_handler = ModelHandler()
    model, _ = model_handler.load_model_and_tokenizer(
        step=step[1],
        model_name=args.model,
        hf_token_path=settings.PATH.unigram_dir / "hf_token.txt",
        device=device,
    )

    # initilize the class
    geometry_analyzer = WeightSpaceHeavyTailedAnalyzer(
        model=model,
        boost_neurons=boost_neuron_indices,
        suppress_neurons=suppress_neuron_indices,
        device=device,
        layer_num=get_layer_num(args.model),
    )
    return geometry_analyzer.run_all_analyses()


#######################################################################################################
# Entry point of the script
#######################################################################################################


def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()
    device, _ = get_device()

    # loop over different steps
    save_path, abl_path, neuron_dir = configure_path(args)
    # load and update result json
    step_processor = StepPathProcessor(abl_path)
    final_results, step_dirs = step_processor.resume_results(args.resume, save_path, neuron_dir)

    for step in step_dirs:
        results = run_analysis(args, device, step, abl_path)
        final_results[str(step[1])] = results
        # assign col headers
        JsonProcessor.save_json(final_results, save_path)
        logger.info(f"Save file to {save_path}")
        """
        except:
            logger.info(f"Something wrong with {step}")
        """


if __name__ == "__main__":
    main()
