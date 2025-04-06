import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch

from neuron_analyzer import settings
from neuron_analyzer.geometry import NeuronGeometricAnalyzer
from neuron_analyzer.surprisal import StepConfig, StepSurprisalExtractor, load_neuron_dict

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compute neuron activation directions.")
    parser.add_argument(
        "-m", "--model_name", type=str, default="EleutherAI/pythia-70m-deduped", help="Target model name"
    )
    parser.add_argument("-n", "--neuron_file", type=str, default="500_50.csv", help="Target model name")
    parser.add_argument("--neuron_num", type=int, default=3, help="Target neuron num")
    parser.add_argument(
        "--vector",
        type=str,
        default="longtail",
        choices=["mean", "longtail"],
        help="Differnt ablation model for freq vectors",
    )
    parser.add_argument("--interval", type=int,default=10, help="Checkpoint interval sampling")
    parser.add_argument("--debug", action="store_true", help="Compute the first few 5 lines if enabled")
    parser.add_argument("--resume", action="store_true", help="Resume from the existing checkpoint")
    return parser.parse_args()

def get_filename(neuron_file:str,neuron_num:int)->str:
    """Insert the neuron num to the saved file.  e.g.500_10.csv """
    if neuron_num==0:
        return neuron_file
    else:
        file_prefix = neuron_file.split("_")[0]
        file_suffix = neuron_file.split(".")[1]
        return f"{file_prefix}_{neuron_num}.{file_suffix}"


def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ###################################
    # load materials and paths
    ###################################
    raw_filename = get_filename(args.neuron_file,args.neuron_num)
    result_dir = settings.PATH.direction_dir / "geometry" / args.model_name
    filename = f"{Path(raw_filename).stem}.debug" if args.debug else raw_filename
    subspace_file = result_dir / "subspace" / filename
    orthogonality_file = result_dir / "orthogonality" / filename
    if args.resume and subspace_file.is_file() and orthogonality_file.is_file():
        logger.info(f"Target files already exist, skipping processing as resume is enabled")
        sys.exit(0)

    subspace_file.parent.mkdir(parents=True, exist_ok=True)
    orthogonality_file.parent.mkdir(parents=True, exist_ok=True)

    # load neuron indices
    boost_step_ablations, layer_num = load_neuron_dict(
        settings.PATH.result_dir / "token_freq" / "boost" / args.vector / args.model_name / args.neuron_file,
        key_col="step",
        value_col="top_neurons",
        top_n = args.neuron_num
    )
    suppress_step_ablations, ayer_num = load_neuron_dict(
        settings.PATH.result_dir / "token_freq" / "suppress" / args.vector / args.model_name / args.neuron_file,
        key_col="step",
        value_col="top_neurons",
        top_n = args.neuron_num
    )


    ###################################
    # Initialize classes
    ###################################

    # Initialize configuration with all Pythia checkpoints
    steps_config = StepConfig(debug=args.debug,interval=args.interval)

    # Initialize extractor
    model_cache_dir = settings.PATH.model_dir / args.model_name
    extractor = StepSurprisalExtractor(
        config=steps_config,
        model_name=args.model_name,
        model_cache_dir=model_cache_dir,  # note here we use the relative path
        layer_num=layer_num,
        device=device,
    )

    ###################################
    # Save the target results
    ###################################

    # loop over different steps
    subspace_df = pd.DataFrame()
    orthogonality_df = pd.DataFrame()
    for step in steps_config.steps:
        try:
            # load model
            model, _ = extractor.load_model_for_step(step)
            # initilize the analyzer class
            geometry_analyzer = NeuronGeometricAnalyzer(
                model=model,
                layer_num=layer_num,
                boost_neurons=boost_step_ablations[step],
                suppress_neurons=suppress_step_ablations[step],
                device=device,
            )
            subspace, orthogonality = geometry_analyzer.run_analyses()
            subspace.insert(0, "step", step)
            orthogonality.insert(0, "step", step)
            subspace_df = pd.concat([subspace_df, subspace])
            orthogonality_df = pd.concat([orthogonality_df, orthogonality])
            logger.info(f"Successfully get result for step {step}")
        except:
            logger.info(f"Something wrong with step {step}")
            pass
    # Save results even if some checkpoints failed
    subspace_df.to_csv(subspace_file)
    logger.info(f"Subspace results saved to: {subspace_file}")
    orthogonality_df.to_csv(orthogonality_file)
    logger.info(f"Orthogonality results saved to: {orthogonality_file}")
    logger.info(f"Processed {len(steps_config.steps)} checkpoints successfully")


if __name__ == "__main__":
    main()
