import argparse
import logging

import pandas as pd
import torch

from neuron_analyzer import settings
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
    parser.add_argument(
        "-n", "--neuron_file", type=str, default="500_10.csv", 
        help="Target model name"
    )
    parser.add_argument("--effect", type=str, choices=["boost", "supress"],
        default="supress", help="boost or supress long-tail"
        )
    parser.add_argument(
        "--vector", type=str, default="longtail",
        choices=["mean", "longtail"],
        help="Differnt ablation model for freq vectors"
        )

    parser.add_argument("--debug", action="store_true", help="Compute the first few 5 lines if enabled")
    parser.add_argument("--resume", action="store_true", help="Resume from the existing checkpoint")
    return parser.parse_args()


def analyze_neuron_directions(model, layer_idx=-1):
    """Analyze orthogonality between all neurons in a layer."""
    # Parse layer index to get the correct layer
    layer_num = str(layer_idx) if layer_idx >= 0 else str(len(model.gpt_neox.layers) + layer_idx)
    
    # Get the MLP input projection layer
    input_layer_path = f"gpt_neox.layers.{layer_num}.mlp.dense_h_to_4h"
    layer_dict = dict(model.named_modules())
    input_layer = layer_dict[input_layer_path]
    
    # Get the weight matrix - shape [4*hidden_size, hidden_size]
    W_in = input_layer.weight.detach().cpu()
    
    # Get dimensions
    intermediate_size, hidden_size = W_in.shape
    print(f"Computing pairwise directions for {intermediate_size} neurons")
    
    # Normalize all neuron directions
    norm = torch.norm(W_in, dim=1, keepdim=True)
    normalized_directions = W_in / (norm + 1e-8)
    
    # Calculate complete pairwise cosine similarity matrix: [intermediate_size, intermediate_size] 
    cosine_sim_matrix = torch.matmul(normalized_directions, normalized_directions.T)

    cosine_df = pd.DataFrame(
        cosine_sim_matrix.numpy(),
        index=[f'{i}' for i in range(intermediate_size)],
        columns=[f'{i}' for i in range(intermediate_size)]
    )
    return cosine_df


def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ###################################
    # load materials and paths
    ###################################

    # load neuron indices
    step_ablations, layer_num = load_neuron_dict(
        settings.PATH.result_dir / "token_freq" /args.effect / args.vector / args.model_name / args.neuron_file,
        key_col = "step",
        value_col = "top_neurons"
        )

    ###################################
    # Initialize classes
    ###################################
    result_dir = settings.PATH.direction_dir / args.model_name 
    result_file =  result_dir / args.neuron_file
    result_file.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        resume_file = result_dir / "resume"/ args.neuron_file
        resume_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        resume_file = None
    # Initialize configuration with all Pythia checkpoints
    steps_config = StepConfig(resume=args.resume,debug= args.debug,file_path = resume_file)

    # Initialize extractor
    model_cache_dir = settings.PATH.model_dir / args.model_name
    extractor = StepSurprisalExtractor(
        config=steps_config,
        model_name=args.model_name,
        model_cache_dir=model_cache_dir,  # note here we use the relative path
        layer_num=layer_num,
        device=device
    )

    ###################################
    # Save the target results
    ###################################
    try:
        # loop over different steps
        for step in steps_config.steps:
            # load model
            model, _ = extractor.load_model_for_step(step)
            # analyze the activation directions
            #results_df = analyze_neuron_directions(model, step_ablations[step], layer_num)
            results_df = analyze_neuron_directions(model, layer_num)
        # Save results even if some checkpoints failed
        if not results_df.empty:
            results_df.to_csv(result_file, index=False)
            logger.info(
                f"Results saved to: {result_file}\n"
                f"Processed {len([col for col in results_df.columns if str(col).isdigit()])} checkpoints successfully"
            )
        else:
            logger.warning("No results were generated")

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise



if __name__ == "__main__":
    main()
