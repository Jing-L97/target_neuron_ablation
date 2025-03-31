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
    parser.add_argument("--layer_num", type=int, default=5,help="layer num")
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



def analyze_neuron_directions(model, layer_num=-1,chunk_size = 1024, device=None):
    """Analyze orthogonality between all neurons in a layer with optimized computation."""
    # Get weight matrix directly on the device where the model is
    input_layer_path = f"gpt_neox.layers.{layer_num}.mlp.dense_h_to_4h"
    layer_dict = dict(model.named_modules())
    input_layer = layer_dict[input_layer_path]

    # Get the weight matrix directly on the appropriate device
    W_in = input_layer.weight.detach()
    if W_in.device != device:
        W_in = W_in.to(device)

    # Get dimensions
    intermediate_size, hidden_size = W_in.shape
    # Normalize all neuron directions
    norm = torch.norm(W_in, dim=1, keepdim=True)
    normalized_directions = W_in / (norm + 1e-8)

    # Compute the cosine similarity matrix more efficiently
    # We can use chunking to avoid memory issues for large matrices
    cosine_sim_matrix = torch.zeros((intermediate_size, intermediate_size), device=device)

    # Calculate only the lower triangular part (including diagonal)
    for i in range(0, intermediate_size, chunk_size):
        end_i = min(i + chunk_size, intermediate_size)
        chunk_i = normalized_directions[i:end_i]

        # Calculate similarity with all neurons up to and including this chunk
        for j in range(0, end_i, chunk_size):
            end_j = min(j + chunk_size, end_i)  # Only calculate lower triangle
            chunk_j = normalized_directions[j:end_j]

            # Compute cosine similarity for this block
            block_sim = torch.matmul(chunk_i, chunk_j.T)

            # Fill in the corresponding part of the matrix
            cosine_sim_matrix[i:end_i, j:end_j] = block_sim

    # Make the matrix symmetric by copying the lower triangle to the upper triangle
    indices = torch.triu_indices(intermediate_size, intermediate_size, 1, device=device)
    cosine_sim_matrix[indices[1], indices[0]] = cosine_sim_matrix[indices[0], indices[1]]

    # Set diagonal to zero (self-similarity is not relevant for orthogonality analysis)
    cosine_sim_matrix.fill_diagonal_(0)

    # Move to CPU for DataFrame conversion
    cosine_sim_matrix_cpu = cosine_sim_matrix.cpu()

    # Convert to DataFrame with neuron indices as both row and column labels
    cosine_df = pd.DataFrame(
        cosine_sim_matrix_cpu.numpy(),
        index=list(range(intermediate_size)),
        columns=list(range(intermediate_size))
    )

    return cosine_df

#TODO: add seleted neuron analyses

def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ###################################
    # load materials and paths
    ###################################

    # load neuron indices
    step_ablations, max_layer_num = load_neuron_dict(
        settings.PATH.result_dir / "token_freq" /args.effect / args.vector / args.model_name / args.neuron_file,
        key_col = "step",
        value_col = "top_neurons"
        )
    if args.layer_num > max_layer_num:
        logger.error(f"Assigned {args.layer_num}: is larger than max MLP layers {max_layer_num}")
        raise

    ###################################
    # Initialize classes
    ###################################

    # Initialize configuration with all Pythia checkpoints
    steps_config = StepConfig(resume=args.resume,debug= args.debug)

    # Initialize extractor
    model_cache_dir = settings.PATH.model_dir / args.model_name
    extractor = StepSurprisalExtractor(
        config=steps_config,
        model_name=args.model_name,
        model_cache_dir=model_cache_dir,  # note here we use the relative path
        layer_num=args.layer_num,
        device=device
    )

    ###################################
    # Save the target results
    ###################################

    # loop over different steps
    for step in steps_config.steps:
        # make the step directory
        result_file =  settings.PATH.direction_dir / "neurons" / args.model_name / str(step)/f"{args.layer_num}.csv"

        if args.resume and result_file.is_file():
            logger.info(f"There exists: {result_file}. Skip")
        else:
            result_file.parent.mkdir(parents=True, exist_ok=True)
            # load model
            model, _ = extractor.load_model_for_step(step)
            # analyze the activation directions
            results_df = analyze_neuron_directions(model=model, layer_num =args.layer_num,device=device)
            # Save results even if some checkpoints failed
            if not results_df.empty:
                results_df.to_csv(result_file)
                logger.info(f"Results saved to: {result_file}")
            else:
                logger.warning("No results were generated for step")

    logger.info(f"Processed {len(steps_config.steps)} checkpoints successfully")





if __name__ == "__main__":
    main()