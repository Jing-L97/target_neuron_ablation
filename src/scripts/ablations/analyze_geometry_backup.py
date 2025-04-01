import argparse
import logging

import numpy as np
import pandas as pd
import torch
from scipy.linalg import subspace_angles

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
    parser.add_argument("--layer_num", type=int, default=5, help="layer num")
    parser.add_argument("-n", "--neuron_file", type=str, default="500_50.csv", help="Target model name")
    parser.add_argument(
        "--vector",type=str,default="longtail",
        choices=["mean", "longtail"],
        help="Differnt ablation model for freq vectors",
    )

    parser.add_argument("--debug", action="store_true", help="Compute the first few 5 lines if enabled")
    parser.add_argument("--resume", action="store_true", help="Resume from the existing checkpoint")
    return parser.parse_args()


class NeuronGeometricAnalyzer:
    def __init__(self, model, layer_num: int, boost_neurons: list, suppress_neurons: list, device):
        """Initialize the analyzer with model and previously identified neuron sets."""
        self.model = model
        self.boost_neurons = boost_neurons
        self.suppress_neurons = suppress_neurons
        self.results = {}
        self.layer_num = layer_num
        self.device = device

    def extract_neuron_weights(self, neuron_indices):
        """Extract weight vectors for specified neurons in a layer."""
        # Adjust this based on your model architecture
        layer_path = f"gpt_neox.layers.{self.layer_num}.mlp.dense_h_to_4h"
        layer_dict = dict(self.model.named_modules())
        layer = layer_dict[layer_path]

        # Get weight matrix
        W = layer.weight.detach().cpu().numpy()
        # Extract weights for specific neurons
        W_neurons = W[neuron_indices]
        return W_neurons

    def subspace_dimensionality(self):
        """Analyze the dimensionality of rare token neuron subspaces."""
        # Extract weights for boost and suppress neurons
        boost_indices = self.boost_neurons
        suppress_indices = self.suppress_neurons

        # Get weights and print shapes
        W_boost = self.extract_neuron_weights(boost_indices)
        W_suppress = self.extract_neuron_weights(suppress_indices)

        # Now proceed with SVD on properly shaped arrays

        U_b, S_b, Vh_b = np.linalg.svd(W_boost, full_matrices=False)
        U_s, S_s, Vh_s = np.linalg.svd(W_suppress, full_matrices=False)

        # Calculate normalized singular values
        S_b_norm = S_b / S_b.sum()
        S_s_norm = S_s / S_s.sum()

        # Calculate cumulative explained variance
        cum_var_b = np.cumsum(S_b_norm)
        cum_var_s = np.cumsum(S_s_norm)

        # Estimate effective dimensionality (number of dimensions for 95% variance)
        dim_b = np.argmax(cum_var_b >= 0.95) + 1 if np.any(cum_var_b >= 0.95) else len(S_b)
        dim_s = np.argmax(cum_var_s >= 0.95) + 1 if np.any(cum_var_s >= 0.95) else len(S_s)

        # Store results
        result = {
            "singular_values_boost": S_b,
            "singular_values_suppress": S_s,
            "normalized_sv_boost": S_b_norm,
            "normalized_sv_suppress": S_s_norm,
            "cumulative_variance_boost": cum_var_b,
            "cumulative_variance_suppress": cum_var_s,
            "effective_dim_boost": dim_b,
            "effective_dim_suppress": dim_s,
            "right_singular_vectors_boost": Vh_b,
            "right_singular_vectors_suppress": Vh_s,
        }

        return result

    def orthogonality_measurement(self):  # TODO: add common tokens
        """Measure orthogonality between boost and suppress subspaces."""
        # Get the right singular vectors from method 1
        Vh_b = self.result["right_singular_vectors_boost"]
        Vh_s = self.result["right_singular_vectors_suppress"]

        # Get effective dimensions
        dim_b = self.result["effective_dim_boost"]
        dim_s = self.result["effective_dim_suppress"]

        # Use effective dimensions to define subspaces
        V_b = Vh_b[:dim_b].T  # Column vectors spanning boost subspace
        V_s = Vh_s[:dim_s].T  # Column vectors spanning suppress subspace

        # Compute principal angles between subspaces
        angles = subspace_angles(V_b, V_s)

        # Calculate summary statistics
        mean_angle = np.mean(angles)
        median_angle = np.median(angles)
        min_angle = np.min(angles)

        # Calculate cosine similarities between all pairs of basis vectors
        cosine_sim_matrix = np.dot(Vh_b[:dim_b], Vh_s[:dim_s].T)

        result = {
            "principal_angles": angles,
            "mean_angle_radians": mean_angle,
            "mean_angle_degrees": np.degrees(mean_angle),
            "median_angle_degrees": np.degrees(median_angle),
            "min_angle_degrees": np.degrees(min_angle),
            "cosine_similarity_matrix": cosine_sim_matrix,
        }

        return result

    def run_analyses(self):
        """Run all methods of analysis."""
        # Initialize results for this layer if not already present
        self.results = {}

        # Run Method 1: Subspace Dimensionality
        m1_result = self.subspace_dimensionality()
        self.results["subspace_dimensionality"] = m1_result

        # Store m1_result as instance attribute for method 2 to use
        self.result = m1_result  # Note: this matches your orthogonality_measurement method which uses self.result

        # Run Method 2 based on Method 1 results
        if m1_result is not None:
            m2_result = self.orthogonality_measurement()
            self.results["orthogonality"] = m2_result

        # Return results as DataFrame
        return pd.DataFrame([self.results])  # Wrap in list to ensure proper DataFrame structure


def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ###################################
    # load materials and paths
    ###################################

    # load neuron indices
    boost_step_ablations, max_layer_num = load_neuron_dict(
        settings.PATH.result_dir / "token_freq" / "boost" / args.vector / args.model_name / args.neuron_file,
        key_col="step",
        value_col="top_neurons",
    )

    suppress_step_ablations, max_layer_num = load_neuron_dict(
        settings.PATH.result_dir / "token_freq" / "suppress" / args.vector / args.model_name / args.neuron_file,
        key_col="step",
        value_col="top_neurons",
    )

    if args.layer_num > max_layer_num:
        logger.error(f"Assigned {args.layer_num}: is larger than max MLP layers {max_layer_num}")
        raise

    ###################################
    # Initialize classes
    ###################################

    # Initialize configuration with all Pythia checkpoints
    steps_config = StepConfig(resume=args.resume, debug=args.debug)

    # Initialize extractor
    model_cache_dir = settings.PATH.model_dir / args.model_name
    extractor = StepSurprisalExtractor(
        config=steps_config,
        model_name=args.model_name,
        model_cache_dir=model_cache_dir,  # note here we use the relative path
        layer_num=args.layer_num,
        device=device,
    )

    ###################################
    # Save the target results
    ###################################

    # loop over different steps
    for step in steps_config.steps:
        # make the step directory
        result_file = settings.PATH.direction_dir / "geometry" / args.model_name / str(step) / f"{args.layer_num}.csv"

        if args.resume and result_file.is_file():
            logger.info(f"There exists: {result_file}. Skip")
        else:
            result_file.parent.mkdir(parents=True, exist_ok=True)
            # load model
            model, _ = extractor.load_model_for_step(step)
            # initilize the analyzer class
            geometry_analyzer = NeuronGeometricAnalyzer(
                model=model,
                layer_num=args.layer_num,
                boost_neurons=boost_step_ablations[step],
                suppress_neurons=suppress_step_ablations[step],
                device=device,
            )
            results_df = geometry_analyzer.run_analyses()
            # Save results even if some checkpoints failed
            if not results_df.empty:
                results_df.to_csv(result_file)
                logger.info(f"Results saved to: {result_file}")
            else:
                logger.warning("No results were generated for step")

    logger.info(f"Processed {len(steps_config.steps)} checkpoints successfully")


if __name__ == "__main__":
    main()
