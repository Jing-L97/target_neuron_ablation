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
    parser.add_argument("-n", "--neuron_file", type=str, default="500_50.csv", help="Target model name")
    parser.add_argument(
        "--vector",type=str,
        default="longtail",choices=["mean", "longtail"],
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
        self.layer_num = layer_num
        self.device = device
        self.results = {}

        # Get common neurons (not boost or suppress)
        self.common_neurons,self.sampled_common_neurons = self._get_common_neurons()

    def _get_common_neurons(self):
        """Get neurons that are neither boosting nor suppressing."""
        # Get layer to determine total neurons
        layer_path = f"gpt_neox.layers.{self.layer_num}.mlp.dense_h_to_4h"
        layer_dict = dict(self.model.named_modules())
        layer = layer_dict[layer_path]
        total_neurons = layer.weight.shape[0]

        # Get neurons that are neither boosting nor suppressing
        all_special = set(self.boost_neurons + self.suppress_neurons)
        common_neurons = [i for i in range(total_neurons) if i not in all_special]

        # Sample a subset of similar size if there are too many
        reference_size = max(len(self.boost_neurons), len(self.suppress_neurons))
        if len(common_neurons) > reference_size:
            sampled_common_neurons = np.random.choice(common_neurons, size=reference_size, replace=False).tolist()

        return common_neurons, sampled_common_neurons

    def extract_neuron_weights(self, neuron_indices):
        """Extract weight vectors for specified neurons in a layer."""
        layer_path = f"gpt_neox.layers.{self.layer_num}.mlp.dense_h_to_4h"
        layer_dict = dict(self.model.named_modules())
        layer = layer_dict[layer_path]

        # Get weight matrix
        W = layer.weight.detach().cpu().numpy()
        # Extract weights for specific neurons
        W_neurons = W[neuron_indices]
        # Ensure we return a 2D array
        if len(W_neurons.shape) == 1:
            W_neurons = W_neurons.reshape(1, -1)

        return W_neurons

    def subspace_dimensionality(self, group_indices):
        """Analyze the dimensionality of neuron subspaces"""

        # Extract weights
        W_group = self.extract_neuron_weights(group_indices)

        # Perform SVD on both neuron groups
        U, S, Vh = np.linalg.svd(W_group, full_matrices=False)
        # Calculate normalized singular values
        S_norm = S / S.sum() if S.sum() > 0 else S

        # Calculate cumulative explained variance
        cum_var = np.cumsum(S_norm)

        # Estimate effective dimensionality (number of dimensions for 95% variance)
        dim = np.argmax(cum_var >= 0.95) + 1 if np.any(cum_var >= 0.95) else len(S)

        # Store metrics
        result = {
            "effective_dim": dim,
            "total_dim": len(S),
            "top5_sv": S_norm[: min(5, len(S_norm))].tolist(),
            "right_singular_vectors": Vh,
        }

        # Calculate decay rates if we have at least 2 singular values
        if len(S) >= 2:
            result["sv_decay_rate_2"] = S[0] / S[1]

        return result

    def orthogonality_measurement(self,Vh_1, Vh_2, dim_1, dim_2):
        """Measure orthogonality between two subspaces."""
        if Vh_1 is None or Vh_2 is None:
            return None

        # Make sure we don't exceed available dimensions
        dim_1 = min(dim_1, Vh_1.shape[0])
        dim_2 = min(dim_2, Vh_2.shape[0])

        # Use effective dimensions to define subspaces
        V_1 = Vh_1[:dim_1].T  # Column vectors spanning subspace 1
        V_2 = Vh_2[:dim_2].T  # Column vectors spanning subspace 2

        # Compute principal angles between subspaces
        try:
            angles = subspace_angles(V_1, V_2)
        except Exception:
            # Handle case when angles calculation fails
            return None

        # Calculate summary statistics
        mean_angle = np.mean(angles)
        median_angle = np.median(angles)
        min_angle = np.min(angles)

        # Calculate percentage of near-orthogonal angles (80°-100°)
        angles_degrees = np.degrees(angles)
        near_orthogonal = ((angles_degrees >= 80) & (angles_degrees <= 100)).mean()

        # Store the metrics
        result = {
            "mean_angle_degrees": np.degrees(mean_angle),
            "median_angle_degrees": np.degrees(median_angle),
            "min_angle_degrees": np.degrees(min_angle),
            "pct_near_orthogonal": near_orthogonal * 100,
        }

        return result

    def get_neruon_pairs(self,dictionary)->list:   #TODO: revise this function
        """Generate all possible pairs of keys from a dictionary without repetition. """
        keys = list(dictionary.keys())
        pair_dict = {}
        # Loop through all possible pairs
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                pair_dict.append([keys[i], keys[j]])
        return pair_dict


    def run_analyses(self):
        """Run analyses on subspace dimensionality."""

        # load neuron subspaces
        neuron_dict = {
            "common":self.common_neurons,
            "sampled_common":self.sampled_common_neurons,
            "supress":self.suppress_neurons,
            "boost":self.boost_neurons
        }
        subspace_lst = []
        for _,neurons in neuron_dict.items():
            subspace_lst.append(self.subspace_dimensionality(neurons))
        subspace_df = pd.DataFrame(subspace_lst)
        subspace_df["neuron"]=neuron_dict.keys()

        # compute orthogonality metrics
        pair_dict = self.get_neuron_pairs(neuron_dict)
        orthogonality_lst = []
        for _,pair in pair_dict.items():
            orthogonality_lst.append
            (
                self.orthogonality_measurement
                    (
                        pair[0]["right_singular_vectors"],
                        pair[1]["right_singular_vectors"],
                        pair[0]["effective_dim"],
                        pair[1]["effective_dim"]
                    )
            )
        orthogonality_df = pd.DataFrame(subspace_lst)
        orthogonality_df["pair"] = pair_dict.keys()
        return subspace_df


    
def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ###################################
    # load materials and paths
    ###################################
    result_dir = settings.PATH.direction_dir / "geometry" / args.model_name 
    subspace_file = result_dir/ "subspace" / args.neuron_file
    orthogonality_file = result_dir/  "orthogonality" / args.neuron_file
    result_file.parent.mkdir(parents=True, exist_ok=True)

    # load neuron indices
    boost_step_ablations,layer_num = load_neuron_dict(
        settings.PATH.result_dir / "token_freq" / "boost" / args.vector / args.model_name / args.neuron_file,
        key_col="step",
        value_col="top_neurons",
    )

    suppress_step_ablations, ayer_num = load_neuron_dict(
        settings.PATH.result_dir / "token_freq" / "suppress" / args.vector / args.model_name / args.neuron_file,
        key_col="step",
        value_col="top_neurons",
    )


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
