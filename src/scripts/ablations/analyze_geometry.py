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



import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from pathlib import Path


class NeuronGeometricAnalyzer:
    def __init__(self, model, layer_num: int, boost_neurons: list[int], suppress_neurons: list[int], device):
        """Initialize the analyzer with model and previously identified neuron sets. """
        self.model = model
        self.boost_neurons = boost_neurons
        self.suppress_neurons = suppress_neurons
        self.layer_num = layer_num
        self.device = device
        self.results = {}
        # Get common neurons (not boost or suppress)
        self.common_neurons, self.sampled_common_neurons_1, self.sampled_common_neurons_2 = self._get_common_neurons()

    def _get_common_neurons(self) -> tuple[list[int], list[int], list[int]]:
        """Get neurons that are neither boosting nor suppressing and create two non-overlapping samples."""
        # Get layer to determine total neurons
        layer_path = f"gpt_neox.layers.{self.layer_num}.mlp.dense_h_to_4h"
        layer_dict = dict(self.model.named_modules())
        layer = layer_dict[layer_path]
        total_neurons = layer.weight.shape[0]

        # Get neurons that are neither boosting nor suppressing
        all_special = set(self.boost_neurons + self.suppress_neurons)
        common_neurons = [i for i in range(total_neurons) if i not in all_special]

        # Sample two non-overlapping subsets of similar size
        reference_size = max(len(self.boost_neurons), len(self.suppress_neurons))
        # Ensure we can create two non-overlapping samples
        if len(common_neurons) >= 2 * reference_size:
            # Shuffle the common neurons
            np.random.shuffle(common_neurons)
            # Create two non-overlapping samples
            sampled_common_neurons_1 = common_neurons[:reference_size]
            sampled_common_neurons_2 = common_neurons[reference_size:2*reference_size]
        else:
            # If we don't have enough neurons, split evenly
            split_point = len(common_neurons) // 2
            sampled_common_neurons_1 = common_neurons[:split_point]
            sampled_common_neurons_2 = common_neurons[split_point:]

        return common_neurons, sampled_common_neurons_1, sampled_common_neurons_2

    def extract_neuron_weights(self, neuron_indices: list[int]) -> np.ndarray:
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

    def subspace_dimensionality(self, group_indices: list[int]) -> dict:
        """Analyze the dimensionality of neuron subspaces."""
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
            "dim_prop": dim/len(S),
            "right_singular_vectors": Vh,
        }

        # Calculate decay rates if we have at least 2 singular values
        if len(S) >= 2:
            result["sv_decay_rate_2"] = S[0] / S[1]

        return result

    def self_orthogonality_measurement(self, Vh: np.ndarray, dim: int) -> dict | None:
        """Measure internal orthogonality within a single subspace."""
        if Vh is None or dim <= 1:
            return None  # Need at least 2 dimensions to calculate angles

        # Make sure we don't exceed available dimensions
        dim = min(dim, Vh.shape[0])
        
        # Use effective dimensions to define the subspace
        V = Vh[:dim].T  # Column vectors spanning the subspace
        
        # Calculate angles between all pairs of basis vectors within the subspace
        full_angles = []
        full_angles_degrees = []
        
        for i in range(V.shape[1]):
            v1 = V[:, i]
            v1_norm = np.linalg.norm(v1)
            
            for j in range(i+1, V.shape[1]):  # Only need upper triangle
                v2 = V[:, j]
                v2_norm = np.linalg.norm(v2)
                
                # Avoid division by zero
                if v1_norm > 0 and v2_norm > 0:
                    # Calculate dot product
                    dot_product = np.dot(v1, v2) / (v1_norm * v2_norm)
                    # Clamp to avoid numerical errors
                    dot_product = max(min(dot_product, 1.0), -1.0)
                    # Calculate angle in radians
                    angle = np.arccos(dot_product)
                    full_angles.append(angle)
                    # Convert to degrees
                    angle_degrees = np.degrees(angle)
                    full_angles_degrees.append(angle_degrees)
        
        if not full_angles:
            return None  # No valid angles calculated
            
        full_angles = np.array(full_angles)
        full_angles_degrees = np.array(full_angles_degrees)
        
        # Calculate metrics
        result = {
            "full_mean_angle_degrees": np.mean(full_angles_degrees),
            "full_median_angle_degrees": np.median(full_angles_degrees),
            "full_min_angle_degrees": np.min(full_angles_degrees),
            "full_max_angle_degrees": np.max(full_angles_degrees),
            "pct_near_orthogonal": (((full_angles_degrees >= 80) & (full_angles_degrees <= 100)).mean() * 100),
            "pct_obtuse_angles": ((full_angles_degrees > 90).mean() * 100),
            "pct_acute_angles": ((full_angles_degrees < 90).mean() * 100),
            "full_angle_degrees": full_angles_degrees.tolist()
        }

        return result

    def orthogonality_measurement(self, Vh_1: np.ndarray, Vh_2: np.ndarray, dim_1: int, dim_2: int) -> dict | None:
        """Measure orthogonality between two subspaces with full angle information."""
        if Vh_1 is None or Vh_2 is None:
            return None

        # Make sure we don't exceed available dimensions
        dim_1 = min(dim_1, Vh_1.shape[0])
        dim_2 = min(dim_2, Vh_2.shape[0])

        # Use effective dimensions to define subspaces
        V_1 = Vh_1[:dim_1].T  # Column vectors spanning subspace 1
        V_2 = Vh_2[:dim_2].T  # Column vectors spanning subspace 2

        # Compute principal angles between subspaces (these are always ≤ 90°)
        try:
            principal_angles = subspace_angles(V_1, V_2)
        except Exception as e:
            print(f"Error calculating subspace angles: {e}")
            return None

        # Convert to degrees
        principal_angles_degrees = np.degrees(principal_angles)
        
        # Calculate full directional angles between all pairs of basis vectors
        # This will capture angles > 90° by calculating the actual angle between vectors
        full_angles = []
        full_angles_degrees = []
        
        for i in range(V_1.shape[1]):
            v1 = V_1[:, i]
            v1_norm = np.linalg.norm(v1)
            
            for j in range(V_2.shape[1]):
                v2 = V_2[:, j]
                v2_norm = np.linalg.norm(v2)
                
                # Avoid division by zero
                if v1_norm > 0 and v2_norm > 0:
                    # Calculate dot product
                    dot_product = np.dot(v1, v2) / (v1_norm * v2_norm)
                    # Clamp to avoid numerical errors
                    dot_product = max(min(dot_product, 1.0), -1.0)
                    # Calculate angle in radians
                    angle = np.arccos(dot_product)
                    full_angles.append(angle)
                    # Convert to degrees
                    angle_degrees = np.degrees(angle)
                    full_angles_degrees.append(angle_degrees)
        
        full_angles = np.array(full_angles)
        full_angles_degrees = np.array(full_angles_degrees)
        
        # Calculate metrics
        result = {
            # Principal angles (≤ 90°)
            "principal_mean_angle_degrees": np.mean(principal_angles_degrees),
            "principal_median_angle_degrees": np.median(principal_angles_degrees),
            "principal_min_angle_degrees": np.min(principal_angles_degrees),
            "principal_max_angle_degrees": np.max(principal_angles_degrees),
            
            # Full directional angles (can be > 90°)
            "full_mean_angle_degrees": np.mean(full_angles_degrees),
            "full_median_angle_degrees": np.median(full_angles_degrees),
            "full_min_angle_degrees": np.min(full_angles_degrees),
            "full_max_angle_degrees": np.max(full_angles_degrees),
            
            # Percentage of angles in different ranges
            "pct_near_orthogonal": (((full_angles_degrees >= 80) & (full_angles_degrees <= 100)).mean() * 100),
            "pct_obtuse_angles": ((full_angles_degrees > 90).mean() * 100),
            "pct_acute_angles": ((full_angles_degrees < 90).mean() * 100),
        }
        
        # Include histograms of angle distributions
        result["principal_angle_degrees"] = principal_angles_degrees.tolist()
        result["full_angle_degrees"] = full_angles_degrees.tolist()

        return result

    def get_neuron_pairs(self, dictionary: dict) -> dict[str, list]:
        """Generate all possible pairs of keys from a dictionary, including self-pairs."""
        keys = list(dictionary.keys())
        pair_dict: dict[str, list] = {}

        # Include all possible pairs, including self-pairs
        for i in range(len(keys)):
            # Add self-pair for measuring internal orthogonality
            self_key = f"{keys[i]}-{keys[i]}"
            pair_dict[self_key] = [dictionary[keys[i]], dictionary[keys[i]]]
            
            # Add cross-pairs
            for j in range(i + 1, len(keys)):
                pair_key = f"{keys[i]}-{keys[j]}"
                pair_dict[pair_key] = [dictionary[keys[i]], dictionary[keys[j]]]

        return pair_dict

    def run_analyses(self) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Run analyses on subspace dimensionality and orthogonality."""
        # load neuron subspaces
        neuron_dict = {
            "all": self.common_neurons,
            "random_1": self.sampled_common_neurons_1,
            "random_2": self.sampled_common_neurons_2,
            "suppress": self.suppress_neurons,
            "boost": self.boost_neurons,
        }

        # First, compute subspace dimensionality for each neuron group
        subspace_results = {}
        subspace_lst = []

        for neuron_type, neurons in neuron_dict.items():
            result = self.subspace_dimensionality(neurons)
            subspace_results[neuron_type] = result  # Store the result for later use
            result_copy = result.copy()
            # Remove the large array before adding to the DataFrame
            if "right_singular_vectors" in result_copy:
                del result_copy["right_singular_vectors"]
            result_copy["neuron"] = neuron_type
            subspace_lst.append(result_copy)

        subspace_df = pd.DataFrame(subspace_lst)
        # Now compute orthogonality metrics using the subspace results
        pair_keys = []
        orthogonality_lst = []

        # Generate all pairs of neuron types, including self-pairs
        neuron_types = list(subspace_results.keys())
        for i in range(len(neuron_types)):
            # Measure self-orthogonality
            type_self = neuron_types[i]
            self_key = f"{type_self}-{type_self}"
            pair_keys.append(self_key)
            
            # Use self-orthogonality measurement for self-pairs
            self_result = self.self_orthogonality_measurement(
                subspace_results[type_self]["right_singular_vectors"],
                subspace_results[type_self]["effective_dim"]
            )
            
            if self_result:
                # Extract angle distributions before adding to the DataFrame
                angle_distributions = {
                    "self_angles": self_result.pop("self_angle_degrees", [])
                }
                
                self_result["pair"] = self_key
                orthogonality_lst.append(self_result)
            
            # Now handle cross-comparisons
            for j in range(i + 1, len(neuron_types)):
                type1 = neuron_types[i]
                type2 = neuron_types[j]
                pair_key = f"{type1}-{type2}"
                pair_keys.append(pair_key)

                # Use the subspace results for orthogonality measurement
                result = self.orthogonality_measurement(
                    subspace_results[type1]["right_singular_vectors"],
                    subspace_results[type2]["right_singular_vectors"],
                    subspace_results[type1]["effective_dim"],
                    subspace_results[type2]["effective_dim"],
                )
                
                if result:
                    # Extract angle distributions before adding to the DataFrame
                    angle_distributions = {
                        "principal_angles": result.pop("principal_angle_degrees", []),
                        "full_angles": result.pop("full_angle_degrees", [])
                    }
                    
                    result["pair"] = pair_key
                    orthogonality_lst.append(result)

        orthogonality_df = pd.DataFrame(orthogonality_lst)

        return subspace_df, orthogonality_df


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compute neuron activation directions.")
    parser.add_argument(
        "-m", "--model_name", type=str, default="EleutherAI/pythia-70m-deduped", help="Target model name"
    )
    parser.add_argument("-n", "--neuron_file", type=str, default="500_50.csv", help="Target model name")
    parser.add_argument("--neuron_num", type=int, default=30, help="Target neuron num")
    parser.add_argument(
        "--vector",
        type=str,
        default="longtail",
        choices=["mean", "longtail"],
        help="Differnt ablation model for freq vectors",
    )
    parser.add_argument("--debug", action="store_true", help="Compute the first few 5 lines if enabled")
    parser.add_argument("--resume", action="store_true", help="Resume from the existing checkpoint")
    return parser.parse_args()


def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ###################################
    # load materials and paths
    ###################################
    result_dir = settings.PATH.direction_dir / "geometry" / args.model_name
    filename = f"{Path(args.neuron_file).stem}.debug" if args.debug else args.neuron_file
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
        #try:
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
        '''
        except:
            logger.info(f"Something wrong with step {step}")
            pass
        '''
    # Save results even if some checkpoints failed
    subspace_df.to_csv(subspace_file)
    logger.info(f"Subspace results saved to: {subspace_file}")
    orthogonality_df.to_csv(orthogonality_file)
    logger.info(f"Orthogonality results saved to: {orthogonality_file}")
    logger.info(f"Processed {len(steps_config.steps)} checkpoints successfully")


if __name__ == "__main__":
    main()
