import numpy as np
import pandas as pd
import torch
from scipy.linalg import subspace_angles

#######################################################
# Neuron group subspace direction analysis


class NeuronGeometricAnalyzer:
    def __init__(self, model, layer_num: int, boost_neurons: list[int], suppress_neurons: list[int], device):
        """Initialize the analyzer with model and previously identified neuron sets."""
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
            sampled_common_neurons_2 = common_neurons[reference_size : 2 * reference_size]
        else:
            # If we don't have enough neurons, split evenly
            split_point = len(common_neurons) // 2
            sampled_common_neurons_1 = common_neurons[:split_point]
            sampled_common_neurons_2 = common_neurons[split_point:]

        return common_neurons, sampled_common_neurons_1, sampled_common_neurons_2

    # TODO: add activation values
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
            "dim_prop": dim / len(S),
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

            for j in range(i + 1, V.shape[1]):  # Only need upper triangle
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

        # Calculate metrics - use the same column names as orthogonality_measurement
        # but with NaN values for principal angle related metrics
        result = {
            # Principal angles (≤ 90°) - set to NaN for self-pairs
            "principal_mean_angle_degrees": np.nan,
            "principal_median_angle_degrees": np.nan,
            "principal_min_angle_degrees": np.nan,
            "principal_max_angle_degrees": np.nan,
            "principal_angle_degrees": np.nan,
            # Full directional angles
            "full_mean_angle_degrees": np.mean(full_angles_degrees),
            "full_median_angle_degrees": np.median(full_angles_degrees),
            "full_min_angle_degrees": np.min(full_angles_degrees),
            "full_max_angle_degrees": np.max(full_angles_degrees),
            # Percentage of angles in different ranges
            "pct_near_orthogonal": (((full_angles_degrees >= 80) & (full_angles_degrees <= 100)).mean() * 100),
            "pct_obtuse_angles": ((full_angles_degrees > 90).mean() * 100),
            "pct_acute_angles": ((full_angles_degrees < 90).mean() * 100),
            # Flag to identify self-pairs
            "is_self_pair": True,
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
            "principal_angle_degrees": principal_angles_degrees.tolist(),
            # Full directional angles (can be > 90°)
            "full_mean_angle_degrees": np.mean(full_angles_degrees),
            "full_median_angle_degrees": np.median(full_angles_degrees),
            "full_min_angle_degrees": np.min(full_angles_degrees),
            "full_max_angle_degrees": np.max(full_angles_degrees),
            # Percentage of angles in different ranges
            "pct_near_orthogonal": (((full_angles_degrees >= 80) & (full_angles_degrees <= 100)).mean() * 100),
            "pct_obtuse_angles": ((full_angles_degrees > 90).mean() * 100),
            "pct_acute_angles": ((full_angles_degrees < 90).mean() * 100),
            # Flag to identify self-pairs
            "is_self_pair": False,
        }

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
        orthogonality_lst = []

        # Generate all pairs of neuron types, including self-pairs
        neuron_types = list(subspace_results.keys())
        for i in range(len(neuron_types)):
            # Measure self-orthogonality
            type_self = neuron_types[i]
            self_key = f"{type_self}-{type_self}"

            # Use self-orthogonality measurement for self-pairs
            self_result = self.self_orthogonality_measurement(
                subspace_results[type_self]["right_singular_vectors"], subspace_results[type_self]["effective_dim"]
            )

            if self_result:
                # We already include NaN values for principal angle metrics in the self_result
                self_result["pair"] = self_key
                orthogonality_lst.append(self_result)

            # Now handle cross-comparisons
            for j in range(i + 1, len(neuron_types)):
                type1 = neuron_types[i]
                type2 = neuron_types[j]
                pair_key = f"{type1}-{type2}"

                # Use the subspace results for orthogonality measurement
                result = self.orthogonality_measurement(
                    subspace_results[type1]["right_singular_vectors"],
                    subspace_results[type2]["right_singular_vectors"],
                    subspace_results[type1]["effective_dim"],
                    subspace_results[type2]["effective_dim"],
                )

                if result:
                    result["pair"] = pair_key
                    orthogonality_lst.append(result)

        # Create DataFrame with consistent columns
        orthogonality_df = pd.DataFrame(orthogonality_lst)

        # Process the angle distributions columns - these contain lists which may cause issues
        # Fix the error by correctly checking types instead of using pd.isna()
        return subspace_df, orthogonality_df


#######################################################
# Neuron direction analysis


def analyze_neuron_directions(model, layer_num=-1, chunk_size=1024, device=None) -> pd.DataFrame:
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
        cosine_sim_matrix_cpu.numpy(), index=list(range(intermediate_size)), columns=list(range(intermediate_size))
    )

    return cosine_df


def get_stat(cosine_df, neuron_idx: list, threshold: float = 0.1) -> pd.DataFrame:
    """Compute neuron statistics based on cosine similarity matrix."""
    # Ensure all neurons are in the dataframe
    all_neurons = set(cosine_df.index)
    valid_neurons = [n for n in neuron_idx if n in all_neurons]

    if len(valid_neurons) < len(neuron_idx):
        missing = set(neuron_idx) - all_neurons
        print(f"Warning: {len(missing)} neurons not found in the matrix: {missing}")

    if not valid_neurons:
        raise ValueError("No valid neurons found in the cosine similarity matrix")

    # Create a mask for neurons in the list and outside the list
    neuron_set = set(valid_neurons)
    all_neurons = set(cosine_df.index)
    neurons_outside = list(all_neurons - neuron_set)

    # Prepare results container
    results = []

    # Process each neuron
    for neuron in valid_neurons:
        # Get cosine similarities
        similarities = cosine_df.loc[neuron]

        # Within list statistics (excluding self)
        within_list = [similarities[n] for n in valid_neurons if n != neuron]
        if within_list:
            within_mean = sum(abs(v) for v in within_list) / len(within_list)
            within_max = max(abs(v) for v in within_list)
            within_min = min(abs(v) for v in within_list)
            within_null_count = sum(1 for v in within_list if abs(v) < threshold)
            within_null_percent = (within_null_count / len(within_list)) * 100 if within_list else 0
        else:
            within_mean = within_max = within_min = within_null_count = within_null_percent = 0

        # Outside list statistics
        outside_list = [similarities[n] for n in neurons_outside]
        if outside_list:
            outside_mean = sum(abs(v) for v in outside_list) / len(outside_list)
            outside_max = max(abs(v) for v in outside_list)
            outside_min = min(abs(v) for v in outside_list)
            outside_null_count = sum(1 for v in outside_list if abs(v) < threshold)
            outside_null_percent = (outside_null_count / len(outside_list)) * 100
        else:
            outside_mean = outside_max = outside_min = outside_null_count = outside_null_percent = 0

        # Add to results (two rows per neuron - within and outside)
        results.append(
            {
                "neuron_idx": neuron,
                "type": "within_list",
                "mean_abs_cosine": within_mean,
                "max_abs_cosine": within_max,
                "min_abs_cosine": within_min,
                "null_space_count": within_null_count,
                "null_space_percent": within_null_percent,
                "total_comparisons": len(within_list),
            }
        )

        results.append(
            {
                "neuron_idx": neuron,
                "type": "outside_list",
                "mean_abs_cosine": outside_mean,
                "max_abs_cosine": outside_max,
                "min_abs_cosine": outside_min,
                "null_space_count": outside_null_count,
                "null_space_percent": outside_null_percent,
                "total_comparisons": len(outside_list),
            }
        )

    # Convert to DataFrame
    stat_df = pd.DataFrame(results)

    # Add interpretation column
    def interpret_orthogonality(row):
        if row["null_space_percent"] > 75:
            return "highly_orthogonal"
        if row["null_space_percent"] > 50:
            return "moderately_orthogonal"
        if row["null_space_percent"] > 25:
            return "slightly_orthogonal"
        return "not_orthogonal"

    stat_df["orthogonality_category"] = stat_df.apply(interpret_orthogonality, axis=1)

    return stat_df
