import numpy as np
import pandas as pd
import torch
from scipy.linalg import subspace_angles


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
        self.common_neurons, self.sampled_common_neurons = self._get_common_neurons()

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
        else:
            sampled_common_neurons = common_neurons.copy()

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
            "right_singular_vectors": Vh,
        }

        # Calculate decay rates if we have at least 2 singular values
        if len(S) >= 2:
            result["sv_decay_rate_2"] = S[0] / S[1]

        return result

    def orthogonality_measurement(self, Vh_1, Vh_2, dim_1, dim_2):
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

    def get_neuron_pairs(self, dictionary: dict) -> dict[str, list]:
        """Generate all possible pairs of keys from a dictionary without repetition."""
        keys = list(dictionary.keys())
        pair_dict: dict[str, list] = {}

        # Loop through all possible pairs
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                # Create hyphenated key name from the two group names
                pair_key = f"{keys[i]}-{keys[j]}"
                pair_dict[pair_key] = [dictionary[keys[i]], dictionary[keys[j]]]

        return pair_dict

    def run_analyses(self):
        """Run analyses on subspace dimensionality."""
        # load neuron subspaces
        neuron_dict = {
            "all": self.common_neurons,
            "random": self.sampled_common_neurons,
            "suppress": self.suppress_neurons,  # Fixed typo
            "boost": self.boost_neurons,
        }

        # First, compute subspace dimensionality for each neuron group
        subspace_results = {}
        subspace_lst = []

        for neuron_type, neurons in neuron_dict.items():
            result = self.subspace_dimensionality(neurons)
            subspace_results[neuron_type] = result  # Store the result for later use
            subspace_lst.append(result)

        subspace_df = pd.DataFrame(subspace_lst)
        subspace_df.insert(0, "neuron", neuron_dict.keys())
        # remove the right_singular_vectors col
        subspace_df = subspace_df.drop(columns="right_singular_vectors")

        # Now compute orthogonality metrics using the subspace results
        pair_keys = []
        orthogonality_lst = []

        # Generate all pairs of neuron types
        neuron_types = list(subspace_results.keys())
        for i in range(len(neuron_types)):
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
                orthogonality_lst.append(result)

        orthogonality_df = pd.DataFrame(orthogonality_lst)
        orthogonality_df.insert(0, "pair", pair_keys)  # Fixed: using pair_keys instead of neuron_dict.keys()

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
        elif row["null_space_percent"] > 50:
            return "moderately_orthogonal"
        elif row["null_space_percent"] > 25:
            return "slightly_orthogonal"
        else:
            return "not_orthogonal"

    stat_df["orthogonality_category"] = stat_df.apply(interpret_orthogonality, axis=1)

    return stat_df
