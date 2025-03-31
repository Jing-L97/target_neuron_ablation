import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.linalg import subspace_angles


class RareTokenGeometricAnalyzer:
    def __init__(self, model, rare_token_boosting_neurons, rare_token_suppressing_neurons):
        """Initialize the analyzer with model and previously identified neuron sets.
        
        Parameters
        ----------
        model : the language model
        rare_token_boosting_neurons : dict mapping layer indices to lists of neuron indices
        rare_token_suppressing_neurons : dict mapping layer indices to lists of neuron indices

        """
        self.model = model
        self.boosting_neurons = rare_token_boosting_neurons
        self.suppressing_neurons = rare_token_suppressing_neurons
        self.device = next(model.parameters()).device
        self.results = {}

    def extract_neuron_weights(self, layer_idx, neuron_indices):
        """Extract weight vectors for specified neurons in a layer."""
        # Adjust this based on your model architecture
        layer_path = f"gpt_neox.layers.{layer_idx}.mlp.dense_h_to_4h"
        layer_dict = dict(self.model.named_modules())
        layer = layer_dict[layer_path]

        # Get weight matrix
        W = layer.weight.detach().cpu().numpy()

        # Extract weights for specific neurons
        W_neurons = W[neuron_indices]

        return W_neurons

    def subspace_dimensionality(self, layer_idx):
        """Method 1: Analyze the dimensionality of rare token neuron subspaces."""
        # Extract weights for boosting and suppressing neurons
        boosting_indices = self.boosting_neurons[layer_idx]
        suppressing_indices = self.suppressing_neurons[layer_idx]

        if not boosting_indices or not suppressing_indices:
            return None

        W_boosting = self.extract_neuron_weights(layer_idx, boosting_indices)
        W_suppressing = self.extract_neuron_weights(layer_idx, suppressing_indices)

        # Perform SVD on both neuron groups
        U_b, S_b, Vh_b = np.linalg.svd(W_boosting, full_matrices=False)
        U_s, S_s, Vh_s = np.linalg.svd(W_suppressing, full_matrices=False)

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
            "singular_values_boosting": S_b,
            "singular_values_suppressing": S_s,
            "normalized_sv_boosting": S_b_norm,
            "normalized_sv_suppressing": S_s_norm,
            "cumulative_variance_boosting": cum_var_b,
            "cumulative_variance_suppressing": cum_var_s,
            "effective_dim_boosting": dim_b,
            "effective_dim_suppressing": dim_s,
            "right_singular_vectors_boosting": Vh_b,
            "right_singular_vectors_suppressing": Vh_s
        }

        return result

    def orthogonality_measurement(self, layer_idx, result_method1=None):
        """Method 2: Measure orthogonality between boosting and suppressing subspaces."""
        if result_method1 is None:
            result_method1 = self.method1_subspace_dimensionality(layer_idx)

        if result_method1 is None:
            return None

        # Get the right singular vectors from method 1
        Vh_b = result_method1["right_singular_vectors_boosting"]
        Vh_s = result_method1["right_singular_vectors_suppressing"]

        # Get effective dimensions
        dim_b = result_method1["effective_dim_boosting"]
        dim_s = result_method1["effective_dim_suppressing"]

        # Use effective dimensions to define subspaces
        V_b = Vh_b[:dim_b].T  # Column vectors spanning boosting subspace
        V_s = Vh_s[:dim_s].T  # Column vectors spanning suppressing subspace

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
            "cosine_similarity_matrix": cosine_sim_matrix
        }

        return result

    def activation_correlation(self, layer_idx, input_texts, batch_size=8):
        """Method 3: Analyze correlations between neuron activations on actual inputs."""
        #TODO: integrate this part into surprisal analysis
        boosting_indices = self.boosting_neurons[layer_idx]
        suppressing_indices = self.suppressing_neurons[layer_idx]

        if not boosting_indices or not suppressing_indices:
            return None

        # Concatenate indices for tracking
        all_indices = boosting_indices + suppressing_indices
        neuron_types = ["boosting"] * len(boosting_indices) + ["suppressing"] * len(suppressing_indices)

        # Initialize activation collection
        all_activations = []

        # Define a hook function to collect activations
        def get_activations(module, input_tensor, output):
            # Extract activations for neurons of interest
            activations = output[:, :, all_indices].detach().cpu().numpy()
            all_activations.append(activations)

        # Register hook
        layer_path = f"gpt_neox.layers.{layer_idx}.mlp.dense_h_to_4h"
        layer_dict = dict(self.model.named_modules())
        layer = layer_dict[layer_path]
        hook = layer.register_forward_hook(get_activations)

        # Process input texts in batches
        tokenizer = self.model.tokenizer if hasattr(self.model, "tokenizer") else None
        if tokenizer is None:
            # Try to find tokenizer in common locations
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.model.config._name_or_path)
            except:
                raise ValueError("Tokenizer not found. Please provide tokenized inputs directly.")

        try:
            with torch.no_grad():
                for i in range(0, len(input_texts), batch_size):
                    batch_texts = input_texts[i:i+batch_size]
                    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    self.model(**inputs)
        finally:
            # Remove the hook
            hook.remove()

        # Combine all collected activations
        if not all_activations:
            return None

        combined_activations = np.concatenate([act.reshape(-1, len(all_indices)) for act in all_activations], axis=0)

        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(combined_activations.T)

        # Calculate average correlations within and between groups
        n_boosting = len(boosting_indices)
        n_suppressing = len(suppressing_indices)

        # Within-group correlations
        boosting_corr = correlation_matrix[:n_boosting, :n_boosting]
        boosting_corr = boosting_corr[~np.eye(boosting_corr.shape[0], dtype=bool)].mean()

        suppressing_corr = correlation_matrix[n_boosting:, n_boosting:]
        suppressing_corr = suppressing_corr[~np.eye(suppressing_corr.shape[0], dtype=bool)].mean()

        # Between-group correlations
        between_corr = correlation_matrix[:n_boosting, n_boosting:].mean()

        result = {
            "correlation_matrix": correlation_matrix,
            "avg_correlation_within_boosting": boosting_corr,
            "avg_correlation_within_suppressing": suppressing_corr,
            "avg_correlation_between_groups": between_corr,
            "neuron_indices": all_indices,
            "neuron_types": neuron_types
        }

        return result

    def cross_layer_comparison(self, layer_indices):
        """Method 4: Compare rare token subspaces across different layers."""
        # First run method 1 on all layers
        m1_results = {}
        for layer_idx in layer_indices:
            m1_results[layer_idx] = self.method1_subspace_dimensionality(layer_idx)

        layer_similarity_matrix = np.zeros((len(layer_indices), len(layer_indices)))
        subspace_evolution = {}

        # Compare each pair of layers
        for i, layer_i in enumerate(layer_indices):
            for j, layer_j in enumerate(layer_indices):
                if m1_results[layer_i] is None or m1_results[layer_j] is None:
                    layer_similarity_matrix[i, j] = np.nan
                    continue

                # Get principal components that explain 95% variance
                Vh_i = m1_results[layer_i]["right_singular_vectors_boosting"]
                dim_i = m1_results[layer_i]["effective_dim_boosting"]

                Vh_j = m1_results[layer_j]["right_singular_vectors_boosting"]
                dim_j = m1_results[layer_j]["effective_dim_boosting"]

                # Use min dimension for comparison
                min_dim = min(dim_i, dim_j)

                # Compare subspaces using Frobenius norm of projection
                projection = np.dot(Vh_i[:min_dim], Vh_j[:min_dim].T)
                similarity = np.linalg.norm(projection, "fro") / min_dim

                layer_similarity_matrix[i, j] = similarity

                # If consecutive layers, store more detailed info
                if j == i + 1:
                    # Calculate angles between corresponding components
                    component_angles = []
                    for k in range(min_dim):
                        cos_angle = np.abs(np.dot(Vh_i[k], Vh_j[k]))
                        angle = np.arccos(min(cos_angle, 1.0))
                        component_angles.append(np.degrees(angle))

                    subspace_evolution[f"{layer_i}_to_{layer_j}"] = {
                        "similarity": similarity,
                        "component_angles": component_angles
                    }

        result = {
            "layer_similarity_matrix": layer_similarity_matrix,
            "subspace_evolution": subspace_evolution
        }

        return result

    def run_all_analyses(self, layer_indices, input_texts=None):
        """Run all four methods of analysis."""
        self.results = {layer_idx: {} for layer_idx in layer_indices}

        # Method 1 and 2 for each layer
        for layer_idx in layer_indices:
            # Run Method 1
            m1_result = self.method1_subspace_dimensionality(layer_idx)
            self.results[layer_idx]["subspace_dimensionality"] = m1_result

            # Run Method 2 based on Method 1 results
            if m1_result is not None:
                m2_result = self.method2_orthogonality_measurement(layer_idx, m1_result)
                self.results[layer_idx]["orthogonality"] = m2_result

            # Run Method 3 if input texts are provided
            if input_texts is not None:
                m3_result = self.method3_activation_correlation(layer_idx, input_texts)
                self.results[layer_idx]["activation_correlation"] = m3_result

        # Method 4 across layers
        m4_result = self.method4_cross_layer_comparison(layer_indices)
        self.results["cross_layer"] = m4_result

        return self.results

    def visualize_results(self, output_dir=None):
        """Visualize the results of the analyses."""
        if not self.results:
            print("No results to visualize. Run analyses first.")
            return

        if output_dir is None:
            output_dir = "geometric_analysis_results"

        import os
        os.makedirs(output_dir, exist_ok=True)

        # Method 1: Singular value decay plots
        for layer_idx, layer_results in self.results.items():
            if layer_idx == "cross_layer":
                continue

            m1_result = layer_results.get("subspace_dimensionality")
            if m1_result is not None:
                plt.figure(figsize=(10, 6))
                plt.plot(m1_result["normalized_sv_boosting"], label="Boosting Neurons")
                plt.plot(m1_result["normalized_sv_suppressing"], label="Suppressing Neurons")
                plt.title(f"Layer {layer_idx}: Singular Value Decay")
                plt.xlabel("Component Index")
                plt.ylabel("Normalized Singular Value")
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{output_dir}/layer{layer_idx}_singular_values.png")
                plt.close()

                # Cumulative variance
                plt.figure(figsize=(10, 6))
                plt.plot(m1_result["cumulative_variance_boosting"], label="Boosting Neurons")
                plt.plot(m1_result["cumulative_variance_suppressing"], label="Suppressing Neurons")
                plt.axhline(y=0.95, color="r", linestyle="--", label="95% Variance Threshold")
                plt.title(f"Layer {layer_idx}: Cumulative Explained Variance")
                plt.xlabel("Number of Components")
                plt.ylabel("Cumulative Explained Variance")
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{output_dir}/layer{layer_idx}_cumulative_variance.png")
                plt.close()

            # Method 2: Orthogonality visualization
            m2_result = layer_results.get("orthogonality")
            if m2_result is not None:
                plt.figure(figsize=(10, 6))
                plt.hist(np.degrees(m2_result["principal_angles"]), bins=20)
                plt.axvline(x=90, color="r", linestyle="--", label="Orthogonal (90°)")
                plt.title(f"Layer {layer_idx}: Distribution of Principal Angles")
                plt.xlabel("Angle (degrees)")
                plt.ylabel("Frequency")
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{output_dir}/layer{layer_idx}_principal_angles.png")
                plt.close()

                # Cosine similarity heatmap
                plt.figure(figsize=(10, 8))
                plt.imshow(m2_result["cosine_similarity_matrix"], cmap="coolwarm", vmin=-1, vmax=1)
                plt.colorbar(label="Cosine Similarity")
                plt.title(f"Layer {layer_idx}: Cosine Similarity Between Subspace Basis Vectors")
                plt.xlabel("Suppressing Neuron Components")
                plt.ylabel("Boosting Neuron Components")
                plt.savefig(f"{output_dir}/layer{layer_idx}_cosine_similarity.png")
                plt.close()

            # Method 3: Activation correlation heatmap
            m3_result = layer_results.get("activation_correlation")
            if m3_result is not None:
                plt.figure(figsize=(12, 10))
                plt.imshow(m3_result["correlation_matrix"], cmap="coolwarm", vmin=-1, vmax=1)
                plt.colorbar(label="Correlation")

                # Add dividing lines between neuron groups
                n_boosting = sum(1 for t in m3_result["neuron_types"] if t == "boosting")
                plt.axhline(y=n_boosting-0.5, color="black", linestyle="-")
                plt.axvline(x=n_boosting-0.5, color="black", linestyle="-")

                plt.title(f"Layer {layer_idx}: Neuron Activation Correlation")
                plt.savefig(f"{output_dir}/layer{layer_idx}_activation_correlation.png")
                plt.close()

                # Bar chart of average correlations
                plt.figure(figsize=(10, 6))
                labels = ["Within Boosting", "Within Suppressing", "Between Groups"]
                values = [
                    m3_result["avg_correlation_within_boosting"],
                    m3_result["avg_correlation_within_suppressing"],
                    m3_result["avg_correlation_between_groups"]
                ]
                plt.bar(labels, values)
                plt.axhline(y=0, color="black", linestyle="-")
                plt.title(f"Layer {layer_idx}: Average Activation Correlations")
                plt.ylabel("Correlation Coefficient")
                plt.savefig(f"{output_dir}/layer{layer_idx}_avg_correlations.png")
                plt.close()

        # Method 4: Cross-layer similarity heatmap
        m4_result = self.results.get("cross_layer")
        if m4_result is not None:
            plt.figure(figsize=(10, 8))
            layer_indices = [layer_idx for layer_idx in self.results if layer_idx != "cross_layer"]
            plt.imshow(m4_result["layer_similarity_matrix"], cmap="viridis")
            plt.colorbar(label="Subspace Similarity")
            plt.title("Cross-Layer Subspace Similarity")
            plt.xticks(range(len(layer_indices)), layer_indices)
            plt.yticks(range(len(layer_indices)), layer_indices)
            plt.xlabel("Layer")
            plt.ylabel("Layer")
            plt.savefig(f"{output_dir}/cross_layer_similarity.png")
            plt.close()

            # Component angle evolution across consecutive layers
            evolution_data = m4_result["subspace_evolution"]
            if evolution_data:
                plt.figure(figsize=(12, 6))
                for layer_pair, data in evolution_data.items():
                    component_indices = range(len(data["component_angles"]))
                    plt.plot(component_indices, data["component_angles"],
                             marker="o", label=f"Layers {layer_pair}")
                plt.axhline(y=90, color="r", linestyle="--", label="Orthogonal (90°)")
                plt.title("Component Angle Evolution Across Layers")
                plt.xlabel("Component Index")
                plt.ylabel("Angle (degrees)")
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{output_dir}/component_angle_evolution.png")
                plt.close()



#######################################################
# Neuron direction analysis

def analyze_neuron_directions(model, layer_num=-1,chunk_size = 1024, device=None)->pd.DataFrame:
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
        results.append({
            'neuron_idx': neuron,
            'type': 'within_list',
            'mean_abs_cosine': within_mean,
            'max_abs_cosine': within_max,
            'min_abs_cosine': within_min,
            'null_space_count': within_null_count,
            'null_space_percent': within_null_percent,
            'total_comparisons': len(within_list)
        })
        
        results.append({
            'neuron_idx': neuron,
            'type': 'outside_list',
            'mean_abs_cosine': outside_mean,
            'max_abs_cosine': outside_max,
            'min_abs_cosine': outside_min,
            'null_space_count': outside_null_count,
            'null_space_percent': outside_null_percent,
            'total_comparisons': len(outside_list)
        })
    
    # Convert to DataFrame
    stat_df = pd.DataFrame(results)
    
    # Add interpretation column
    def interpret_orthogonality(row):
        if row['null_space_percent'] > 75:
            return 'highly_orthogonal'
        elif row['null_space_percent'] > 50:
            return 'moderately_orthogonal'
        elif row['null_space_percent'] > 25:
            return 'slightly_orthogonal'
        else:
            return 'not_orthogonal'
    
    stat_df['orthogonality_category'] = stat_df.apply(interpret_orthogonality, axis=1)
    
    return stat_df

