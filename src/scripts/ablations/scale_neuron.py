from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
import transformer_lens
from sklearn.decomposition import PCA


def compute_unigram_direction(token_frequencies: torch.Tensor) -> torch.Tensor:
    """
    Compute the normalized unigram direction vector from token frequencies.
    
    Parameters:
    -----------
    token_frequencies: torch.Tensor
        Tensor of shape (vocab_size,) containing token frequencies
        
    Returns:
    --------
    torch.Tensor
        Normalized unigram direction vector in logit space
    """
    # Convert frequencies to log probabilities and center
    unigram_direction_vocab = token_frequencies.log() - token_frequencies.log().mean()
    
    # Normalize to unit vector
    unigram_direction_vocab /= unigram_direction_vocab.norm()
    
    return unigram_direction_vocab


def find_null_space(model, 
                    token_frequencies: torch.Tensor, 
                    top_k_percent: float = 0.05,
                    threshold: float = 0.95,
                    device: str = 'cuda') -> torch.Tensor:
    """
    Find the null space of the most frequent tokens.
    
    Parameters:
    -----------
    model: HookedTransformer
        Model containing the token embeddings
    token_frequencies: torch.Tensor
        Tensor of shape (vocab_size,) containing token frequencies
    top_k_percent: float
        Fraction of most frequent tokens to consider
    threshold: float
        Fraction of variance to preserve when finding the principal components
    device: str
        Device to run computations on
        
    Returns:
    --------
    torch.Tensor
        Basis vectors of the null space
    """
    vocab_size = token_frequencies.shape[0]
    embedding_dim = model.cfg.d_model
    
    # Get the most frequent tokens
    top_k = int(vocab_size * top_k_percent)
    top_indices = torch.argsort(token_frequencies, descending=True)[:top_k]
    
    # Extract the embeddings of top frequent tokens using W_U
    # W_U is the unembedding matrix (d_model, vocab_size)
    frequent_token_embeddings = model.W_U[:, top_indices].T.to(device)
    
    # Convert to numpy for SVD calculation
    embeddings_np = frequent_token_embeddings.detach().cpu().numpy()
    
    # Perform SVD to find the principal components
    U, S, Vh = np.linalg.svd(embeddings_np, full_matrices=True)
    
    # Calculate cumulative explained variance
    explained_variance_ratio = S**2 / np.sum(S**2)
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Find how many components to keep based on threshold
    k = np.argmax(cumulative_variance >= threshold) + 1
    print(f"Using {k} principal components to span the space of frequent tokens")
    
    # The null space is the remaining basis vectors
    null_space_basis = Vh[k:].T
    
    return torch.tensor(null_space_basis, dtype=torch.float32, device=device)


def scale_rare_token_weights(model,
                            token_frequencies: torch.Tensor,
                            null_space_basis: torch.Tensor,
                            rare_token_threshold: float = 0.001,
                            scaling_factor: float = 1.5,
                            device: str = 'cuda') -> torch.Tensor:
    """
    Scale weights in the null space for rare tokens.
    
    Parameters:
    -----------
    model: HookedTransformer
        Model containing the token embeddings
    token_frequencies: torch.Tensor
        Tensor of shape (vocab_size,) containing token frequencies
    null_space_basis: torch.Tensor
        Basis vectors of the null space
    rare_token_threshold: float
        Threshold below which tokens are considered rare
    scaling_factor: float
        Factor by which to scale the weights in the null space
        
    Returns:
    --------
    torch.Tensor
        Modified model weights with scaled null space components for rare tokens
    """
    # Get the unembedding matrix
    model_weights = model.W_U.clone()
    
    # Find rare tokens based on frequency threshold
    rare_token_mask = token_frequencies < rare_token_threshold
    
    # Create a projection matrix onto the null space
    projection_matrix = null_space_basis @ null_space_basis.T
    
    # For each rare token, scale its component in the null space
    rare_token_indices = torch.where(rare_token_mask)[0]
    
    for idx in tqdm.tqdm(rare_token_indices, desc="Scaling rare tokens"):
        # Get the token embedding (column of W_U)
        token_embedding = model_weights[:, idx]
        
        # Project the embedding onto the null space
        null_component = projection_matrix @ token_embedding
        
        # Scale the null space component
        modified_embedding = token_embedding + (scaling_factor - 1.0) * null_component
        
        # Update the model weights
        model_weights[:, idx] = modified_embedding
    
    return model_weights


def evaluate_token_probabilities(model, 
                                modified_model, 
                                tokenized_data, 
                                token_frequencies, 
                                rare_token_threshold: float = 0.001,
                                k: int = 10,
                                device: str = 'cuda'):
    """
    Evaluate the change in token probabilities after scaling the null space.
    
    Parameters:
    -----------
    model: HookedTransformer
        Original model
    modified_model: HookedTransformer
        Model with modified weights
    tokenized_data: Dataset
        Dataset containing tokenized text
    token_frequencies: torch.Tensor
        Tensor of shape (vocab_size,) containing token frequencies
    rare_token_threshold: float
        Threshold below which tokens are considered rare
    k: int
        Number of samples to evaluate
    device: str
        Device to run computations on
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing evaluation metrics
    """
    # Find rare tokens
    rare_token_mask = token_frequencies < rare_token_threshold
    rare_token_indices = torch.where(rare_token_mask)[0]
    
    # Sample some sequences
    random_sequence_indices = np.random.choice(len(tokenized_data), k, replace=False)
    
    results = []
    
    for batch_n in tqdm.tqdm(random_sequence_indices, desc="Evaluating probabilities"):
        tok_seq = tokenized_data['tokens'][batch_n].to(device)
        
        # Original model probabilities
        with torch.no_grad():
            original_logits = model(tok_seq.unsqueeze(0))[0]
            original_probs = original_logits.softmax(dim=-1)
            
            # Modified model probabilities
            modified_logits = modified_model(tok_seq.unsqueeze(0))[0]
            modified_probs = modified_logits.softmax(dim=-1)
        
        # Average probability for rare tokens
        orig_rare_probs = original_probs[:, rare_token_indices].mean().item()
        mod_rare_probs = modified_probs[:, rare_token_indices].mean().item()
        
        # KL divergence
        kl = F.kl_div(
            modified_probs.log(), 
            original_probs, 
            reduction='batchmean'
        ).item()
        
        # Top predictions
        orig_top_tokens = original_probs.topk(5, dim=-1).indices
        mod_top_tokens = modified_probs.topk(5, dim=-1).indices
        
        # Count rare tokens in top predictions
        orig_rare_in_top = sum([torch.sum((orig_top_tokens == idx).float()).item() for idx in rare_token_indices])
        mod_rare_in_top = sum([torch.sum((mod_top_tokens == idx).float()).item() for idx in rare_token_indices])
        
        results.append({
            'batch_n': batch_n,
            'original_rare_prob': orig_rare_probs,
            'modified_rare_prob': mod_rare_probs,
            'prob_increase': mod_rare_probs - orig_rare_probs,
            'kl_divergence': kl,
            'original_rare_in_top': orig_rare_in_top,
            'modified_rare_in_top': mod_rare_in_top
        })
    
    return pd.DataFrame(results)


def analyze_null_space_impact(unigram_direction, null_space_basis, model, device='cuda'):
    """
    Analyze the impact of null space scaling on the unigram direction.
    
    Parameters:
    -----------
    unigram_direction: torch.Tensor
        Unigram direction vector
    null_space_basis: torch.Tensor
        Basis vectors of the null space
    model: HookedTransformer
        Model to analyze
    device: str
        Device to run computations on
        
    Returns:
    --------
    dict
        Dictionary containing analysis metrics
    """
    # Project unigram direction onto null space
    projection_matrix = null_space_basis @ null_space_basis.T
    unigram_projection = projection_matrix @ unigram_direction
    
    # Calculate projection magnitude
    projection_magnitude = unigram_projection.norm().item()
    original_magnitude = unigram_direction.norm().item()
    projection_ratio = projection_magnitude / original_magnitude
    
    # Analyze impact on b_U
    b_U_projection = projection_matrix @ model.b_U
    b_U_projection_magnitude = b_U_projection.norm().item()
    b_U_original_magnitude = model.b_U.norm().item()
    b_U_projection_ratio = b_U_projection_magnitude / b_U_original_magnitude
    
    return {
        'unigram_projection_magnitude': projection_magnitude,
        'unigram_original_magnitude': original_magnitude,
        'unigram_projection_ratio': projection_ratio,
        'b_U_projection_magnitude': b_U_projection_magnitude,
        'b_U_original_magnitude': b_U_original_magnitude,
        'b_U_projection_ratio': b_U_projection_ratio
    }


def visualize_token_distribution_pca(token_frequencies: torch.Tensor, 
                                   original_weights: torch.Tensor, 
                                   modified_weights: torch.Tensor,
                                   rare_threshold: float = 0.001,
                                   output_path: Path = Path("token_distribution_pca.png")):
    """
    Visualize the distribution of tokens before and after scaling using PCA.
    
    Parameters:
    -----------
    token_frequencies: torch.Tensor
        Tensor of shape (vocab_size,) containing token frequencies
    original_weights: torch.Tensor
        Original model weights (d_model, vocab_size)
    modified_weights: torch.Tensor
        Modified model weights after null space scaling
    rare_threshold: float
        Threshold below which tokens are considered rare
    output_path: Path
        Path to save the visualization
    """
    # Transpose weights for easier processing (vocab_size, d_model)
    orig_weights_t = original_weights.T.detach().cpu().numpy()
    mod_weights_t = modified_weights.T.detach().cpu().numpy()
    
    # Use PCA to reduce dimensionality for visualization
    pca = PCA(n_components=2)
    
    # Apply PCA to both original and modified weights
    original_2d = pca.fit_transform(orig_weights_t)
    modified_2d = pca.transform(mod_weights_t)
    
    # Get frequency and rare token information
    log_frequencies = np.log(token_frequencies.detach().cpu().numpy() + 1e-10)
    rare_token_mask = (token_frequencies < rare_threshold).detach().cpu().numpy()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot original embeddings
    scatter1 = ax1.scatter(
        original_2d[:, 0], original_2d[:, 1], 
        c=log_frequencies, cmap='viridis', alpha=0.7,
        s=np.where(rare_token_mask, 15, 5)  # Larger points for rare tokens
    )
    ax1.set_title('Original Token Embeddings')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    
    # Plot modified embeddings
    scatter2 = ax2.scatter(
        modified_2d[:, 0], modified_2d[:, 1], 
        c=log_frequencies, cmap='viridis', alpha=0.7,
        s=np.where(rare_token_mask, 15, 5)  # Larger points for rare tokens
    )
    ax2.set_title('Modified Token Embeddings (Null Space Scaling)')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    
    # Add colorbar
    cbar = fig.colorbar(scatter1, ax=[ax1, ax2], label='Log Frequency')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return original_2d, modified_2d


def run_null_space_scaling_experiment(model, tokenized_data, token_frequencies, 
                                     top_k_percent=0.05, 
                                     rare_token_threshold=0.001,
                                     scaling_factor=1.5,
                                     eval_samples=10,
                                     device='cuda'):
    """
    Run the entire null space scaling experiment.
    
    Parameters:
    -----------
    model: HookedTransformer
        Model to modify
    tokenized_data: Dataset
        Dataset containing tokenized text
    token_frequencies: torch.Tensor
        Tensor of shape (vocab_size,) containing token frequencies
    top_k_percent: float
        Fraction of most frequent tokens to consider
    rare_token_threshold: float
        Threshold below which tokens are considered rare
    scaling_factor: float
        Factor by which to scale the weights in the null space
    eval_samples: int
        Number of samples to evaluate
    device: str
        Device to run computations on
        
    Returns:
    --------
    dict
        Dictionary containing experiment results
    """
    # Compute unigram direction
    print("Computing unigram direction...")
    unigram_direction = compute_unigram_direction(token_frequencies)
    
    # Find null space of frequent tokens
    print("Finding null space of frequent tokens...")
    null_space_basis = find_null_space(
        model, token_frequencies, top_k_percent, device=device
    )
    
    # Analyze null space impact on unigram direction
    print("Analyzing null space impact...")
    null_space_analysis = analyze_null_space_impact(
        unigram_direction, null_space_basis, model, device
    )
    
    # Scale rare token weights
    print("Scaling rare token weights...")
    modified_weights = scale_rare_token_weights(
        model, token_frequencies, null_space_basis, 
        rare_token_threshold, scaling_factor, device
    )
    
    # Create a modified model
    modified_model = model.copy()
    modified_model.W_U = modified_weights
    
    # Evaluate token probabilities
    print("Evaluating token probabilities...")
    evaluation = evaluate_token_probabilities(
        model, modified_model, tokenized_data, token_frequencies,
        rare_token_threshold, eval_samples, device
    )
    
    # Visualize token distribution
    print("Visualizing token distribution...")
    original_2d, modified_2d = visualize_token_distribution_pca(
        token_frequencies, model.W_U, modified_model.W_U,
        rare_token_threshold
    )
    
    return {
        'null_space_basis': null_space_basis,
        'null_space_analysis': null_space_analysis,
        'modified_weights': modified_weights,
        'evaluation': evaluation,
        'original_2d': original_2d,
        'modified_2d': modified_2d
    }


def main(model, tokenized_data, unigram_distrib, device='cuda'):
    """
    Main function to run the experiment.
    
    Parameters:
    -----------
    model: HookedTransformer
        Model to modify
    tokenized_data: Dataset
        Dataset containing tokenized text
    unigram_distrib: torch.Tensor
        Unigram distribution
    device: str
        Device to run computations on
    """
    # Check if unigram distribution is provided, otherwise compute it
    if unigram_distrib is None:
        # You need to provide a way to compute unigram_distrib here
        raise ValueError("Unigram distribution must be provided")
    
    # Run experiment
    results = run_null_space_scaling_experiment(
        model, tokenized_data, unigram_distrib,
        top_k_percent=0.05,
        rare_token_threshold=unigram_distrib.median().item() * 0.1,
        scaling_factor=1.5,
        eval_samples=10,
        device=device
    )
    
    # Print results
    print("\nResults:")
    print(f"Null space dimension: {results['null_space_basis'].shape[1]}")
    print(f"Unigram projection ratio: {results['null_space_analysis']['unigram_projection_ratio']:.4f}")
    print(f"b_U projection ratio: {results['null_space_analysis']['b_U_projection_ratio']:.4f}")
    
    print("\nEvaluation summary:")
    print(results['evaluation'].describe())
    
    return results

'''
if __name__ == "__main__":
    # The following assumes you have already loaded your model, data, and unigram distribution
    # Example usage:
    # import transformer_lens
    # model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")
    # tokenized_data = ...  # Your tokenized dataset
    # unigram_distrib = get_pile_unigram_distribution(model_name="gpt2-small", device="cuda")
    # main(model, tokenized_data, unigram_distrib)
    
    # For testing purposes, you could create dummy data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # This is a placeholder for actual implementation
    print("To run this code, please load your model, data, and unigram distribution first.")
'''


if __name__ == "__main__":
    import torch
    import numpy as np
    from pathlib import Path
    import transformer_lens
    import matplotlib.pyplot as plt
    from datasets import load_dataset
    import time
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running experiment on device: {device}")
    
    # Load GPT2-small
    print("Loading GPT2-small model...")
    model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small", device=device)
    tokenizer = model.tokenizer
    
    # Load a subset of the C4 dataset
    print("Loading dataset subset...")
    dataset = load_dataset("stas/c4-en-10k", split="train[:1000]")  # Use only first 1000 examples
    
    # Tokenize a small subset for unigram statistics
    print("Tokenizing dataset...")
    
    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, max_length=128)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Convert to format expected by our code
    tokenized_data = {
        'tokens': [torch.tensor(tokens, device=device) for tokens in tokenized_dataset["input_ids"]]
    }
    
    # Generate unigram distribution from these samples
    print("Computing unigram distribution...")
    token_counts = {}
    for tokens in tokenized_dataset["input_ids"]:
        for token in tokens:
            if token in token_counts:
                token_counts[token] += 1
            else:
                token_counts[token] = 1
    
    # Fill in missing tokens and smooth counts
    vocab_size = model.cfg.d_vocab
    unigram_counts = torch.ones(vocab_size, device=device)
    for token, count in token_counts.items():
        if token < vocab_size:
            unigram_counts[token] = count + 1  # +1 smoothing
    
    # Normalize to get distribution
    unigram_distrib = unigram_counts / unigram_counts.sum()
    
    print(f"Processed {len(tokenized_dataset)} text examples")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Found {len(token_counts)} unique tokens in dataset")
    
    # Set experiment parameters
    top_k_percent = 0.05  # Use top 5% of tokens as frequent
    rare_threshold = torch.quantile(unigram_distrib, 0.75).item()  # Bottom 25% as rare
    scaling_factor = 1.5  # Scale the null space component by 1.5x for rare tokens
    
    # Run the experiment phases
    print("\nComputing unigram direction...")
    unigram_direction = compute_unigram_direction(unigram_distrib)
    
    print("Finding null space of frequent tokens...")
    # Make sure we can access W_U
    try:
        W_U = model.W_U
        print(f"W_U shape: {W_U.shape}")
    except:
        # Try alternative ways to access the unembedding matrix
        try:
            W_U = model.unembed.weight
            print(f"Accessed W_U through model.unembed.weight, shape: {W_U.shape}")
        except:
            print("Could not access W_U directly, checking model parameters:")
            for name, param in model.named_parameters():
                if 'unembed' in name or 'W_U' in name:
                    print(f"Found parameter: {name}, shape: {param.shape}")
                    W_U = param
                    break
            else:
                raise ValueError("Cannot access the unembedding matrix (W_U)")
    
    start_time = time.time()
    
    # Now proceed with null space calculation
    null_space_basis = find_null_space(
        model, unigram_distrib, top_k_percent, device=device
    )
    
    print(f"Null space calculation time: {time.time() - start_time:.2f} seconds")
    print(f"Null space dimension: {null_space_basis.shape[1]}")
    
    # Calculate the null space component for each token embedding directly
    projection_matrix = null_space_basis @ null_space_basis.T  # Shape: [d_model, d_model]
    
    # Create a modified W_U for rare tokens
    modified_W_U = W_U.clone()
    
    # Find rare tokens based on frequency threshold
    rare_token_mask = unigram_distrib < rare_threshold
    rare_token_indices = torch.where(rare_token_mask)[0]
    
    print(f"Scaling {len(rare_token_indices)} rare tokens in null space...")
    scaling_start_time = time.time()
    
    for idx in rare_token_indices:
        # Get the token embedding (column of W_U)
        token_embedding = W_U[:, idx]
        
        # Project the embedding onto the null space
        null_component = projection_matrix @ token_embedding
        
        # Scale the null space component
        modified_embedding = token_embedding + (scaling_factor - 1.0) * null_component
        
        # Update the model weights
        modified_W_U[:, idx] = modified_embedding
    
    print(f"Scaling time: {time.time() - scaling_start_time:.2f} seconds")
    
    # Select test sequences from dataset
    test_texts = [
        "The technology could potentially revolutionize",
        "Scientists have discovered a new species of",
        "The financial markets reacted strongly to the",
        "In the aftermath of the recent political",
        "Researchers are developing innovative approaches to"
    ]
    
    # Run evaluation on test examples
    for test_text in test_texts:
        print("\n" + "="*50)
        print(f"Test prompt: \"{test_text}\"")
        
        test_tokens = tokenizer.encode(test_text)
        test_tensor = torch.tensor([test_tokens], device=device)
        
        # Get predictions from original model
        with torch.no_grad():
            original_logits = model(test_tensor)
        
        # Get the final residual output before unembedding
        with torch.no_grad():
            _, cache = model.run_with_cache(test_tensor)
            final_residual = cache["ln_final.hook_normalized"]
            
            # Apply the modified W_U
            modified_logits = torch.matmul(final_residual, modified_W_U)
        
        # Compare top 5 predictions for the next token
        orig_probs = torch.softmax(original_logits[0, -1], dim=-1)
        mod_probs = torch.softmax(modified_logits[0, -1], dim=-1)
        
        orig_top5 = orig_probs.topk(5)
        mod_top5 = mod_probs.topk(5)
        
        print("\nOriginal model's top 5 predictions:")
        for i in range(5):
            token_id = orig_top5.indices[i].item()
            token = tokenizer.decode([token_id])
            probability = orig_top5.values[i].item()
            is_rare = "RARE" if rare_token_mask[token_id].item() else ""
            print(f"  {token_id:5d} | '{token}' | Prob: {probability:.4f} {is_rare}")
        
        print("\nModified model's top 5 predictions:")
        for i in range(5):
            token_id = mod_top5.indices[i].item()
            token = tokenizer.decode([token_id])
            probability = mod_top5.values[i].item()
            is_rare = "RARE" if rare_token_mask[token_id].item() else ""
            print(f"  {token_id:5d} | '{token}' | Prob: {probability:.4f} {is_rare}")
        
        # Count rare tokens in top k predictions
        k = 20
        orig_rare_count = sum(rare_token_mask[orig_probs.topk(k).indices].cpu().numpy())
        mod_rare_count = sum(rare_token_mask[mod_probs.topk(k).indices].cpu().numpy())
        
        print(f"\nRare tokens in top {k} predictions:")
        print(f"  Original model: {orig_rare_count} ({orig_rare_count/k*100:.1f}%)")
        print(f"  Modified model: {mod_rare_count} ({mod_rare_count/k*100:.1f}%)")
    
    # Visualize how the model's token probabilities changed for the last test example
    sample_size = 200
    if sample_size > vocab_size:
        sample_size = vocab_size
    
    # Sample tokens across frequency spectrum
    # Instead of linear sampling, let's sample tokens more evenly across frequency bands
    freq_ranks = torch.argsort(unigram_distrib, descending=True)
    
    # Get indices from different frequency bands
    high_freq = freq_ranks[:sample_size//4]  # Top 25%
    mid_high_freq = freq_ranks[sample_size//4:sample_size//2]  # 25-50%
    mid_low_freq = freq_ranks[sample_size//2:3*sample_size//4]  # 50-75%
    low_freq = freq_ranks[3*sample_size//4:sample_size]  # Bottom 25%
    
    sampled_indices = torch.cat([high_freq, mid_high_freq, mid_low_freq, low_freq])
    
    # Get original and modified probabilities for these tokens
    orig_sampled_probs = orig_probs[sampled_indices].cpu().numpy()
    mod_sampled_probs = mod_probs[sampled_indices].cpu().numpy()
    
    # Calculate probability ratios (log scale for better visualization)
    prob_ratios = np.log10(mod_sampled_probs / (orig_sampled_probs + 1e-10))
    
    # Get frequencies for these tokens
    token_freqs = unigram_distrib[sampled_indices].cpu().numpy()
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(np.log10(token_freqs + 1e-10), prob_ratios, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.xlabel('Log Token Frequency')
    plt.ylabel('Log Probability Ratio (Modified/Original)')
    plt.title('Effect of Null Space Scaling on Token Probabilities')
    
    # Add a trend line
    z = np.polyfit(np.log10(token_freqs + 1e-10), prob_ratios, 1)
    p = np.poly1d(z)
    plt.plot(np.log10(token_freqs + 1e-10), p(np.log10(token_freqs + 1e-10)), 
             "r--", alpha=0.8, label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")
    plt.legend()
    
    # Highlight rare tokens
    rare_indices = np.where(rare_token_mask[sampled_indices].cpu().numpy())[0]
    plt.scatter(np.log10(token_freqs[rare_indices] + 1e-10), 
                prob_ratios[rare_indices], 
                color='red', alpha=0.5, s=30)
    
    visualization_path = Path("probability_changes.png")
    plt.tight_layout()
    plt.savefig(visualization_path)
    
    print(f"\nVisualization saved to {visualization_path}")
    print("\nExperiment complete.")