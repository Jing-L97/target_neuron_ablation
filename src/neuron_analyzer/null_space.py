import gc

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class NullSpaceScaler:
    """Scales rare token embeddings in the null space of frequent tokens to increase their probability."""

    def __init__(
        self,
        model,
        tokenizer,
        dataset,
        token_frequencies: torch.Tensor,
        top_k_percent: float = 0.05,
        rare_threshold: float = None,
        base_scaling_factor: float = 1.5,
        scaling_method: str = "linear",
        variance_threshold: float = 0.95,
        exponent: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the TokenNullSpaceScaler."""
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.token_frequencies = token_frequencies
        self.top_k_percent = top_k_percent

        # Set rare threshold to 25th percentile if not specified
        if rare_threshold is None:
            self.rare_threshold = torch.quantile(token_frequencies, 0.25).item()
        else:
            self.rare_threshold = rare_threshold

        self.base_scaling_factor = base_scaling_factor
        self.scaling_method = scaling_method
        self.variance_threshold = variance_threshold
        self.exponent = exponent
        self.device = device

        # Get the unembedding matrix
        self.W_U = self._get_unembedding_matrix()

        # Calculated attributes
        self.unigram_direction = None
        self.null_space_basis = None
        self.projection_matrix = None
        self.modified_weights = None
        self.analysis_results = None

    def _get_unembedding_matrix(self) -> torch.Tensor:
        """Access the unembedding matrix, trying multiple approaches."""
        try:
            return self.model.W_U
        except:
            try:
                return self.model.unembed.weight
            except:
                for name, param in self.model.named_parameters():
                    if "unembed" in name or "W_U" in name:
                        return param
                raise ValueError("Cannot access the unembedding matrix (W_U)")

    def compute_unigram_direction(self) -> torch.Tensor:
        """Compute the normalized unigram direction vector from token frequencies."""
        # Convert frequencies to log probabilities and center
        unigram_direction_vocab = self.token_frequencies.log() - self.token_frequencies.log().mean()

        # Normalize to unit vector
        unigram_direction_vocab /= unigram_direction_vocab.norm()

        self.unigram_direction = unigram_direction_vocab.to(self.device)
        return unigram_direction_vocab

    def find_null_space(self) -> torch.Tensor:
        """Find the null space of the most frequent tokens."""
        vocab_size = self.token_frequencies.shape[0]

        # Get the most frequent tokens
        top_k = int(vocab_size * self.top_k_percent)
        top_indices = torch.argsort(self.token_frequencies, descending=True)[:top_k]

        # Extract the embeddings of top frequent tokens using W_U
        frequent_token_embeddings = self.W_U[:, top_indices].T.to(self.device)

        # Convert to numpy for SVD calculation
        embeddings_np = frequent_token_embeddings.detach().cpu().numpy()

        # Perform SVD to find the principal components
        U, S, Vh = np.linalg.svd(embeddings_np, full_matrices=True)

        # Calculate cumulative explained variance
        explained_variance_ratio = S**2 / np.sum(S**2)
        cumulative_variance = np.cumsum(explained_variance_ratio)

        # Find how many components to keep based on threshold
        k = np.argmax(cumulative_variance >= self.variance_threshold) + 1
        print(f"Using {k} principal components to span the space of frequent tokens")

        # The null space is the remaining basis vectors
        null_space_basis = Vh[k:].T

        self.null_space_basis = torch.tensor(null_space_basis, dtype=torch.float32, device=self.device)
        self.projection_matrix = self.null_space_basis @ self.null_space_basis.T.to(self.device)

        return self.null_space_basis

    def scale_rare_token_weights(self) -> torch.Tensor:
        """Scale weights in the null space for rare tokens with frequency-dependent scaling."""
        if self.null_space_basis is None:
            self.find_null_space()

        # Find rare tokens based on frequency threshold
        rare_token_mask = self.token_frequencies < self.rare_threshold

        # Clone the model weights to avoid modifying the original
        self.modified_weights = self.W_U.clone()

        # For each rare token, scale its component in the null space
        rare_token_indices = torch.where(rare_token_mask)[0]
        min_freq = self.token_frequencies[rare_token_indices].min().item()

        # Added logging
        print(f"Scaling {len(rare_token_indices)} rare tokens with {self.scaling_method} scaling")
        print(f"Base scaling factor: {self.base_scaling_factor}, Rare threshold: {self.rare_threshold}")
        print(f"Min frequency: {min_freq}, Max possible scaling: {self.base_scaling_factor}")

        # Create a dictionary to hold scaling statistics
        scaling_stats = {"min_scaling": float("inf"), "max_scaling": 0.0, "avg_scaling": 0.0, "total_scaled": 0}

        for idx in tqdm(rare_token_indices, desc="Scaling rare tokens"):
            # Get token frequency
            token_freq = self.token_frequencies[idx].item()

            # Calculate dynamic scaling factor based on the chosen method
            if self.scaling_method == "linear":
                # Linear scaling: rarest tokens get max scaling, tokens at threshold get no scaling
                dynamic_scaling = 1.0 + (self.base_scaling_factor - 1.0) * (
                    (self.rare_threshold - token_freq) / (self.rare_threshold - min_freq + 1e-10)
                )
            elif self.scaling_method == "log":
                # Logarithmic scaling: more aggressive scaling for very rare tokens
                dynamic_scaling = 1.0 + (self.base_scaling_factor - 1.0) * (
                    np.log(self.rare_threshold / token_freq) / np.log(self.rare_threshold / (min_freq + 1e-10))
                )
            elif self.scaling_method == "power":
                # Power law scaling: adjustable curve based on exponent
                dynamic_scaling = 1.0 + (self.base_scaling_factor - 1.0) * (
                    (self.rare_threshold / token_freq) ** self.exponent
                    / (self.rare_threshold / (min_freq + 1e-10)) ** self.exponent
                )
            else:
                # Default to constant scaling if method not recognized
                dynamic_scaling = self.base_scaling_factor

            # Update scaling statistics
            scaling_stats["min_scaling"] = min(scaling_stats["min_scaling"], dynamic_scaling)
            scaling_stats["max_scaling"] = max(scaling_stats["max_scaling"], dynamic_scaling)
            scaling_stats["avg_scaling"] += dynamic_scaling
            scaling_stats["total_scaled"] += 1

            # Get the token embedding (column of W_U)
            token_embedding = self.W_U[:, idx]

            # Project the embedding onto the null space
            null_component = self.projection_matrix @ token_embedding

            # Scale the null space component with dynamic scaling
            modified_embedding = token_embedding + (dynamic_scaling - 1.0) * null_component

            # Update the model weights
            self.modified_weights[:, idx] = modified_embedding

        # Calculate average scaling
        if scaling_stats["total_scaled"] > 0:
            scaling_stats["avg_scaling"] /= scaling_stats["total_scaled"]

        print(f"Scaling complete. Statistics:")
        print(f"  Min scaling: {scaling_stats['min_scaling']:.4f}")
        print(f"  Max scaling: {scaling_stats['max_scaling']:.4f}")
        print(f"  Avg scaling: {scaling_stats['avg_scaling']:.4f}")

        return self.modified_weights

    def analyze_null_space_impact(self) -> dict:
        """Analyze the impact of null space scaling on the unigram direction."""
        if self.unigram_direction is None:
            self.compute_unigram_direction()

        if self.null_space_basis is None:
            self.find_null_space()

        # Check if dimensions match between unigram_direction and W_U
        W_U_shape = self.W_U.shape[1]
        unigram_shape = self.unigram_direction.shape[0]

        if W_U_shape != unigram_shape:
            print(f"Dimension mismatch detected: W_U columns ({W_U_shape}) vs unigram_direction ({unigram_shape})")

            # Pad unigram_direction with zeros to match W_U size
            if unigram_shape < W_U_shape:
                token_discrepancy = W_U_shape - unigram_shape
                padding = torch.zeros(token_discrepancy, device=self.unigram_direction.device)
                self.unigram_direction = torch.cat([self.unigram_direction, padding])
                print(f"Padded unigram_direction with {token_discrepancy} zeros to match W_U dimensions")

        # Move tensors to the same device
        self.unigram_direction = self.unigram_direction.to(self.device)
        self.W_U = self.W_U.to(self.device)

        # Reshape unigram_direction to column vector for matrix multiplication
        unigram_direction_col = self.unigram_direction.reshape(-1, 1)

        # Project from vocabulary space to embedding space
        unigram_direction_embedding = self.W_U @ unigram_direction_col

        # Flatten back to vector
        unigram_direction_embedding = unigram_direction_embedding.flatten()

        # Project onto null space
        unigram_projection = self.projection_matrix @ unigram_direction_embedding

        # Calculate projection magnitude
        projection_magnitude = unigram_projection.norm().item()
        original_magnitude = unigram_direction_embedding.norm().item()
        projection_ratio = projection_magnitude / original_magnitude

        # Analyze impact on b_U
        try:
            # Get b_U and move to device
            b_U = self.model.b_U.to(self.device)

            # Check shapes and print for debugging
            print(
                f"b_U shape: {b_U.shape}, W_U shape: {self.W_U.shape}, projection_matrix shape: {self.projection_matrix.shape}"
            )

            # The bias doesn't need to be projected through W_U since it's already in the right space
            # We just need to make sure it's compatible with the projection matrix
            if len(b_U.shape) == 1:  # If b_U is a vector
                if b_U.shape[0] != self.projection_matrix.shape[1]:
                    print(
                        f"b_U dimension ({b_U.shape[0]}) doesn't match projection matrix inner dimension ({self.projection_matrix.shape[1]})"
                    )

                    # We need a bias vector that's compatible with our embedding dimension
                    # If we can't reshape b_U directly, we'll skip this calculation
                    b_U_projection_magnitude = 0.0
                    b_U_original_magnitude = b_U.norm().item()
                    b_U_projection_ratio = 0.0
                else:
                    b_U_projection = self.projection_matrix @ b_U
                    b_U_projection_magnitude = b_U_projection.norm().item()
                    b_U_original_magnitude = b_U.norm().item()
                    b_U_projection_ratio = b_U_projection_magnitude / b_U_original_magnitude
            else:
                print(f"Unexpected shape for b_U: {b_U.shape}")
                b_U_projection_magnitude = 0.0
                b_U_original_magnitude = b_U.norm().item()
                b_U_projection_ratio = 0.0

        except (AttributeError, RuntimeError) as e:
            print(f"Error analyzing b_U: {e}")
            # Default values if b_U analysis fails
            b_U_projection_magnitude = 0.0
            b_U_original_magnitude = 0.0
            b_U_projection_ratio = 0.0

        self.analysis_results = {
            "unigram_projection_magnitude": projection_magnitude,
            "unigram_original_magnitude": original_magnitude,
            "unigram_projection_ratio": projection_ratio,
            "b_U_projection_magnitude": b_U_projection_magnitude,
            "b_U_original_magnitude": b_U_original_magnitude,
            "b_U_projection_ratio": b_U_projection_ratio,
        }

        return self.analysis_results

    
    def evaluate_token_prob(self, tokenized_data=None) -> pd.DataFrame:
        """
        Evaluate the change in token probabilities after scaling the null space.
        Handles tensor dimension issues, fixes KL divergence calculation, and adds diagnostics.
        """
        if self.modified_weights is None:
            self.scale_rare_token_weights()
            
        # Find rare tokens
        rare_token_mask = self.token_frequencies < self.rare_threshold
        rare_token_indices = torch.where(rare_token_mask)[0].to(self.device)
        
        # Fix: Move CUDA tensor to CPU before converting to numpy
        rare_token_set = set(rare_token_indices.cpu().numpy())
        
        # Get the tokenized data if not provided
        if tokenized_data is None:
            tokenized_data = self._tokenize_dataset()
        
        print(f"Type of tokenized_data: {type(tokenized_data)}")
        
        # Extract token sequences based on tokenized_data format
        if hasattr(tokenized_data, 'input_ids'):
            token_sequences = tokenized_data.input_ids
        elif isinstance(tokenized_data, dict) and 'input_ids' in tokenized_data:
            token_sequences = tokenized_data['input_ids']
        else:
            print(f"Unexpected format for tokenized_data: {type(tokenized_data)}")
            if hasattr(tokenized_data, '__getitem__') and 'tokens' in tokenized_data:
                token_sequences = tokenized_data['tokens']
            else:
                raise ValueError(f"Could not extract token sequences from: {type(tokenized_data)}")
        
        # Process sequence token by token
        results = []
        
        # Determine shape and iteration approach
        if hasattr(token_sequences, 'shape'):
            if len(token_sequences.shape) == 2:  # [batch_size, seq_length]
                batch_size, seq_length = token_sequences.shape
            else:
                # Handle single dimension case
                seq_length = token_sequences.shape[0]
                batch_size = 1
        else:
            # Handle list-like objects
            batch_size = len(token_sequences)
            seq_length = len(token_sequences[0]) if batch_size > 0 else 0
        
        print(f"Processing {batch_size} sequences of length {seq_length}")
        
        # Process in smaller chunks to avoid CUDA memory issues
        chunk_size = 8  # Smaller chunks are safer
        
        # Print diagnostic information
        vocab_size = self.token_frequencies.shape[0]
        max_rare_index = rare_token_indices.max().item() if len(rare_token_indices) > 0 else 0
        print(f"Vocabulary size: {vocab_size}")
        print(f"Max rare token index: {max_rare_index}")
        print(f"W_U shape: {self.W_U.shape}")
        print(f"Number of rare tokens: {len(rare_token_indices)}")
        
        for batch_idx in range(batch_size):
            for chunk_start in range(0, seq_length, chunk_size):
                chunk_end = min(chunk_start + chunk_size, seq_length)
                
                # Clear memory before processing this chunk
                torch.cuda.empty_cache()
                gc.collect()
                
                for pos in tqdm(range(chunk_start, chunk_end), 
                            desc=f"Batch {batch_idx+1}/{batch_size}, Chunk {chunk_start}-{chunk_end-1}"):
                    try:
                        # Extract single token sequence
                        if hasattr(token_sequences, 'shape') and len(token_sequences.shape) == 2:
                            tok_seq = token_sequences[batch_idx][pos].clone()
                        elif hasattr(token_sequences, 'shape'):
                            tok_seq = token_sequences[pos].clone()
                        else:
                            tok_seq = token_sequences[batch_idx][pos]
                        
                        # Ensure tok_seq is a tensor and handle device
                        if not isinstance(tok_seq, torch.Tensor):
                            tok_seq = torch.tensor([tok_seq], device=self.device)
                        else:
                            tok_seq = tok_seq.to(self.device)
                        
                        # Ensure proper dimensions for model input
                        if len(tok_seq.shape) == 0:  # Single token scalar
                            tok_seq = tok_seq.unsqueeze(0).unsqueeze(0)  # [1, 1]
                        elif len(tok_seq.shape) == 1:  # Single token in a vector
                            tok_seq = tok_seq.unsqueeze(0)  # [1, seq]
                        
                        # Process with error handling
                        try:
                            with torch.no_grad():
                                # Get original probabilities
                                original_output = self.model(tok_seq)
                                original_logits = original_output[0] if isinstance(original_output, tuple) else original_output
                                
                                # Print shapes for debugging
                                if pos == chunk_start:  # Only print once per chunk
                                    print(f"Model output type: {type(original_output)}")
                                    if isinstance(original_output, tuple):
                                        print(f"Model output tuple length: {len(original_output)}")
                                        for i, item in enumerate(original_output):
                                            print(f"Model output[{i}] shape: {item.shape if hasattr(item, 'shape') else 'not a tensor'}")
                                    print(f"Original logits shape: {original_logits.shape}")
                                
                                # Apply softmax - handle different shapes properly
                                if len(original_logits.shape) == 3:  # [batch, seq, vocab]
                                    original_probs = original_logits.softmax(dim=-1)
                                    # For transformer models that return sequence outputs, take the last token
                                    original_probs = original_probs[:, -1, :]
                                elif len(original_logits.shape) == 2:  # [batch, vocab]
                                    original_probs = original_logits.softmax(dim=-1)
                                else:
                                    print(f"Unexpected logits shape: {original_logits.shape}")
                                    continue
                                
                                # Get residual
                                _, cache = self.model.run_with_cache(tok_seq)
                                final_residual = cache["ln_final.hook_normalized"]
                                
                                # Print debug info
                                if pos == chunk_start:  # Only print once per chunk
                                    print(f"Final residual shape: {final_residual.shape}")
                                    print(f"Modified weights shape: {self.modified_weights.shape}")
                                
                                # Apply modified weights - handle different shapes
                                if len(final_residual.shape) == 3:  # [batch, seq, hidden]
                                    # Use the last token's residual
                                    modified_logits = torch.matmul(final_residual[:, -1, :], self.modified_weights)
                                else:  # [batch, hidden]
                                    modified_logits = torch.matmul(final_residual, self.modified_weights)
                                    
                                modified_probs = modified_logits.softmax(dim=-1)
                                
                                # Print once per chunk for debugging
                                if pos == chunk_start:
                                    print(f"Original probs shape: {original_probs.shape}")
                                    print(f"Modified probs shape: {modified_probs.shape}")
                                
                                # Clean up intermediate tensors
                                del original_output, original_logits, cache, final_residual, modified_logits
                        except RuntimeError as e:
                            print(f"CUDA error at position {pos}, skipping: {e}")
                            continue
                        
                        # Calculate metrics carefully
                        try:
                            # Check if the rare token indices are within the vocabulary bounds
                            vocab_size_probs = original_probs.shape[1]
                            valid_indices = rare_token_indices[rare_token_indices < vocab_size_probs]
                            
                            if len(valid_indices) == 0:
                                print(f"Warning: No valid rare token indices found within vocabulary bounds ({vocab_size_probs})")
                                continue
                                
                            # Print first time for debugging
                            if pos == chunk_start:
                                print(f"Number of valid rare tokens: {len(valid_indices)} out of {len(rare_token_indices)}")
                            
                            # Calculate mean probability for rare tokens - direct indexing with valid indices
                            orig_rare_probs = original_probs[:, valid_indices].mean().item()
                            mod_rare_probs = modified_probs[:, valid_indices].mean().item()
                            
                            # Debug: Print some actual probability values to verify they're non-zero
                            if pos == chunk_start:
                                print(f"Sample of original rare token probs: {original_probs[:, valid_indices[:5]].flatten().tolist()}")
                                print(f"Sample of modified rare token probs: {modified_probs[:, valid_indices[:5]].flatten().tolist()}")
                            
                            # Calculate KL divergence more robustly
                            # Add small epsilon to avoid log(0) issues
                            epsilon = 1e-10
                            orig_log_probs = torch.log(original_probs + epsilon)
                            mod_log_probs = torch.log(modified_probs + epsilon)
                            
                            # KL(original || modified)
                            kl_div1 = torch.sum(original_probs * (orig_log_probs - mod_log_probs), dim=1).mean().item()
                            
                            # KL(modified || original) - reverse direction
                            kl_div2 = torch.sum(modified_probs * (mod_log_probs - orig_log_probs), dim=1).mean().item()
                            
                            # Jensen-Shannon divergence (symmetric)
                            js_div = 0.5 * (kl_div1 + kl_div2)
                            
                            # Print divergence values for debugging
                            if pos == chunk_start:
                                print(f"KL(orig||mod): {kl_div1:.6f}, KL(mod||orig): {kl_div2:.6f}, JS: {js_div:.6f}")
                                
                                # Check if distributions are nearly identical
                                max_prob_diff = torch.max(torch.abs(original_probs - modified_probs)).item()
                                print(f"Maximum probability difference: {max_prob_diff:.6f}")
                                
                                # Check distribution sparsity
                                orig_nonzero = (original_probs > epsilon).float().mean().item()
                                mod_nonzero = (modified_probs > epsilon).float().mean().item()
                                print(f"Non-zero probabilities: original {orig_nonzero:.2%}, modified {mod_nonzero:.2%}")
                            
                            # Top predictions with safe k
                            safe_k = min(5, original_probs.shape[1])
                            orig_top_indices = original_probs[0].topk(safe_k).indices.cpu().numpy()
                            mod_top_indices = modified_probs[0].topk(safe_k).indices.cpu().numpy()
                            
                            # Count rare tokens in top predictions
                            orig_rare_in_top = sum(1 for idx in orig_top_indices if idx in rare_token_set)
                            mod_rare_in_top = sum(1 for idx in mod_top_indices if idx in rare_token_set)
                            
                            # Get the actual token ID
                            token_id = tok_seq.flatten()[0].item() if tok_seq.numel() > 0 else -1
                            
                            # Collect all metrics
                            results.append({
                                "batch_idx": batch_idx,
                                "position": pos,
                                "token_id": token_id,
                                "original_rare_prob": orig_rare_probs,
                                "modified_rare_prob": mod_rare_probs,
                                "prob_increase": mod_rare_probs - orig_rare_probs,
                                "kl_divergence": kl_div1,  # Original PyTorch KL
                                "kl_divergence_orig_to_mod": kl_div1,
                                "kl_divergence_mod_to_orig": kl_div2,
                                "jensen_shannon_div": js_div,
                                "original_rare_in_top": orig_rare_in_top,
                                "modified_rare_in_top": mod_rare_in_top,
                            })
                        except Exception as e:
                            print(f"Error calculating metrics: {e}")
                            import traceback
                            traceback.print_exc()
                        
                        # Clean up references
                        del original_probs, modified_probs
                        
                    except Exception as e:
                        print(f"Error processing token at batch {batch_idx}, position {pos}: {e}")
                        import traceback
                        traceback.print_exc()
                        
                # Force cleanup after each chunk
                torch.cuda.empty_cache()
                gc.collect()
        
        # Check if we have any results
        if not results:
            print("Warning: No valid results were collected!")
            empty_df = pd.DataFrame(columns=[
                "batch_idx", "position", "token_id", 
                "original_rare_prob", "modified_rare_prob", "prob_increase",
                "kl_divergence", "kl_divergence_orig_to_mod", "kl_divergence_mod_to_orig",
                "jensen_shannon_div", "original_rare_in_top", "modified_rare_in_top"
            ])
            return empty_df
            
        # Print summary statistics of results
        df = pd.DataFrame(results)
        print("\nResults Summary:")
        print(f"Total tokens processed: {len(df)}")
        print(f"Average original rare probability: {df['original_rare_prob'].mean():.6f}")
        print(f"Average modified rare probability: {df['modified_rare_prob'].mean():.6f}")
        print(f"Average probability increase: {df['prob_increase'].mean():.6f}")
        print(f"Average KL divergence (orig->mod): {df['kl_divergence_orig_to_mod'].mean():.6f}")
        print(f"Average KL divergence (mod->orig): {df['kl_divergence_mod_to_orig'].mean():.6f}")
        print(f"Average JS divergence: {df['jensen_shannon_div'].mean():.6f}")
        
        return df

    def run_pipeline(self):
        """Run the complete null space scaling pipeline."""
        print("1. Computing unigram direction...")
        self.compute_unigram_direction()

        print("2. Finding null space of frequent tokens...")
        self.find_null_space()

        print("3. Analyzing null space impact...")
        # null space change
        analysis = self.analyze_null_space_impact()
        print(f"   Unigram projection ratio: {analysis['unigram_projection_ratio']:.4f}")
        print(f"   b_U projection ratio: {analysis['b_U_projection_ratio']:.4f}")

        print("4. Scaling rare token weights...")
        self.scale_rare_token_weights()

        print("5. Evaluate token probs ...")
        # tokenize data
        tokenized_data = self._tokenize_dataset()
        results = self.evaluate_token_prob(tokenized_data)
        print("Finish running pipeline")
        return results

    def _tokenize_dataset(self):
        data = " ".join(self.dataset["text"])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer(data, return_tensors="pt", padding=True, max_length=128, truncation=True)

