import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
import transformer_lens
from datasets import load_dataset

from neuron_analyzer import settings


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract word surprisal across different training steps.")
    parser.add_argument(
        "--freq_path",
        type=Path,
        default="src/unigram/pythia-unigrams.npy",
        help="Relative path to the target words",
    )

    parser.add_argument(
        "-m", "--model_name", type=str, default="EleutherAI/pythia-70m-deduped", help="Target model name"
    )
    parser.add_argument(
        "-n", "--neuron_file", type=str, default="500_10.csv", 
        help="Target model name"
    )
    parser.add_argument(
        "-a","--ablate", type=str, default="base", 
        choices=["base", "zero", "random"],
        help="Neuron options for computing surprisal"
        )
    parser.add_argument("--use_bos_only", action="store_true", help="use_bos_only if enabled")
    parser.add_argument("--debug", action="store_true", help="Compute the first few 5 lines if enabled")
    parser.add_argument("--resume", action="store_true", help="Resume from the existing checkpoint")
    return parser.parse_args()




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
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize the TokenNullSpaceScaler."""
        self.model = model
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

        self.unigram_direction = unigram_direction_vocab
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
        self.projection_matrix = self.null_space_basis @ self.null_space_basis.T

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

        for idx in tqdm.tqdm(rare_token_indices, desc="Scaling rare tokens"):
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

        # Project unigram direction onto null space
        unigram_projection = self.projection_matrix @ self.unigram_direction

        # Calculate projection magnitude
        projection_magnitude = unigram_projection.norm().item()
        original_magnitude = self.unigram_direction.norm().item()
        projection_ratio = projection_magnitude / original_magnitude

        # Analyze impact on b_U
        b_U_projection = self.projection_matrix @ self.model.b_U
        b_U_projection_magnitude = b_U_projection.norm().item()
        b_U_original_magnitude = self.model.b_U.norm().item()
        b_U_projection_ratio = b_U_projection_magnitude / b_U_original_magnitude

        self.analysis_results = {
            "unigram_projection_magnitude": projection_magnitude,
            "unigram_original_magnitude": original_magnitude,
            "unigram_projection_ratio": projection_ratio,
            "b_U_projection_magnitude": b_U_projection_magnitude,
            "b_U_original_magnitude": b_U_original_magnitude,
            "b_U_projection_ratio": b_U_projection_ratio,
        }

        return self.analysis_results

    def evaluate_token_prob(self, tokenized_data, k: int = 10) -> pd.DataFrame:
        """Evaluate the change in token probabilities after scaling the null space."""
        if self.modified_weights is None:
            self.scale_rare_token_weights()

        # Find rare tokens
        rare_token_mask = self.token_frequencies < self.rare_threshold
        rare_token_indices = torch.where(rare_token_mask)[0]

        # Sample some sequences
        random_sequence_indices = np.random.choice(len(tokenized_data), k, replace=False)

        results = []

        for batch_n in tqdm.tqdm(random_sequence_indices, desc="Evaluating probabilities"):
            tok_seq = tokenized_data["tokens"][batch_n].to(self.device)

            # Original model probabilities
            with torch.no_grad():
                original_logits = self.model(tok_seq.unsqueeze(0))[0]
                original_probs = original_logits.softmax(dim=-1)

                # Get the final residual output before unembedding
                _, cache = self.model.run_with_cache(tok_seq.unsqueeze(0))
                final_residual = cache["ln_final.hook_normalized"]

                # Apply the modified W_U
                modified_logits = torch.matmul(final_residual, self.modified_weights)
                modified_probs = modified_logits.softmax(dim=-1)

            # Average probability for rare tokens
            orig_rare_probs = original_probs[:, rare_token_indices].mean().item()
            mod_rare_probs = modified_probs[:, rare_token_indices].mean().item()

            # KL divergence
            kl = F.kl_div(modified_probs.log(), original_probs, reduction="batchmean").item()

            # Top predictions
            orig_top_tokens = original_probs.topk(5, dim=-1).indices
            mod_top_tokens = modified_probs.topk(5, dim=-1).indices

            # Count rare tokens in top predictions
            orig_rare_in_top = sum([torch.sum((orig_top_tokens == idx).float()).item() for idx in rare_token_indices])
            mod_rare_in_top = sum([torch.sum((mod_top_tokens == idx).float()).item() for idx in rare_token_indices])

            results.append(
                {
                    "batch_n": batch_n,
                    "original_rare_prob": orig_rare_probs,
                    "modified_rare_prob": mod_rare_probs,
                    "prob_increase": mod_rare_probs - orig_rare_probs,
                    "kl_divergence": kl,
                    "original_rare_in_top": orig_rare_in_top,
                    "modified_rare_in_top": mod_rare_in_top,
                }
            )

        return pd.DataFrame(results)

    def run_pipeline(self):
        """Run the complete null space scaling pipeline."""
        print("1. Computing unigram direction...")
        self.compute_unigram_direction()

        print("2. Finding null space of frequent tokens...")
        self.find_null_space()

        print("3. Analyzing null space impact...")
        analysis = self.analyze_null_space_impact()
        print(f"   Unigram projection ratio: {analysis['unigram_projection_ratio']:.4f}")
        print(f"   b_U projection ratio: {analysis['b_U_projection_ratio']:.4f}")

        print("4. Scaling rare token weights...")
        self.scale_rare_token_weights()

        print("5. Evaluate token probs ...")
        # tokenize data
        tokenized_data = self._tokenize_dataset(self.dataset, self.tokenizer)
        print("Finish tokenizing dataset")
        results = self.evaluate_token_prob(tokenized_data)
        print("Finish running pipeline")
        return results

    def _tokenize_dataset(self,dataset, tokenizer):
        return tokenizer(dataset["text"], truncation=True, max_length=128)






def main():
    # Set device
    args = parse_args()
    # Load GPT2-small
    print("Loading GPT2-small model...")
    model = transformer_lens.HookedTransformer.from_pretrained(args.model_name)
    tokenizer = model.tokenizer

    # Load a subset of the C4 dataset
    print("Loading dataset subset...")
    dataset = load_dataset("stas/c4-en-10k", split="train[:1000]")  # Use only first 1000 examples

    # load unifgram freq as a torch tensor
    token_frequencies = torch.tensor(np.load(settings.PATH.dataset_root / args.freq_path))
    print(f"Unigram freq has been loaded from {settings.PATH.dataset_root / args.freq_path}")
    # Tokenize a small subset for unigram statistics
    print("Tokenizing dataset...")

    # Scale rare token weights
    print("Scaling rare token weights...")
    freq_scaler = NullSpaceScaler(
        model,
        tokenizer,
        dataset,
        token_frequencies,
        top_k_percent= 0.05,
        rare_threshold = None,
        base_scaling_factor = 1.5,
        scaling_method= "linear",
        variance_threshold = 0.95,
        exponent= 1.0
    )
    results = freq_scaler.run_pipeline()
    print(results)


if __name__ == "__main__":
    main()
