# Phase 1: Revised Attention Routing Influence Measurement
# Adapting existing attention ablation infrastructure for gradient-free influence scoring

import logging
import pickle
import re
from functools import partial

import numpy as np
import pandas as pd
import torch
import transformer_lens.utils as tl_utils
from tqdm import tqdm

from neuron_analyzer import settings
from neuron_analyzer.load_util import load_tail_threshold_stat, load_unigram
from neuron_analyzer.model_util import ModelHandler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


class AttentionRoutingInfluenceAnalyzer:
    """Measures attention head influence on plateau neurons using longtail token filtering
    Adapts existing attention ablation code for routing analysis
    """

    def __init__(
        self,
        model,
        entropy_df,
        tokenized_data,
        plateau_neuron_data,
        unigram_distrib,
        longtail_threshold=0.001,
        device="cuda",
    ):
        self.model = model
        self.entropy_df = entropy_df
        self.tokenized_data = tokenized_data
        self.unigram_distrib = unigram_distrib
        self.longtail_threshold = longtail_threshold
        self.device = device

        # Parse plateau neurons from CSV format ('5.1326' -> layer=5, neuron=1326)
        self.plateau_neurons = self._parse_plateau_neurons(plateau_neuron_data)

        # Generate target attention heads (layers 3-5 for Pythia-70M)
        self.target_heads = self._generate_target_heads()

        # Filter entropy_df to longtail contexts only
        self.filtered_entropy_df = self._filter_to_longtail_contexts()

        print(f"Original entropy_df size: {len(entropy_df)}")
        print(f"Longtail-filtered size: {len(self.filtered_entropy_df)}")

    def _parse_plateau_neurons(self, plateau_data):
        """Parse plateau neuron data from CSV format to (layer, neuron) tuples"""
        # Extract from last step (143000) in your data
        final_step_data = plateau_data[plateau_data["step"] == 143000].iloc[0]
        neuron_strings = eval(final_step_data["top_neurons"])  # Parse string list

        plateau_neurons = []
        for neuron_str in neuron_strings:
            layer, neuron = map(int, neuron_str.split("."))
            plateau_neurons.append((layer, neuron))

        return plateau_neurons

    def _generate_target_heads(self):
        """Generate attention head names for layers 3-5 (Pythia-70M)"""
        target_heads = []
        for layer in range(3, min(6, self.model.cfg.n_layers)):
            for head in range(self.model.cfg.n_heads):
                target_heads.append(f"L{layer}H{head}")
        return target_heads

    def _filter_to_longtail_contexts(self):
        """Filter entropy_df to only include contexts with longtail tokens using mask approach"""
        print("Filtering entropy_df to longtail contexts using mask approach...")

        # Create longtail mask (same as neuron ablation code)
        self.longtail_mask = (self.unigram_distrib < self.longtail_threshold).float()
        longtail_token_ids = torch.nonzero(self.longtail_mask, as_tuple=True)[0]
        longtail_token_set = set(longtail_token_ids.cpu().numpy())

        print(f"Total longtail tokens identified: {len(longtail_token_set)}")
        print(f"Longtail threshold: {self.longtail_threshold}")

        longtail_indices = []
        special_tokens = {0, 1, 2, 3}  # Common special token IDs

        for idx, row in self.entropy_df.iterrows():
            try:
                # Get token at the measured position
                token_at_position = self.tokenized_data["tokens"][row.batch][row.pos]

                # Ensure we get the scalar token ID
                if hasattr(token_at_position, "item"):
                    token_id = token_at_position.item()
                else:
                    token_id = int(token_at_position)

                # Skip special tokens
                if token_id in special_tokens:
                    continue

                # Check if token is in longtail set
                if token_id in longtail_token_set:
                    longtail_indices.append(idx)

            except (IndexError, KeyError, RuntimeError) as e:
                # Skip malformed entries
                if idx < 10:  # Only print first few errors to avoid spam
                    print(f"Warning: Error processing row {idx}: {e}")
                continue

        filtered_df = self.entropy_df.loc[longtail_indices].copy()

        print(f"Longtail contexts found: {len(filtered_df)}")
        print(f"Filtering ratio: {len(filtered_df) / len(self.entropy_df):.3f}")

        # Debug information if very few contexts found
        if len(filtered_df) < 100:
            print(f"Warning: Very few longtail contexts found ({len(filtered_df)})")
            print("Sample token analysis:")
            sample_longtail_ids = list(longtail_token_set)[:10]
            print(f"First 10 longtail token IDs: {sample_longtail_ids}")

            # Check a few sample rows
            for idx, row in self.entropy_df.head(20).iterrows():
                try:
                    token_id = self.tokenized_data["tokens"][row.batch][row.pos].item()
                    is_longtail = token_id in longtail_token_set
                    freq = self.unigram_distrib[token_id].item() if token_id < len(self.unigram_distrib) else -1
                    print(f"  Row {idx}: token_id={token_id}, freq={freq:.6f}, longtail={is_longtail}")
                    if idx > 10 and any(
                        self.tokenized_data["tokens"][row.batch][row.pos].item() in longtail_token_set
                        for _, row in self.entropy_df.head(20).iterrows()
                    ):
                        break
                except Exception as e:
                    print(f"  Row {idx}: Error - {e}")

        return filtered_df

    def measure_head_influence_on_plateaus(self, head_name: str, sample_size: int = 500) -> float:
        """Measure single attention head's influence on plateau neuron activations
        Uses longtail-filtered contexts only

        Args:
            head_name: "L{layer}H{head}" format
            sample_size: Number of longtail contexts to test

        Returns:
            influence_score: Magnitude of activation change when head is ablated

        """
        # Sample from pre-filtered longtail contexts
        available_contexts = min(sample_size, len(self.filtered_entropy_df))
        sampled_contexts = self.filtered_entropy_df.sample(n=available_contexts, replace=False)

        if len(sampled_contexts) < 10:
            print(f"Warning: Only {len(sampled_contexts)} longtail contexts available for {head_name}")
            return 0.0

        # Storage for activation changes
        activation_changes = []

        # Parse head info
        layer_idx, head_idx = self._parse_head_name(head_name)

        def zero_ablation_hook(value, hook, position, head):
            """Zero out attention head at specific position"""
            value[..., head, position, :] = 0.0
            value[..., head, position, 0] = 1.0  # Set attention to BOS token
            return value

        # Process each context
        for _, row in tqdm(sampled_contexts.iterrows(), desc=f"Testing {head_name}", leave=False):
            inp = self.tokenized_data["tokens"][row.batch].to(self.device)
            position = row.pos

            # Verify this is indeed a longtail token
            token_at_pos = inp[position]
            # Extract scalar token ID
            token_id = token_at_pos.item() if hasattr(token_at_pos, "item") else int(token_at_pos)

            if token_id >= len(self.unigram_distrib):
                continue  # Skip if token ID out of vocab range

            token_freq = self.unigram_distrib[token_id]
            freq_value = token_freq.item() if hasattr(token_freq, "item") else float(token_freq)

            if freq_value >= self.longtail_threshold:
                continue  # Skip if not longtail

            # Get original plateau activations (from pre-computed entropy_df)
            original_activations = self._extract_original_plateau_activations(row)

            if len(original_activations) == 0:
                continue  # Skip if no plateau neurons available

            # Run with attention head ablated
            hooks = [
                (
                    tl_utils.get_act_name("pattern", layer_idx),
                    partial(zero_ablation_hook, position=position, head=head_idx),
                )
            ]

            try:
                self.model.reset_hooks()
                with self.model.hooks(fwd_hooks=hooks):
                    _, ablated_cache = self.model.run_with_cache(inp.unsqueeze(0))

                    # Extract post-ablation plateau activations
                    ablated_activations = self._extract_ablated_plateau_activations(ablated_cache, position)

                # Compute activation magnitude change
                if len(ablated_activations) == len(original_activations):
                    activation_delta = np.abs(ablated_activations - original_activations)
                    activation_changes.append(np.sum(activation_delta))  # Sum across plateau neurons

            except Exception as e:
                print(f"Error processing {head_name} at position {position}: {e}")
                continue

        # Return mean influence across longtail contexts
        if len(activation_changes) == 0:
            return 0.0

        return np.mean(activation_changes)

    def _parse_head_name(self, head_name: str) -> tuple[int, int]:
        """Parse 'L3H5' -> (3, 5)"""
        pattern = r"L(\d+)H(\d+)"
        match = re.search(pattern, head_name)
        return tuple(map(int, match.groups()))

    def _extract_original_plateau_activations(self, row) -> np.ndarray:
        """Extract plateau neuron activations from entropy_df row"""
        activations = []
        for layer, neuron in self.plateau_neurons:
            col_name = f"{layer}.{neuron}_activation"
            if col_name in row.index:
                activations.append(row[col_name])
            else:
                # Skip missing plateau neurons
                continue
        return np.array(activations)

    def _extract_ablated_plateau_activations(self, cache, position) -> np.ndarray:
        """Extract plateau neuron activations from ablated model cache"""
        activations = []
        for layer, neuron in self.plateau_neurons:
            try:
                activation = cache[tl_utils.get_act_name("post", layer)][0, position, neuron].cpu().numpy()
                activations.append(activation)
            except (IndexError, KeyError):
                # Skip if neuron index out of range
                continue
        return np.array(activations)

    def run_full_screening(self, sample_size: int = 300) -> dict[str, float]:
        """Screen all target attention heads for influence on plateau neurons (longtail contexts only)

        Args:
            sample_size: Longtail contexts per head to test (reduced due to filtering)

        Returns:
            head_influence_scores: dict mapping head names to influence scores

        """
        influence_scores = {}

        print(f"Screening {len(self.target_heads)} attention heads on longtail contexts...")
        print(f"Using {len(self.plateau_neurons)} plateau neurons as targets")
        print(f"Available longtail contexts: {len(self.filtered_entropy_df)}")
        print(f"Sample size per head: {sample_size}")

        for head_name in self.target_heads:
            try:
                influence = self.measure_head_influence_on_plateaus(head_name, sample_size)
                influence_scores[head_name] = influence
                print(f"{head_name}: {influence:.6f}")
            except Exception as e:
                print(f"Error processing {head_name}: {e}")
                influence_scores[head_name] = 0.0

        return influence_scores


# Data Pipeline for Phase 1
class Phase1DataPipeline:
    """Manages data loading and preprocessing for attention routing analysis"""

    def __init__(self, model_name: str, device: str, step: int = 143000):
        self.model_name = model_name
        self.step = step
        self.device = device

    def load_required_data(self):
        """Load all required data components

        Returns:
            data_bundle: dict containing all necessary components

        """
        # 1. Load model (reuse existing function)
        # save_path.parent.mkdir(parents=True, exist_ok=True)

        # initlize the model handler class
        model_handler = ModelHandler()

        # Load model and tokenizer for specific step
        model, tokenizer = model_handler.load_model_and_tokenizer(
            step=self.step,
            model_name=self.model_name,
            hf_token_path=settings.PATH.unigram_dir / "hf_token.txt",
            device=self.device,
        )
        logger.info("Finished loading model and tokenizer")
        # Load and process dataset
        tokenized_data, _ = model_handler.tokenize_data(
            dataset="stas/c4-en-10k",
            data_range_start=0,
            data_range_end=500,
            seed=42,
            get_df=True,
        )

        logger.info("Finished tokenizing data")

        # 2. Load entropy_df (preprocessed activations and contexts)
        entropy_df_path = (
            settings.PATH.ablation_dir / "longtail_50/EleutherAI/pythia-70m-deduped/143000/500/entropy_df.csv"
        )

        entropy_df = pd.read_csv(entropy_df_path)

        logger.info("Finished loading entropy df")
        # 4. Load plateau neuron data (from your CSV)
        neuron_path = (
            settings.PATH.neuron_dir / "neuron/longtail_50/EleutherAI/pythia-70m-deduped/prob/boost/500_10.csv"
        )

        plateau_df = pd.read_csv(neuron_path)  # Your provided data
        logger.info("Finished loading neuron df")
        # 5. load longtail threshold
        threshold_path = (
            settings.PATH.ablation_dir / "longtail_50/EleutherAI/pythia-70m-deduped/zipf_threshold_stats.json"
        )
        longtail_threshold = load_tail_threshold_stat(threshold_path)
        logger.info(f"Loaded longtail threshold: {longtail_threshold}")

        # load unigram
        unigram_distrib, _ = load_unigram(self.model_name, self.device)

        return {
            "model": model,
            "tokenizer": tokenizer,
            "entropy_df": entropy_df,
            "tokenized_data": tokenized_data,
            "plateau_data": plateau_df,
            "longtail_threshold": longtail_threshold,
            "unigram_distrib": unigram_distrib,
        }


def run_phase1_longtail_screening():
    """Execute Phase 1: Attention Head Influence Screening (Longtail contexts only)"""
    # Load all required data
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Use device: {device}")
    pipeline = Phase1DataPipeline(model_name="EleutherAI/pythia-70m-deduped", device=device)  # Adjust model name
    data_bundle = pipeline.load_required_data()

    # Initialize influence analyzer with longtail filtering
    analyzer = AttentionRoutingInfluenceAnalyzer(
        model=data_bundle["model"],
        entropy_df=data_bundle["entropy_df"],
        tokenized_data=data_bundle["tokenized_data"],
        plateau_neuron_data=data_bundle["plateau_data"],
        unigram_distrib=data_bundle["unigram_distrib"],
        longtail_threshold=data_bundle["longtail_threshold"],  # Same threshold as neuron ablation
        device=device,
    )

    # Run screening experiment on longtail contexts only
    influence_scores = analyzer.run_full_screening(sample_size=300)

    # Rank and select top candidates
    sorted_heads = sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)
    top_candidates = [head for head, score in sorted_heads[:8]]

    # Print results summary
    logger.info("\n=== LONGTAIL SCREENING RESULTS ===")
    logger.info("Top 8 routing heads (longtail contexts):")
    for i, (head, score) in enumerate(sorted_heads[:8]):
        logger.info(f"{i + 1:2d}. {head}: {score:.6f}")

    logger.info("\nBottom 3 heads for comparison:")
    for head, score in sorted_heads[-3:]:
        logger.info(f"    {head}: {score:.6f}")

    # Prepare results
    results = {
        "head_influence_scores": influence_scores,
        "top_candidates": top_candidates,
        "screening_metadata": {
            "n_heads_screened": len(analyzer.target_heads),
            "n_plateau_neurons": len(analyzer.plateau_neurons),
            "sample_size_per_head": 300,
            "total_contexts_processed": len(analyzer.target_heads) * 300,
            "longtail_threshold": 0.001,
            "longtail_contexts_available": len(analyzer.filtered_entropy_df),
            "original_contexts_total": len(data_bundle["entropy_df"]),
            "longtail_filtering_ratio": len(analyzer.filtered_entropy_df) / len(data_bundle["entropy_df"]),
            "model_checkpoint": 143000,
            "target_layers": [3, 4, 5],
        },
    }

    out_dir = settings.PATH.attention_dir / "longtail_50/EleutherAI/pythia-70m-deduped/143000/500"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save for Phase 2
    with open(out_dir / "head.pkl", "wb") as f:
        pickle.dump(results, f)

    logger.info("\nPhase 1 complete. Results saved to phase1_longtail_attention_routing_results.pkl")
    logger.info(f"Top routing heads (longtail-specific): {top_candidates}")

    return results


run_phase1_longtail_screening()

# Expected computational profile:
# - Contexts per head: 500
# - Heads to screen: ~80 (layers 20-30, assuming 8 heads per layer)
# - Total forward passes: ~40,000
# - Forward pass time: ~50ms per pass with ablation
# - Total compute time: ~30-40 minutes
# - Memory requirement: Minimal (processing one context at a time)
