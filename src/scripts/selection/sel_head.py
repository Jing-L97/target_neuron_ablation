# Phase 1: Attention Head Influence on Plateau Neurons (Using k10.feather)
# Revised to be executable from command line with proper logging and file handling

import argparse
import logging
import pickle
import re
from collections import defaultdict
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for attention routing analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze attention head influence on plateau neurons using k10.feather data."
    )
    parser.add_argument(
        "--token_type",
        type=str,
        default="longtail",
        choices=["longtail", "common"],
        help="Token type to analyze: longtail or common",
    )
    parser.add_argument(
        "--model_name", type=str, default="EleutherAI/pythia-410m-deduped", help="Model name to analyze"
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with limited data")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    parser.add_argument("--sample_size", type=int, default=300, help="Sample size per attention head")
    parser.add_argument("--step", type=int, default=143000, help="Training step to analyze")
    return parser.parse_args()


class AttentionRoutingInfluenceAnalyzer:
    """Measures attention head influence on plateau neurons using complete ablation dataset
    Parses from k10.feather file with component_name and activation columns
    """

    def __init__(
        self,
        model,
        ablation_results_df,
        tokenized_data,
        plateau_neuron_data,
        unigram_distrib,
        token_type,
        longtail_threshold=0.001,
        device="cuda",
    ):
        self.model = model
        self.ablation_results_df = ablation_results_df  # Full k10.feather data
        self.tokenized_data = tokenized_data
        self.unigram_distrib = unigram_distrib
        self.longtail_threshold = longtail_threshold
        self.device = device
        self.token_type = token_type

        # Parse plateau neurons from CSV format
        self.plateau_neurons = self._parse_plateau_neurons(plateau_neuron_data)
        self.plateau_neuron_names = set([f"{layer}.{neuron}" for layer, neuron in self.plateau_neurons])

        # Preprocess ablation results for efficient lookup
        self.context_neuron_data = self._preprocess_ablation_data()

        # Generate target attention heads (adjust layer range based on model)
        self.target_heads = self._generate_target_heads()

        # Create longtail mask for token classification
        if self.token_type == "longtail":
            self.longtail_mask = (self.unigram_distrib < self.longtail_threshold).float()
        else:
            self.longtail_mask = (self.unigram_distrib >= self.longtail_threshold).float()

        self.target_token_ids = set(torch.nonzero(self.longtail_mask, as_tuple=True)[0].cpu().numpy())

        # Separate target and other token contexts
        self.target_contexts, self.other_contexts = self._separate_token_contexts()

        logger.info(f"Loaded {len(self.ablation_results_df)} total ablation records from k10.feather")
        logger.info(f"Unique contexts: {len(self.context_neuron_data)}")
        logger.info(f"Plateau neurons found: {len(self.plateau_neuron_names)}")
        logger.info(f"Target ({self.token_type}) contexts: {len(self.target_contexts)}")
        logger.info(f"Other contexts: {len(self.other_contexts)}")

    def _parse_plateau_neurons(self, plateau_data):
        """Parse plateau neuron data from CSV format to (layer, neuron) tuples"""
        final_step_data = plateau_data[plateau_data["step"] == 143000].iloc[0]
        neuron_strings = eval(final_step_data["top_neurons"])

        plateau_neurons = []
        for neuron_str in neuron_strings:
            layer, neuron = map(int, neuron_str.split("."))
            plateau_neurons.append((layer, neuron))

        logger.info(f"Parsed {len(plateau_neurons)} plateau neurons")
        return plateau_neurons

    def _preprocess_ablation_data(self):
        """Preprocess ablation data for efficient context-based lookup

        Returns:
            dict: {(batch, pos): {component_name: activation, ...}}

        """
        logger.info("Preprocessing ablation data for efficient lookup...")

        context_data = defaultdict(dict)

        for _, row in tqdm(self.ablation_results_df.iterrows(), desc="Processing ablation records"):
            context_key = (row["batch"], row["pos"])
            component_name = row["component_name"]
            activation = row["activation"]

            # Store activation for this component at this context
            context_data[context_key][component_name] = activation

            # Also store other useful context info (only once per context)
            if "context_info" not in context_data[context_key]:
                context_data[context_key]["context_info"] = {
                    "token_id": row["token_id"],
                    "str_tokens": row["str_tokens"],
                    "context": row["context"],
                    "batch": row["batch"],
                    "pos": row["pos"],
                    "entropy": row["entropy"],
                    "loss": row["loss"],
                }

        logger.info(f"Preprocessed {len(context_data)} unique contexts")
        return dict(context_data)

    def _generate_target_heads(self):
        """Generate attention head names based on model size"""
        target_heads = []

        # Adjust layer range based on model size
        if self.model.cfg.n_layers <= 6:  # Pythia-70M
            layer_range = range(3, 6)
        elif self.model.cfg.n_layers <= 24:  # Pythia-410M
            layer_range = range(20, 24)  # Focus on final layers
        else:  # Larger models
            layer_range = range(self.model.cfg.n_layers - 6, self.model.cfg.n_layers)

        for layer in layer_range:
            for head in range(self.model.cfg.n_heads):
                target_heads.append(f"L{layer}H{head}")

        logger.info(f"Generated {len(target_heads)} target attention heads from layers {list(layer_range)}")
        return target_heads

    def _separate_token_contexts(self, sample_size: int = 50000):
        """Separate contexts into target vs other token contexts
        Sample both groups for computational efficiency
        """
        logger.info(f"Separating {self.token_type} and other token contexts...")

        target_contexts = []
        other_contexts = []
        special_tokens = {0, 1, 2, 3}  # BOS, EOS, PAD, UNK

        # Get all unique contexts and sample if needed
        all_contexts = list(self.context_neuron_data.keys())
        if len(all_contexts) > sample_size:
            sampled_contexts = np.random.choice(all_contexts, size=sample_size, replace=False)
            logger.info(f"Sampled {len(sampled_contexts)} contexts from {len(all_contexts)} total")
        else:
            sampled_contexts = all_contexts
            logger.info(f"Using all {len(sampled_contexts)} contexts")

        for context_key in sampled_contexts:
            context_info = self.context_neuron_data[context_key]["context_info"]
            token_id = context_info["token_id"]

            # Skip special tokens
            if token_id in special_tokens:
                continue

            # Check if this context has plateau neuron data
            has_plateau_data = any(
                neuron_name in self.context_neuron_data[context_key] for neuron_name in self.plateau_neuron_names
            )

            if not has_plateau_data:
                continue

            # Classify as target or other
            if token_id in self.target_token_ids:
                target_contexts.append(context_key)
            else:
                other_contexts.append(context_key)

        logger.info(f"Found {len(target_contexts)} {self.token_type} contexts and {len(other_contexts)} other contexts")
        return target_contexts, other_contexts

    def measure_head_influence_on_plateaus(
        self, head_name: str, context_type: str = "target", sample_size: int = 300
    ) -> float:
        """Measure attention head's influence on plateau neuron activations

        Args:
            head_name: "L{layer}H{head}" format
            context_type: "target" or "other" - which token contexts to use
            sample_size: Number of contexts to test

        Returns:
            influence_score: Magnitude of activation change when head is ablated

        """
        # Select appropriate context set
        if context_type == "target":
            contexts = self.target_contexts
        elif context_type == "other":
            contexts = self.other_contexts
        else:
            raise ValueError("context_type must be 'target' or 'other'")

        if len(contexts) == 0:
            logger.warning(f"No {context_type} contexts available for {head_name}")
            return 0.0

        # Sample contexts for testing
        available_contexts = min(sample_size, len(contexts))
        sampled_context_keys = np.random.choice(contexts, size=available_contexts, replace=False)

        # Parse head info
        layer_idx, head_idx = self._parse_head_name(head_name)

        def zero_ablation_hook(value, hook, position, head):
            """Zero out attention head at specific position"""
            value[..., head, position, :] = 0.0
            value[..., head, position, 0] = 1.0  # Set attention to BOS token
            return value

        activation_changes = []

        # Process each context
        for context_key in tqdm(sampled_context_keys, desc=f"Testing {head_name} ({context_type})", leave=False):
            batch_idx, pos = context_key

            # Get original plateau activations from feather data
            original_activations = self._extract_original_plateau_activations(context_key)

            if len(original_activations) == 0:
                continue

            # Get input tokens
            try:
                inp = self.tokenized_data["tokens"][batch_idx].to(self.device)
            except (IndexError, KeyError):
                continue

            # Run with attention head ablated
            hooks = [
                (
                    tl_utils.get_act_name("pattern", layer_idx),
                    partial(zero_ablation_hook, position=pos, head=head_idx),
                )
            ]

            try:
                self.model.reset_hooks()
                with self.model.hooks(fwd_hooks=hooks):
                    _, ablated_cache = self.model.run_with_cache(inp.unsqueeze(0))

                    # Extract post-ablation plateau activations
                    ablated_activations = self._extract_ablated_plateau_activations(ablated_cache, pos)

                # Compute activation magnitude change
                if len(ablated_activations) == len(original_activations):
                    activation_delta = np.abs(ablated_activations - original_activations)
                    activation_changes.append(np.sum(activation_delta))

            except Exception as e:
                logger.error(f"Error processing {head_name} at context {context_key}: {e}")
                continue

        # Return mean influence
        return np.mean(activation_changes) if activation_changes else 0.0

    def _parse_head_name(self, head_name: str) -> tuple[int, int]:
        """Parse 'L22H6' -> (22, 6)"""
        pattern = r"L(\d+)H(\d+)"
        match = re.search(pattern, head_name)
        return tuple(map(int, match.groups()))

    def _extract_original_plateau_activations(self, context_key) -> np.ndarray:
        """Extract plateau neuron activations from preprocessed feather data

        Args:
            context_key: (batch, pos) tuple

        Returns:
            np.ndarray: Array of plateau neuron activations

        """
        activations = []
        context_data = self.context_neuron_data[context_key]

        for layer, neuron in self.plateau_neurons:
            neuron_name = f"{layer}.{neuron}"
            if neuron_name in context_data:
                activations.append(context_data[neuron_name])

        return np.array(activations)

    def _extract_ablated_plateau_activations(self, cache, position) -> np.ndarray:
        """Extract plateau neuron activations from ablated model cache"""
        activations = []
        for layer, neuron in self.plateau_neurons:
            try:
                activation = cache[tl_utils.get_act_name("post", layer)][0, position, neuron].cpu().numpy()
                activations.append(activation)
            except (IndexError, KeyError):
                continue
        return np.array(activations)

    def run_comparative_screening(self, sample_size: int = 300) -> dict[str, dict[str, float]]:
        """Screen attention heads on both target and other tokens for comparison

        Returns:
            results: Dict with structure {head_name: {"target": score, "other": score}}

        """
        results = {}

        logger.info(f"Running comparative screening on {len(self.target_heads)} attention heads...")
        logger.info(f"Sample size per head per token type: {sample_size}")

        for head_name in self.target_heads:
            try:
                # Test on target tokens
                target_influence = self.measure_head_influence_on_plateaus(
                    head_name, context_type="target", sample_size=sample_size
                )

                # Test on other tokens
                other_influence = self.measure_head_influence_on_plateaus(
                    head_name, context_type="other", sample_size=sample_size
                )

                results[head_name] = {
                    "target": target_influence,
                    "other": other_influence,
                    "selectivity": target_influence - other_influence,
                    "selectivity_ratio": target_influence / (other_influence + 1e-8),
                }

                logger.info(
                    f"{head_name}: {self.token_type}={target_influence:.6f}, other={other_influence:.6f}, "
                    f"selectivity={results[head_name]['selectivity']:.6f}"
                )

            except Exception as e:
                logger.error(f"Error processing {head_name}: {e}")
                results[head_name] = {"target": 0.0, "other": 0.0, "selectivity": 0.0, "selectivity_ratio": 1.0}

        return results

    def get_context_summary_stats(self):
        """Get summary statistics about the loaded contexts"""
        stats = {
            "total_unique_contexts": len(self.context_neuron_data),
            "target_contexts": len(self.target_contexts),
            "other_contexts": len(self.other_contexts),
            "plateau_neurons_tracked": len(self.plateau_neurons),
            "contexts_with_plateau_data": 0,
        }

        # Count contexts that have plateau neuron data
        for context_key in self.context_neuron_data:
            has_plateau_data = any(
                neuron_name in self.context_neuron_data[context_key] for neuron_name in self.plateau_neuron_names
            )
            if has_plateau_data:
                stats["contexts_with_plateau_data"] += 1

        return stats


# Data Pipeline for k10.feather
class Phase1DataPipeline:
    """Manages data loading from k10.feather ablation results"""

    def __init__(self, model_name: str, device: str, step: int = 143000):
        self.model_name = model_name
        self.step = step
        self.device = device

    def load_required_data(self):
        """Load all required data components from k10.feather and related files"""
        # 1. Load model
        model_handler = ModelHandler()
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

        # 2. Load k10.feather (complete ablation results)
        feather_path = settings.PATH.ablation_dir / f"longtail_50/{self.model_name}/{self.step}/500/k10.feather"
        ablation_results_df = pd.read_feather(feather_path)
        logger.info(f"Loaded {len(ablation_results_df)} ablation records from k10.feather")
        logger.info(f"Columns: {list(ablation_results_df.columns)}")

        # Verify expected columns exist
        required_cols = ["batch", "pos", "component_name", "activation", "token_id"]
        missing_cols = [col for col in required_cols if col not in ablation_results_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in feather file: {missing_cols}")

        # 3. Load plateau neuron data
        neuron_path = settings.PATH.neuron_dir / f"neuron/longtail_50/{self.model_name}/prob/boost/500_10.csv"
        plateau_df = pd.read_csv(neuron_path)
        logger.info("Finished loading neuron df")

        # 4. Load longtail threshold
        threshold_path = settings.PATH.ablation_dir / f"longtail_50/{self.model_name}/zipf_threshold_stats.json"
        longtail_threshold = load_tail_threshold_stat(threshold_path)
        logger.info(f"Loaded longtail threshold: {longtail_threshold}")

        # 5. Load unigram distribution
        unigram_distrib, _ = load_unigram(self.model_name, self.device)
        logger.info("Loaded unigram distribution")

        return {
            "model": model,
            "tokenizer": tokenizer,
            "ablation_results_df": ablation_results_df,
            "tokenized_data": tokenized_data,
            "plateau_data": plateau_df,
            "longtail_threshold": longtail_threshold,
            "unigram_distrib": unigram_distrib,
        }


# Main Phase 1 Execution with Control Experiment
def run_phase1_comparative_screening(token_type, model_name, sample_size, step, debug=False, resume=False):
    """Execute Phase 1: Comparative Attention Head Screening (Target vs Other tokens)

    Tests whether attention heads show selective influence on plateau neurons
    for target tokens vs general influence for all tokens
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Check for existing results if resume is requested
    out_dir = settings.PATH.attention_dir / f"longtail_50/{model_name}/{step}/500"
    result_file = out_dir / "head_comp.pkl"

    if resume and result_file.exists():
        logger.info(f"Resuming from existing results: {result_file}")
        with open(result_file, "rb") as f:
            return pickle.load(f)

    # Load all required data
    pipeline = Phase1DataPipeline(model_name=model_name, device=device, step=step)
    data_bundle = pipeline.load_required_data()

    # Limit data for debug mode
    if debug:
        logger.info("Running in debug mode - limiting data size")
        data_bundle["ablation_results_df"] = data_bundle["ablation_results_df"].head(1000)
        sample_size = min(sample_size, 50)

    # Initialize analyzer with complete dataset
    analyzer = AttentionRoutingInfluenceAnalyzer(
        model=data_bundle["model"],
        ablation_results_df=data_bundle["ablation_results_df"],
        tokenized_data=data_bundle["tokenized_data"],
        plateau_neuron_data=data_bundle["plateau_data"],
        unigram_distrib=data_bundle["unigram_distrib"],
        token_type=token_type,
        longtail_threshold=data_bundle["longtail_threshold"],
        device=device,
    )

    # Print summary stats
    stats = analyzer.get_context_summary_stats()
    logger.info("=== CONTEXT SUMMARY STATS ===")
    for key, value in stats.items():
        logger.info(f"{key}: {value}")

    # Run comparative screening
    comparative_results = analyzer.run_comparative_screening(sample_size=sample_size)

    # Analyze results for selectivity
    logger.info("=== COMPARATIVE SCREENING RESULTS ===")

    # Sort by selectivity (target influence - other influence)
    sorted_by_selectivity = sorted(comparative_results.items(), key=lambda x: x[1]["selectivity"], reverse=True)

    logger.info(f"\nTop 8 heads by SELECTIVITY ({token_type} - other influence):")
    for i, (head, scores) in enumerate(sorted_by_selectivity[:8]):
        logger.info(
            f"{i + 1:2d}. {head}: {token_type}={scores['target']:.6f}, "
            f"other={scores['other']:.6f}, selectivity={scores['selectivity']:.6f}"
        )

    # Sort by target influence only
    sorted_by_target = sorted(comparative_results.items(), key=lambda x: x[1]["target"], reverse=True)

    logger.info(f"\nTop 8 heads by {token_type.upper()} influence only:")
    for i, (head, scores) in enumerate(sorted_by_target[:8]):
        logger.info(
            f"{i + 1:2d}. {head}: {token_type}={scores['target']:.6f}, "
            f"other={scores['other']:.6f}, selectivity={scores['selectivity']:.6f}"
        )

    # Check for measurement artifacts
    layer_distribution_target = {}
    layer_distribution_selective = {}

    for head, scores in sorted_by_target[:8]:
        layer = int(head.split("H")[0][1:])
        layer_distribution_target[layer] = layer_distribution_target.get(layer, 0) + 1

    for head, scores in sorted_by_selectivity[:8]:
        layer = int(head.split("H")[0][1:])
        layer_distribution_selective[layer] = layer_distribution_selective.get(layer, 0) + 1

    logger.info(f"\nLayer distribution - {token_type} influence: {layer_distribution_target}")
    logger.info(f"Layer distribution - Selectivity: {layer_distribution_selective}")

    # Save results
    results = {
        "comparative_results": comparative_results,
        "top_selective_heads": [head for head, _ in sorted_by_selectivity[:8]],
        "top_target_heads": [head for head, _ in sorted_by_target[:8]],
        "context_stats": stats,
        "screening_metadata": {
            "n_heads_screened": len(analyzer.target_heads),
            "n_plateau_neurons": len(analyzer.plateau_neurons),
            "target_contexts": len(analyzer.target_contexts),
            "other_contexts": len(analyzer.other_contexts),
            "model_checkpoint": step,
            "token_type": token_type,
            "sample_size": sample_size,
            "debug_mode": debug,
        },
    }

    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(result_file, "wb") as f:
        pickle.dump(results, f)

    logger.info(f"\nPhase 1 complete. Results saved to {result_file}")

    return results


def main():
    """Main entry point that handles command line arguments."""
    args = parse_args()

    logger.info(f"Starting Phase 1 analysis with arguments: {vars(args)}")

    results = run_phase1_comparative_screening(
        token_type=args.token_type,
        model_name=args.model_name,
        sample_size=args.sample_size,
        step=args.step,
        debug=args.debug,
        resume=args.resume,
    )

    logger.info("Analysis completed successfully!")


if __name__ == "__main__":
    main()
