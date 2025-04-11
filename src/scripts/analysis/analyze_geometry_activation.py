import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoTokenizer

from neuron_analyzer import settings
from neuron_analyzer.analysis.geometry import NeuronGeometricAnalyzer
from neuron_analyzer.eval.surprisal import StepSurprisalExtractor
from neuron_analyzer.model_util import NeuronLoader, StepConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def analyze_rare_token_activations(model, prompts, rare_tokens, layer_idx=-1, top_k=5):
    """Analyze activation patterns that lead to rare token predictions.

    Args:
        model: HuggingFace transformer model
        prompts: List of prompt prefixes that will be used to generate completions
        rare_tokens: List of rare token IDs we're interested in analyzing
        layer_idx: Layer to extract activations from (-1 for last layer)
        top_k: Number of top neurons to analyze

    """
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)

    # Find the correct layer index
    if layer_idx < 0:
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layer_idx = len(model.transformer.h) + layer_idx
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            layer_idx = len(model.model.layers) + layer_idx

    # Storage for activations that led to rare tokens
    rare_token_activations = []
    other_token_activations = []
    generated_tokens = []

    for prompt in prompts:
        # Tokenize the prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        # Storage for this prompt's activations
        activation_data = None

        # Hook to capture activations
        def activation_hook(module, input, output):
            nonlocal activation_data
            activation_data = output.detach()

        # Register hook on the activation function
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            hook_handle = model.transformer.h[layer_idx].mlp.act.register_forward_hook(activation_hook)
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            try:
                hook_handle = model.model.layers[layer_idx].feed_forward.act.register_forward_hook(activation_hook)
            except:
                hook_handle = model.model.layers[layer_idx].mlp.act_fn.register_forward_hook(activation_hook)

        # Generate a token (use greedy decoding for simplicity)
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()

        # Remove hook
        hook_handle.remove()

        # Get activation for the last token position (which determines the next token)
        last_token_activation = activation_data[0, -1, :]

        # Store based on whether the generated token is in our rare tokens list
        generated_tokens.append(next_token_id)
        if next_token_id in rare_tokens:
            rare_token_activations.append(last_token_activation)
        else:
            other_token_activations.append(last_token_activation)

    # Convert to tensors
    if rare_token_activations:
        rare_activations = torch.stack(rare_token_activations)
        print(f"Collected {len(rare_token_activations)} activations for rare tokens")
    else:
        print("No rare tokens were generated. Try more or different prompts.")
        return None

    if other_token_activations:
        other_activations = torch.stack(other_token_activations)

    # Analysis of the activation vectors
    results = {}

    # 1. Find neurons that are most active for rare tokens
    mean_rare_activations = rare_activations.mean(dim=0)
    if other_token_activations:
        mean_other_activations = other_activations.mean(dim=0)
        # Find neurons with highest differential activation
        activation_diff = mean_rare_activations - mean_other_activations
        top_neurons = torch.topk(activation_diff, top_k).indices.tolist()
        results["distinctive_neurons"] = top_neurons
        results["activation_diffs"] = activation_diff[top_neurons].tolist()

    # 2. Analyze variance of activations for rare tokens
    activation_variance = rare_activations.var(dim=0)
    top_variance_neurons = torch.topk(activation_variance, top_k).indices.tolist()
    results["high_variance_neurons"] = top_variance_neurons

    # 3. Perform dimensionality reduction to visualize activation space
    all_activations = torch.cat([rare_activations, other_activations]) if other_token_activations else rare_activations
    all_labels = (
        ["Rare" for _ in range(len(rare_activations))] + ["Other" for _ in range(len(other_token_activations))]
        if other_token_activations
        else ["Rare" for _ in range(len(rare_activations))]
    )

    # Use PCA for initial dimensionality reduction
    if all_activations.shape[0] > 2:
        pca = PCA(n_components=min(50, all_activations.shape[0] - 1))
        reduced_activations = pca.fit_transform(all_activations.cpu().numpy())

        # Then use t-SNE for visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, all_activations.shape[0] - 1))
        tsne_results = tsne.fit_transform(reduced_activations)

        # Store for visualization
        results["visualization_coords"] = tsne_results
        results["labels"] = all_labels

    # 4. Analyze geometric properties of the rare token activation subspace
    if len(rare_activations) > 1:
        # Calculate centroid of rare token activations
        centroid = rare_activations.mean(dim=0)

        # Calculate distances from centroid
        distances = torch.norm(rare_activations - centroid, dim=1)
        results["centroid_distances"] = distances.tolist()

        # Calculate pairwise distances between rare token activations
        pairwise_distances = torch.cdist(rare_activations, rare_activations)
        results["pairwise_distances"] = pairwise_distances.tolist()

        # Measure spread/compactness of the rare token cluster
        results["cluster_radius"] = distances.max().item()
        results["cluster_density"] = 1 / (distances.mean().item() + 1e-5)

    return results


# Example usage with prompts that might generate rare tokens
rare_token_ids = [30000, 40000, 50000]  # Replace with actual rare token IDs
prompts = [
    "In cryptography, the algorithm known as",
    "The scientific name for this rare species is",
    "The chemical compound is represented by the formula",
    "In ancient Sumerian, this word translates to",
    "The technical term for this phenomenon is",
]

results = analyze_rare_token_activations(model, prompts, rare_token_ids)

# Visualize the results
if results and "visualization_coords" in results:
    plt.figure(figsize=(10, 8))
    coords = results["visualization_coords"]
    labels = results["labels"]

    # Plot points colored by label
    for label, color in zip(["Rare", "Other"], ["red", "blue"], strict=False):
        mask = [l == label for l in labels]
        plt.scatter(coords[mask, 0], coords[mask, 1], c=color, label=label, alpha=0.7)

    plt.title("t-SNE Visualization of Activation Vectors")
    plt.legend()
    plt.show()

    # Print the distinctive neurons
    if "distinctive_neurons" in results:
        print(f"Top neurons distinctive for rare tokens: {results['distinctive_neurons']}")
        print(f"Their activation differences: {results['activation_diffs']}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compute neuron activation directions.")
    parser.add_argument(
        "-m", "--model_name", type=str, default="EleutherAI/pythia-70m-deduped", help="Target model name"
    )
    parser.add_argument("-n", "--neuron_file", type=str, default="500_50.csv", help="Target model name")
    parser.add_argument("--neuron_num", type=int, default=3, help="Target neuron num")
    parser.add_argument(
        "--vector",
        type=str,
        default="longtail",
        choices=["mean", "longtail"],
        help="Differnt ablation model for freq vectors",
    )
    parser.add_argument("--interval", type=int, default=10, help="Checkpoint interval sampling")
    parser.add_argument("--debug", action="store_true", help="Compute the first few 5 lines if enabled")
    parser.add_argument("--resume", action="store_true", help="Resume from the existing checkpoint")
    return parser.parse_args()


def get_filename(neuron_file: str, neuron_num: int) -> str:
    """Insert the neuron num to the saved file.  e.g.500_10.csv"""
    if neuron_num == 0:
        return neuron_file
    file_prefix = neuron_file.split("_")[0]
    file_suffix = neuron_file.split(".")[1]
    return f"{file_prefix}_{neuron_num}.{file_suffix}"


def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ###################################
    # load materials and paths
    ###################################
    raw_filename = get_filename(args.neuron_file, args.neuron_num)
    result_dir = settings.PATH.direction_dir / "geometry" / args.model_name
    filename = f"{Path(raw_filename).stem}.debug" if args.debug else raw_filename
    subspace_file = result_dir / "subspace" / filename
    orthogonality_file = result_dir / "orthogonality" / filename
    if args.resume and subspace_file.is_file() and orthogonality_file.is_file():
        logger.info("Target files already exist, skipping processing as resume is enabled")
        sys.exit(0)

    subspace_file.parent.mkdir(parents=True, exist_ok=True)
    orthogonality_file.parent.mkdir(parents=True, exist_ok=True)

    # load neuron indices
    neuron_loader = NeuronLoader()
    boost_step_ablations, layer_num = neuron_loader.load_neuron_dict(
        settings.PATH.result_dir / "token_freq" / "boost" / args.vector / args.model_name / args.neuron_file,
        key_col="step",
        value_col="top_neurons",
        top_n=args.neuron_num,
        random_base=False,
    )
    suppress_step_ablations, layer_num = neuron_loader.load_neuron_dict(
        settings.PATH.result_dir / "token_freq" / "suppress" / args.vector / args.model_name / args.neuron_file,
        key_col="step",
        value_col="top_neurons",
        top_n=args.neuron_num,
        random_base=False,
    )

    ###################################
    # Initialize classes
    ###################################

    # Initialize configuration with all Pythia checkpoints
    steps_config = StepConfig(debug=args.debug, interval=args.interval)

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
        try:
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
        except:
            logger.info(f"Something wrong with step {step}")
    # Save results even if some checkpoints failed
    subspace_df.to_csv(subspace_file)
    logger.info(f"Subspace results saved to: {subspace_file}")
    orthogonality_df.to_csv(orthogonality_file)
    logger.info(f"Orthogonality results saved to: {orthogonality_file}")
    logger.info(f"Processed {len(steps_config.steps)} checkpoints successfully")


if __name__ == "__main__":
    main()
