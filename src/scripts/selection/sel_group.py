#!/usr/bin/env python
import argparse
import logging
from pathlib import Path

import neel.utils as nutils
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformer_lens import utils

from neuron_analyzer import settings
from neuron_analyzer.ablation.abl_util import (
    filter_entropy_activation_df,
    load_model_from_tl_name,
)
from neuron_analyzer.analysis.freq import ZipfThresholdAnalyzer
from neuron_analyzer.load_util import JsonProcessor
from neuron_analyzer.selection.group import GroupAblationAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Seach neuron groups across different training steps.")

    parser.add_argument("-m", "--model", type=str, default="EleutherAI/pythia-70m-deduped", help="Target model name")
    parser.add_argument("--vector", choices=["mean", "longtail"], default="longtail")
    parser.add_argument(
        "--effect", type=str, choices=["boost", "suppress"], default="suppress", help="boost or suppress long-tail prob"
    )
    parser.add_argument("--top_n", type=int, default=10, help="use_bos_only if enabled")
    parser.add_argument("--data_range_end", type=int, default=500, help="the selected datarange")
    parser.add_argument("--k", type=int, default=10, help="use_bos_only if enabled")
    return parser.parse_args()


#######################################################################################################
# Fucntions applied in the main scripts
#######################################################################################################


class NeuronAblationProcessor:
    """Class to run experiments of neuron group search."""

    def __init__(self, args, device, logger: logging.Logger | None = None):
        """Initialize the ablation processor with configuration."""
        # Set up logger
        self.logger = logger or logging.getLogger(__name__)

        # Initialize parameters from args
        self.args = args
        self.seed: int = args.seed
        self.device: str = device

        # Initialize random seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.set_grad_enabled(False)

    def get_tail_threshold_stat(self, unigram_distrib, save_path: Path) -> tuple[float | None, dict | None]:
        """Calculate threshold for long-tail ablation mode."""
        if self.args.ablation_mode == "longtail":
            analyzer = ZipfThresholdAnalyzer(
                unigram_distrib=unigram_distrib,
                window_size=self.args.window_size,
                tail_threshold=self.args.tail_threshold,
                apply_elbow=self.args.apply_elbow,
            )
            longtail_threshold, threshold_stats = analyzer.get_tail_threshold()
            JsonProcessor.save_json(threshold_stats, save_path / "zipf_threshold_stats.json")
            self.logger.info(f"Saved threshold statistics to {save_path}/zipf_threshold_stats.json")
            return longtail_threshold
        # Not in longtail mode, use default threshold
        return None

    def process_single_step(self, step: int, unigram_distrib, longtail_threshold, save_path: Path) -> None:
        """Process a single step with the given configuration."""
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Load model and tokenizer for specific step
        model, tokenizer = self.load_model_and_tokenizer(step)

        self.logger.info("Finished loading model and tokenizer")
        # Load and process dataset
        data = load_dataset(self.args.dataset, split="train")
        first_1k = data.select([i for i in range(self.args.data_range_start, self.args.data_range_end)])
        tokenized_data = utils.tokenize_and_concatenate(first_1k, tokenizer, max_length=256, column_name="text")
        tokenized_data = tokenized_data.shuffle(self.args.seed)
        token_df = nutils.make_token_df(tokenized_data["tokens"], model=model)

        self.logger.info("Finished tokenizing data")

        # Setup neuron indices
        entropy_neuron_layer = model.cfg.n_layers - 1
        if self.args.neuron_range is not None:
            start, end = map(int, self.args.neuron_range.split("-"))
            all_neuron_indices = list(range(start, end))
        else:
            all_neuron_indices = list(range(model.cfg.d_mlp))

        all_neurons = [f"{entropy_neuron_layer}.{i}" for i in all_neuron_indices]
        self.logger.info("Loaded all the neurons")

        if self.args.dry_run:
            all_neurons = all_neurons[:10]

        # Ablate the dimensions
        model.set_use_attn_result(False)

        analyzer = GroupAblationAnalyzer(
            model=model,  # Your model
            unigram_distrib=unigram_distrib,  # Unigram distribution
            tokenized_data=tokenized_data,  # Your tokenized data
            entropy_df=entropy_df,  # DataFrame with entropy information
            components_to_ablate=["5.123", "5.456", "5.789"],  # Initial components (layer.neuron format)
            device="cuda" if torch.cuda.is_available() else "cpu",
            ablation_mode="mean",
            cache_dir=Path("./ablation_cache"),  # Directory to store cache files
        )

        # 3. Simple search for a single optimal group
        search_result = analyzer.run_neuron_group_search(
            neurons=all_layer5_neurons,
            target_size=10,  # Looking for 10 neurons
            method="hybrid",  # Using hybrid search method
            resume_path="./previous_search.pkl",  # Optional: resume from previous run
        )

        # Print the search result
        print(f"Best neuron group: {search_result.neurons}")
        print(f"Mediation score: {search_result.mediation:.4f}")
        print(f"KL divergence: {search_result.kl_divergence:.4f}")

        # 4. Run a more comprehensive experiment with multiple group sizes and methods
        experiment_results = analyzer.group_ablation_experiment(
            neurons=all_layer5_neurons,
            target_sizes=[5, 10, 15, 20],  # Try different group sizes
            search_methods=["progressive_beam", "hybrid"],  # Try different search methods
            n_batches=5,  # Number of batches to ablate for final evaluation
            resume_dir=Path("./experiment_results"),  # Directory for resumable experiments
        )

        # 5. Compare the results across different methods and sizes
        comparison = analyzer.compare_neuron_groups(experiment_results)
        print("\nComparison of different neuron groups:")
        print(comparison)

        # 6. Detailed ablation of the best found group
        best_method = comparison.iloc[0]["method"]
        best_size = comparison.iloc[0]["size"]
        best_result, ablation_df = experiment_results[best_method][best_size]

        self.logger.info("Finished ablations!")

        # Process and save results
        self._save_results(results, tokenizer, step, save_path)

    def load_model_and_tokenizer(self, step: int) -> tuple[t.Any, t.Any]:
        """Load model and tokenizer for processing."""
        # Load HF token
        with open(settings.PATH.unigram_dir / self.args.hf_token_path) as f:
            hf_token = f.read()

        # Load model and tokenizer
        model, tokenizer = load_model_from_tl_name(
            self.args.model, self.device, step=step, cache_dir=settings.PATH.model_dir, hf_token=hf_token
        )
        model = model.to(self.device)
        model.eval()
        return model, tokenizer

    def _save_results(
        self,
        results: dict,
        tokenizer,
        step: int,
        save_path: Path,
    ) -> None:
        """Process and save ablation results."""
        final_df = pd.concat(results.values())
        final_df = filter_entropy_activation_df(
            final_df.reset_index(), model_name=self.args.model, tokenizer=tokenizer, start_pos=3, end_pos=-1
        )

        # Save results
        final_df = final_df.reset_index(drop=True)
        output_path = save_path / f"k{self.args.k}.feather"
        final_df.to_feather(output_path)
        self.logger.info(f"Saved results for step {step} to {output_path}")

    def get_save_dir(self):
        """Get the savepath based on current configurations."""
        if self.args.ablation_mode == "longtail" and self.args.apply_elbow:
            ablation_name = "longtail_elbow"
        if self.args.ablation_mode == "longtail" and not self.args.apply_elbow:
            ablation_name = f"longtail_{self.args.tail_threshold}"
        else:
            ablation_name = self.args.ablation_mode
        base_save_dir = settings.PATH.result_dir / self.args.output_dir / ablation_name / self.args.model
        base_save_dir.mkdir(parents=True, exist_ok=True)
        return base_save_dir


#######################################################################################################
# Entry point of the script
#######################################################################################################


def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()

    # loop over different steps
    abl_path = settings.PATH.result_dir / "ablations" / args.vector / args.model
    save_path = (
        settings.PATH.result_dir
        / "token_freq"
        / args.effect
        / args.vector
        / args.model
        / f"{args.data_range_end}_{args.top_n}.csv"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.is_file():
        logger.info(f"{save_path} already exists, skip!")
    else:
        neuron_df = pd.DataFrame()
        # check whether the target file has been created
        for step in abl_path.iterdir():
            feather_path = abl_path / str(step) / str(args.data_range_end) / f"k{args.k}.feather"
            frame = select_top_token_frequency_neurons(feather_path, args.top_n, step.name, args.effect)
            neuron_df = pd.concat([neuron_df, frame])
        # assign col headers
        neuron_df.to_csv(save_path)
        logger.info(f"Save file to {save_path}")


if __name__ == "__main__":
    main()
