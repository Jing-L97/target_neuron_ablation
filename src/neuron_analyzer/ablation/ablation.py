import os
import sys

sys.path.append("../")
import logging
import typing as t
from pathlib import Path
from warnings import simplefilter

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from transformer_lens import utils

from neuron_analyzer import settings
from neuron_analyzer.ablation.abl_util import (
    filter_entropy_activation_df,
    get_entropy_activation_df,
)
from neuron_analyzer.ablation.ablation import ModelAblationAnalyzer
from neuron_analyzer.analysis.freq import ZipfThresholdAnalyzer
from neuron_analyzer.load_util import JsonProcessor
from neuron_analyzer.model_util import ModelHandler

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# Type variable for generic typing
T = t.TypeVar("T")


######################################################
# Configure altogether


class NeuronAblationProcessor:
    """Class to handle neural network ablation processing."""

    def __init__(self, args: DictConfig, device: str, logger: logging.Logger | None = None):
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

        # Change directory if specified
        if hasattr(args, "chdir") and args.chdir:
            os.chdir(args.chdir)

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

    def load_entropy_df(self, step) -> pd.DataFrame:
        """Load entropy df if needed."""
        h_path = self._config_entropy_path(step)
        # load hte exisitng file
        if h_path.is_file():
            return pd.read_csv(h_path)

        token_df = get_entropy_activation_df(
            self.all_neurons,
            self.tokenized_data,
            self.token_df,
            self.model,
            batch_size=self.args.batch_size,
            device=self.device,
            cache_residuals=False,
            cache_pre_activations=False,
            compute_kl_from_bu=False,
            residuals_layer=self.entropy_dim_layer,
            residuals_dict={},
        )
        token_df.to_csv(h_path)
        return token_df

    def process_single_step(self, step: int, unigram_distrib, longtail_threshold, save_path: Path) -> None:
        """Process a single step with the given configuration."""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info("Finished loading model and tokenizer")
        # initlize the model handler class
        model_handler = ModelHandler()

        # Load model and tokenizer for specific step
        self.model, self.tokenizer = model_handler.load_model_and_tokenizer(
            step=step,
            model_name=self.args.model,
            hf_token_path=settings.PATH.unigram_dir / "hf_token.txt",
            device=self.device,
        )

        # Load and process dataset
        self.tokenized_data, self.token_df = model_handler.tokenize_data(
            dataset=self.args.dataset,
            data_range_start=self.args.data_range_start,
            data_range_end=self.args.data_range_end,
            seed=self.args.seed,
            get_df=True,
        )

        self.logger.info("Finished tokenizing data")

        # Setup neuron indices
        entropy_neuron_layer = self.model.cfg.n_layers - 1
        if self.args.neuron_range is not None:
            start, end = map(int, self.args.neuron_range.split("-"))
            all_neuron_indices = list(range(start, end))
        else:
            all_neuron_indices = list(range(self.model.cfg.d_mlp))

        self.all_neurons = [f"{entropy_neuron_layer}.{i}" for i in all_neuron_indices]
        self.logger.info("Loaded all the neurons")

        if self.args.dry_run:
            self.all_neurons = self.all_neurons[:10]

        # Compute entropy and activation for each neuron
        self.entropy_dim_layer = self.model.cfg.n_layers - 1

        self.entropy_df = self.load_entropy_df(step)
        self.logger.info("Finished computing all the entropy")
        if self.args.debug:
            row_num = 1_000_000
            self.entropy_df = self.entropy_df.head(row_num)
            logger.info(f"Enter debugging mode. Apply {row_num} rows.")

        # Ablate the dimensions
        self.model.set_use_attn_result(False)

        analyzer = ModelAblationAnalyzer(
            components_to_ablate=self.all_neurons,
            model=self.model,
            device=self.device,
            unigram_distrib=unigram_distrib,
            tokenized_data=self.tokenized_data,
            entropy_df=self.entropy_df,
            k=self.args.k,
            ablation_mode=self.args.ablation_mode,
            longtail_threshold=longtail_threshold,
        )

        results = analyzer.mean_ablate_components()
        self.logger.info("Finished ablations!")
        # Process and save results
        self._save_results(results, self.tokenizer, step, save_path)

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

    def _config_entropy_path(self, step) -> Path:
        """Configure entorpy df path based on current settings."""
        path_prefix = settings.PATH.ablation_dir / self.args.ablation_mode / self.args.model
        if step is None:
            return path_prefix / str(self.args.data_range_end) / "entropy_df.csv"
        return path_prefix / str(step) / str(self.args.data_range_end) / "entropy_df.csv"


######################################################
# Run ablation experiments


class ModelAblationAnalyzer:
    """Class for analyzing neural network components through ablations."""

    def __init__(
        self,
        model,
        unigram_distrib,
        tokenized_data,
        entropy_df,
        components_to_ablate,
        device: str,
        k: int = 10,
        chunk_size: int = 20,
        ablation_mode: str = "mean",  # "mean" or "longtail"
        longtail_threshold: float = 0.001,  # Threshold for long-tail tokens
    ):
        """Initialize the ModelAblationAnalyzer."""
        self.model = model
        self.unigram_distrib = unigram_distrib
        self.tokenized_data = tokenized_data
        self.entropy_df = entropy_df
        self.device = device
        self.k = k
        self.chunk_size = chunk_size
        self.ablation_mode = ablation_mode
        self.longtail_threshold = longtail_threshold
        self.results = {}
        self.log_unigram_distrib = self.unigram_distrib.log()
        self.components_to_ablate = components_to_ablate
        self.longtail_mask = None
        self.unigram_direction_vocab = None

    def build_vector(self) -> None:
        """Build frequency-related vectors for ablation analysis."""
        if "longtail" in self.ablation_mode:
            # Create long-tail token mask (1 for long-tail tokens, 0 for common tokens)
            self.longtail_mask = (self.unigram_distrib < self.longtail_threshold).float()
            # Create token frequency vector focusing on long-tail tokens only
            # Original token frequency vector from the unigram distribution
            full_unigram_direction_vocab = self.unigram_distrib.log() - self.unigram_distrib.log().mean()
            full_unigram_direction_vocab /= full_unigram_direction_vocab.norm()

            # This makes the vector only consider contributions from long-tail tokens
            self.unigram_direction_vocab = full_unigram_direction_vocab * self.longtail_mask
            # Re-normalize to keep it a unit vector
            if self.unigram_direction_vocab.norm() > 0:
                self.unigram_direction_vocab /= self.unigram_direction_vocab.norm()
        else:
            # Standard frequency direction for regular mean ablation
            self.unigram_direction_vocab = self.unigram_distrib.log() - self.unigram_distrib.log().mean()
            self.unigram_direction_vocab /= self.unigram_direction_vocab.norm()
            self.longtail_mask = None

    def project_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Get the value of logits projected onto the unigram direction."""
        unigram_projection_values = logits @ self.unigram_direction_vocab
        return unigram_projection_values.squeeze()

    def adjust_vectors_3dim(self, v: torch.Tensor, u: torch.Tensor, target_values: torch.Tensor) -> torch.Tensor:
        """Adjusts a batch of vectors v such that their projections along the unit vector u equal the target values.

        v: A 3D tensor of shape (n, m, d), representing the batch of vectors to be adjusted
        u: A 1D unit tensor of shape (d,), representing the direction along which the adjustment is made
        target_values: A 2D tensor of shape (n, m), representing the desired projection values of the vectors in v along u
        """
        current_projections = (v @ u.unsqueeze(-1)).squeeze(-1)  # Current projections of v onto u
        delta = target_values - current_projections  # Differences needed to reach the target projections
        adjusted_v = v + delta.unsqueeze(-1) * u  # Adjust v by the deltas along the direction of u
        return adjusted_v

    def project_neurons(
        self,
        logits: torch.Tensor,
        unigram_projection_values: torch.Tensor,
        ablated_logits_chunk: torch.Tensor,
        res_deltas_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Project neurons based on ablation mode and adjust vectors."""
        # If we're in long-tail mode, apply the mask
        if "longtail" in self.ablation_mode:
            # Get the original logits to preserve for common tokens
            original_logits = logits.repeat(res_deltas_chunk.shape[0], 1, 1)
            # Create a binary mask for the vocabulary dimension
            # 1 for long-tail tokens (to be modified), 0 for common tokens (to keep original)
            vocab_mask = self.longtail_mask.unsqueeze(0).unsqueeze(0)  # Shape: 1 x 1 x vocab_size
            # Apply the mask: use original_logits where mask is 0, use ablated_logits where mask is 1
            ablated_logits_chunk = (1 - vocab_mask) * original_logits + vocab_mask * ablated_logits_chunk

        # Adjust vectors to maintain unigram projections
        ablated_logits_with_frozen_unigram_chunk = self.adjust_vectors_3dim(
            ablated_logits_chunk, self.unigram_direction_vocab, unigram_projection_values
        )
        return ablated_logits_with_frozen_unigram_chunk

    def get_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Calculate entropy from logits."""
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(-1)
        return entropy

    def kl_div(
        self,
        input_logprobs: torch.Tensor,
        target_logprobs: torch.Tensor,
        reduction: str = "none",
        log_target: bool = True,
    ) -> torch.Tensor:
        """Calculate KL divergence between distributions."""
        if log_target:
            kl = torch.exp(input_logprobs) * (input_logprobs - target_logprobs)
        else:
            kl = torch.exp(input_logprobs) * (input_logprobs - torch.log(target_logprobs))

        if reduction == "none":
            return kl
        if reduction == "sum":
            return kl.sum()
        if reduction == "mean":
            return kl.mean()
        raise ValueError(f"Unsupported reduction: {reduction}")

    def compute_kl(
        self,
        ablated_logits_with_frozen_unigram_chunk: torch.Tensor,
        abl_logprobs: torch.Tensor,
        kl_divergence_after: list,
        kl_divergence_after_frozen_unigram: list,
    ) -> tuple[list, list]:
        """Compute KL divergence metrics for ablated distributions."""
        # compute KL divergence between the distribution ablated with frozen unigram and the og distribution
        abl_logprobs_with_frozen_unigram = ablated_logits_with_frozen_unigram_chunk.log_softmax(dim=-1)

        # compute KL divergence between the ablated distribution and the distribution from the unigram direction
        kl_divergence_after_chunk = (
            self.kl_div(
                abl_logprobs, self.log_unigram_distrib.expand_as(abl_logprobs), reduction="none", log_target=True
            )
            .sum(axis=-1)
            .cpu()
            .numpy()
        )

        del abl_logprobs
        kl_divergence_after.append(kl_divergence_after_chunk)

        if "longtail" in self.ablation_mode:
            # For long-tail mode, compute KL divergence with focus on the long-tail tokens
            masked_logprobs = abl_logprobs_with_frozen_unigram.clone()
            masked_logprobs = masked_logprobs + (1 - self.longtail_mask).unsqueeze(0).unsqueeze(0) * -1e10
            masked_logprobs = torch.nn.functional.log_softmax(masked_logprobs, dim=-1)
            kl_divergence_after_frozen_unigram_chunk = (
                self.kl_div(
                    masked_logprobs,
                    self.log_unigram_distrib.expand_as(masked_logprobs),
                    reduction="none",
                    log_target=True,
                )
                .sum(axis=-1)
                .cpu()
                .numpy()
            )
        else:
            # Standard KL divergence for regular mean ablation
            kl_divergence_after_frozen_unigram_chunk = (
                self.kl_div(
                    abl_logprobs_with_frozen_unigram,
                    self.log_unigram_distrib.expand_as(abl_logprobs_with_frozen_unigram),
                    reduction="none",
                    log_target=True,
                )
                .sum(axis=-1)
                .cpu()
                .numpy()
            )

        del abl_logprobs_with_frozen_unigram
        kl_divergence_after_frozen_unigram.append(kl_divergence_after_frozen_unigram_chunk)

        del ablated_logits_with_frozen_unigram_chunk
        return kl_divergence_after, kl_divergence_after_frozen_unigram

    def process_batch_results(
        self,
        batch_n: int,
        filtered_entropy_df: pd.DataFrame,
        loss_post_ablation: np.ndarray,
        loss_post_ablation_with_frozen_unigram: np.ndarray,
        entropy_post_ablation: np.ndarray,
        entropy_post_ablation_with_frozen_unigram: np.ndarray,
        kl_divergence_before: np.ndarray,
        kl_divergence_after: np.ndarray,
        kl_divergence_after_frozen_unigram: np.ndarray,
        pos_to_idx: dict[int, int] = None,
    ) -> pd.DataFrame:
        """Process results for a batch and create a dataframe."""
        final_df = None

        # Get batch-specific DataFrame
        batch_df = filtered_entropy_df[filtered_entropy_df.batch == batch_n]

        # Get positions for this batch
        positions = sorted(batch_df["pos"].unique())

        # Log diagnostic information
        logger.debug(f"Processing batch {batch_n} with {len(positions)} positions")

        # Create position-to-index mapping if not provided
        if pos_to_idx is None:
            pos_to_idx = {pos: i for i, pos in enumerate(positions)}

        for i, component_name in enumerate(self.components_to_ablate):
            # Start with a copy of batch data
            df_to_append = batch_df.copy()

            # Drop all the columns that are not the component_name
            component_cols = [
                f"{neuron}_activation" for neuron in self.components_to_ablate if neuron != component_name
            ]
            if component_cols:
                df_to_append = df_to_append.drop(columns=component_cols, errors="ignore")

            # Rename the component_name column to 'activation'
            if f"{component_name}_activation" in df_to_append.columns:
                df_to_append = df_to_append.rename(columns={f"{component_name}_activation": "activation"})

            df_to_append["component_name"] = component_name

            # Create result columns with appropriate sizes
            for pos in positions:
                # Get the corresponding index for tensors
                idx = pos_to_idx[pos]

                # Skip if the index is out of range
                if idx >= loss_post_ablation[i].shape[1]:
                    logger.warning(f"Position {pos} (idx {idx}) is out of range for tensors. Skipping.")
                    continue

                # Get rows for this position
                pos_mask = df_to_append["pos"] == pos
                if not pos_mask.any():
                    logger.warning(f"Position {pos} not found in DataFrame. Skipping.")
                    continue

                # Set values for this position
                df_to_append.loc[pos_mask, "loss_post_ablation"] = loss_post_ablation[i][0, idx]
                df_to_append.loc[pos_mask, "loss_post_ablation_with_frozen_unigram"] = (
                    loss_post_ablation_with_frozen_unigram[i][0, idx]
                )
                df_to_append.loc[pos_mask, "entropy_post_ablation"] = entropy_post_ablation[i][0, idx]
                df_to_append.loc[pos_mask, "entropy_post_ablation_with_frozen_unigram"] = (
                    entropy_post_ablation_with_frozen_unigram[i][0, idx]
                )

                # KL divergence values
                if idx < len(kl_divergence_before):
                    df_to_append.loc[pos_mask, "kl_divergence_before"] = kl_divergence_before[idx]
                if idx < kl_divergence_after[i].shape[0]:
                    df_to_append.loc[pos_mask, "kl_divergence_after"] = kl_divergence_after[i][idx]
                if idx < kl_divergence_after_frozen_unigram[i].shape[0]:
                    df_to_append.loc[pos_mask, "kl_divergence_after_frozen_unigram"] = (
                        kl_divergence_after_frozen_unigram[i][idx]
                    )

            # Add ablation information
            df_to_append["ablation_mode"] = self.ablation_mode
            if "longtail" in self.ablation_mode and self.longtail_mask is not None:
                df_to_append["longtail_threshold"] = self.longtail_threshold
                df_to_append["num_longtail_tokens"] = self.longtail_mask.sum().item()

            # Append to our results
            final_df = df_to_append if final_df is None else pd.concat([final_df, df_to_append])

        # Handle case where we couldn't process any results
        if final_df is None:
            logger.warning(f"No valid results for batch {batch_n}, returning empty DataFrame")
            return pd.DataFrame()

        return final_df

    def mean_ablate_components(self) -> dict[int, pd.DataFrame]:
        """Perform mean ablation on specified components."""
        # Sample a set of random batch indices
        try:
            random_sequence_indices = np.random.choice(self.entropy_df.batch.unique(), self.k, replace=False)
        except ValueError as e:
            logger.error(f"Error sampling batch indices: {e}")
            # If there are fewer batches than k, use all available batches
            random_sequence_indices = self.entropy_df.batch.unique()
            logger.info(f"Using all available batches: {len(random_sequence_indices)}")

        logger.info(f"ablate_components: ablate with k = {self.k}, long-tail threshold = {self.longtail_threshold}")

        # Get entropy_df with only the random sequences
        filtered_entropy_df = self.entropy_df[self.entropy_df.batch.isin(random_sequence_indices)].copy()

        # Calculate mean activation values for the components to ablate
        activation_mean_values = torch.tensor(
            self.entropy_df[[f"{component_name}_activation" for component_name in self.components_to_ablate]].mean()
        )

        # Add diagnostic logging for first few batches
        for batch_idx, batch_n in enumerate(list(filtered_entropy_df.batch.unique())[:3]):
            batch_df = filtered_entropy_df[filtered_entropy_df.batch == batch_n]
            tok_seq = self.tokenized_data["tokens"][batch_n]
            positions = sorted(batch_df["pos"].unique())

            logger.info(f"Diagnostic for batch {batch_n}:")
            logger.info(f"  DataFrame rows: {len(batch_df)}")
            logger.info(f"  Unique positions: {len(positions)}")
            logger.info(f"  Token sequence length: {len(tok_seq)}")
            if positions:
                logger.info(f"  Position range: {min(positions)} to {max(positions)}")

                # Check for position gaps
                if len(positions) > 1:
                    gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
                    all_sequential = all(gap == 1 for gap in gaps)
                    logger.info(f"  Positions are sequential: {all_sequential}")
                    if not all_sequential:
                        gap_counts = sum(1 for gap in gaps if gap > 1)
                        logger.info(f"  Number of gaps: {gap_counts}")

        # Build frequency vectors
        self.build_vector()

        # Get neuron indices
        self.neuron_indices = [int(neuron_name.split(".")[1]) for neuron_name in self.components_to_ablate]

        # Get layer indices
        layer_indices = [int(neuron_name.split(".")[0]) for neuron_name in self.components_to_ablate]
        self.layer_idx = layer_indices[0]

        for batch_n in filtered_entropy_df.batch.unique():
            # Get token sequence
            tok_seq = self.tokenized_data["tokens"][batch_n]

            # Get positions for this batch
            batch_df = filtered_entropy_df[filtered_entropy_df.batch == batch_n]
            positions = sorted(batch_df["pos"].unique())

            # Create position-to-index mapping
            pos_to_idx = {pos: i for i, pos in enumerate(positions)}

            # Check for position mismatch
            if len(positions) != len(tok_seq):
                logger.warning(
                    f"Position count mismatch for batch {batch_n}: "
                    f"DataFrame has {len(positions)} unique positions, "
                    f"token sequence has {len(tok_seq)} tokens"
                )

                # Filter positions to valid range
                valid_positions = [pos for pos in positions if pos < len(tok_seq)]
                if len(valid_positions) < len(positions):
                    logger.warning(f"Filtered out {len(positions) - len(valid_positions)} invalid positions")
                    positions = valid_positions
                    # Update mapping
                    pos_to_idx = {pos: i for i, pos in enumerate(positions)}

            # Get unaltered logits
            self.model.reset_hooks()
            inp = tok_seq.unsqueeze(0).to(self.device)
            logits, cache = self.model.run_with_cache(inp)
            logprobs = logits[0, :, :].log_softmax(dim=-1)

            # Get residual stream from cache
            res_stream_full = cache[utils.get_act_name("resid_post", self.layer_idx)][0]

            # Extract residual stream only for positions we care about
            res_stream = res_stream_full[positions]

            # Get the value of the logits projected onto the b_U direction, for our positions
            full_unigram_projection_values = self.project_logits(logits)
            unigram_projection_values = torch.tensor(
                [full_unigram_projection_values[pos] for pos in positions], device=full_unigram_projection_values.device
            )

            # Get neuron activations for our positions
            previous_activation_full = cache[utils.get_act_name("post", self.layer_idx)][0]
            previous_activation = previous_activation_full[positions][:, self.neuron_indices]

            del cache

            # Calculate activation deltas
            activation_deltas = activation_mean_values.to(previous_activation.device) - previous_activation
            # activation deltas is filtered_seq_len x n_neurons

            # Multiple deltas by W_out
            res_deltas = activation_deltas.unsqueeze(-1) * self.model.W_out[self.layer_idx, self.neuron_indices, :]
            res_deltas = res_deltas.permute(1, 0, 2)

            # Initialize result containers
            loss_post_ablation = []
            entropy_post_ablation = []
            loss_post_ablation_with_frozen_unigram = []
            entropy_post_ablation_with_frozen_unigram = []
            kl_divergence_after = []
            kl_divergence_after_frozen_unigram = []

            # Calculate KL divergence before ablation
            # We need to extract logprobs only for our positions
            position_logprobs = logprobs[positions]
            kl_divergence_before = (
                self.kl_div(position_logprobs, self.log_unigram_distrib, reduction="none", log_target=True)
                .sum(axis=-1)
                .cpu()
                .numpy()
            )

            # Process in chunks to manage memory
            for i in range(0, res_deltas.shape[0], self.chunk_size):
                res_deltas_chunk = res_deltas[i : i + self.chunk_size]
                updated_res_stream_chunk = res_stream.repeat(res_deltas_chunk.shape[0], 1, 1) + res_deltas_chunk

                # Apply layer normalization
                updated_res_stream_chunk = self.model.ln_final(updated_res_stream_chunk)

                # Project to logit space
                ablated_logits_chunk = updated_res_stream_chunk @ self.model.W_U + self.model.b_U

                # Create filtered input for loss calculation
                filtered_inp = tok_seq[positions].unsqueeze(0).to(self.device)
                filtered_inp_repeated = filtered_inp.repeat(res_deltas_chunk.shape[0], 1)

                # Project neurons for frequency adjustment
                ablated_logits_with_frozen_unigram_chunk = self.project_neurons(
                    logits[:, positions, :],  # Only use logits for positions we care about
                    unigram_projection_values,
                    ablated_logits_chunk,
                    res_deltas_chunk,
                )

                # Compute loss for the chunk
                loss_post_ablation_chunk = self.model.loss_fn(
                    ablated_logits_chunk, filtered_inp_repeated, per_token=True
                ).cpu()
                loss_post_ablation_chunk = np.concatenate(
                    (loss_post_ablation_chunk, np.zeros((loss_post_ablation_chunk.shape[0], 1))), axis=1
                )
                loss_post_ablation.append(loss_post_ablation_chunk)

                # Compute entropy for the chunk
                entropy_post_ablation_chunk = self.get_entropy(ablated_logits_chunk)
                entropy_post_ablation.append(entropy_post_ablation_chunk.cpu())

                # Get log probabilities
                abl_logprobs = ablated_logits_chunk.log_softmax(dim=-1)
                del ablated_logits_chunk

                # Compute loss for ablated_logits_with_frozen_unigram_chunk
                loss_post_ablation_with_frozen_unigram_chunk = self.model.loss_fn(
                    ablated_logits_with_frozen_unigram_chunk, filtered_inp_repeated, per_token=True
                ).cpu()
                loss_post_ablation_with_frozen_unigram_chunk = np.concatenate(
                    (
                        loss_post_ablation_with_frozen_unigram_chunk,
                        np.zeros((loss_post_ablation_with_frozen_unigram_chunk.shape[0], 1)),
                    ),
                    axis=1,
                )
                loss_post_ablation_with_frozen_unigram.append(loss_post_ablation_with_frozen_unigram_chunk)

                # Compute entropy for ablated_logits_with_frozen_unigram_chunk
                entropy_post_ablation_with_frozen_unigram_chunk = self.get_entropy(
                    ablated_logits_with_frozen_unigram_chunk
                )
                entropy_post_ablation_with_frozen_unigram.append(entropy_post_ablation_with_frozen_unigram_chunk.cpu())

                # Calculate KL divergence
                kl_divergence_after, kl_divergence_after_frozen_unigram = self.compute_kl(
                    ablated_logits_with_frozen_unigram_chunk,
                    abl_logprobs,
                    kl_divergence_after,
                    kl_divergence_after_frozen_unigram,
                )

            # Concatenate results
            loss_post_ablation = np.concatenate(loss_post_ablation, axis=0)
            entropy_post_ablation = np.concatenate(entropy_post_ablation, axis=0)
            loss_post_ablation_with_frozen_unigram = np.concatenate(loss_post_ablation_with_frozen_unigram, axis=0)
            entropy_post_ablation_with_frozen_unigram = np.concatenate(
                entropy_post_ablation_with_frozen_unigram, axis=0
            )
            kl_divergence_after = np.concatenate(kl_divergence_after, axis=0)
            kl_divergence_after_frozen_unigram = np.concatenate(kl_divergence_after_frozen_unigram, axis=0)

            del res_deltas
            torch.cuda.empty_cache()  # Empty the cache

            # Process results and prepare dataframe
            batch_results = self.process_batch_results(
                batch_n,
                filtered_entropy_df,
                loss_post_ablation,
                loss_post_ablation_with_frozen_unigram,
                entropy_post_ablation,
                entropy_post_ablation_with_frozen_unigram,
                kl_divergence_before,
                kl_divergence_after,
                kl_divergence_after_frozen_unigram,
                pos_to_idx,  # Pass the position mapping
            )

            self.results[batch_n] = batch_results

        return self.results
