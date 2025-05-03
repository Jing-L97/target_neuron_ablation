#!/usr/bin/env python
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from neuron_analyzer.analysis.freq import UnigramAnalyzer
from neuron_analyzer.load_util import load_tail_threshold_stat

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

#######################################################################################################
# Class to select neurons


class NeuronSelector:
    """Class for selecting neurons based on various criteria."""

    def __init__(
        self,
        feather_path: Path,
        top_n: int,
        step: int,
        tokenizer_name: str,
        threshold_path: Path,
        device: str,
        debug: bool,
        sel_freq: str | bool = False,
        sel_by_med: bool = False,
    ):
        """Initialize the NeuronSelector."""
        self.feather_path = feather_path
        self.top_n = top_n
        self.step = step
        self.debug = debug
        self.sel_freq = sel_freq
        self.sel_by_med = sel_by_med
        # only load the arguments when needing longtail
        if self.sel_freq:
            self.unigram_analyzer = UnigramAnalyzer(model_name=tokenizer_name, device=device)
            self.threshold_path = threshold_path

    def load_and_filter_df(self):
        """Load and filter df by frequency if required."""
        final_df = pd.read_feather(self.feather_path)
        logger.info(f"Analyzing file from {self.feather_path}")
        if self.debug:
            first_n_rows = 5000
            final_df = final_df.head(first_n_rows)
            logger.info(f"Entering debugging mode. Loading first {first_n_rows} rows.")
        # filter the df by stats
        if self.sel_freq:
            # get word freq
            final_df["freq"] = final_df["str_tokens"].apply(self._extract_freq)
            logger.info(f"{final_df.shape[0]} words before filtering")
            # filter by the threshold
            prob_threshold = load_tail_threshold_stat(self.threshold_path)
            if "longtail" in self.sel_freq:
                final_df = final_df[final_df["freq"] < prob_threshold]
            else:
                final_df = final_df[final_df["freq"] > prob_threshold]
            logger.info(f"{final_df.shape[0]} words after filtering.")

        # Calculate delta loss metrics
        final_df["delta_loss_post_ablation"] = final_df["loss_post_ablation"] - final_df["loss"]
        final_df["delta_loss_post_ablation_with_frozen_unigram"] = (
            final_df["loss_post_ablation_with_frozen_unigram"] - final_df["loss"]
        )
        final_df["abs_delta_loss_post_ablation"] = np.abs(final_df["delta_loss_post_ablation"])
        final_df["abs_delta_loss_post_ablation_with_frozen_unigram"] = np.abs(
            final_df["delta_loss_post_ablation_with_frozen_unigram"]
        )

        return final_df

    def _extract_freq(self, word):
        """Extract frequency from the unifram analyzer."""
        stat = self.unigram_analyzer.get_unigram_freq(word)
        return stat[0][2]

    def _prepare_dataframe(self, final_df: pd.DataFrame) -> pd.DataFrame | None:
        """Common preprocessing for the DataFrame."""
        # Group by neuron idx
        final_df = final_df.groupby("component_name").mean(numeric_only=True).reset_index()
        # Calculate the mediation effect if required
        if self.sel_by_med:
            final_df["mediation_effect"] = (
                1
                - final_df["abs_delta_loss_post_ablation_with_frozen_unigram"]
                / final_df["abs_delta_loss_post_ablation"]
            )
        return final_df

    def select_by_KL(self, effect: str, final_df: pd.DataFrame) -> pd.DataFrame | None:
        """Select neurons by mediation effect and KL."""
        if "kl_divergence_before" in final_df.columns:
            final_df["kl_from_unigram_diff"] = final_df["kl_divergence_after"] - final_df["kl_divergence_before"]
            final_df["abs_kl_from_unigram_diff"] = np.abs(
                final_df["kl_divergence_after"] - final_df["kl_divergence_before"]
            )

        # Apply effect filtering
        if effect == "suppress":
            # Filter the neurons that push towards the unigram freq
            final_df = final_df[final_df["kl_from_unigram_diff"] < 0]
        elif effect == "boost":
            # Filter the neurons that push away from the unigram freq
            final_df = final_df[final_df["kl_from_unigram_diff"] > 0]

        # Rank neurons
        if self.sel_by_med:
            ranked_neurons = final_df.sort_values(
                by=["mediation_effect", "abs_kl_from_unigram_diff"], ascending=[False, False]
            )
            # Define header dictionary
            header_dict = {
                "component_name": "top_neurons",
                "mediation_effect": "med_effect",
                "kl_from_unigram_diff": "kl_diff",
                "delta_loss_post_ablation": "delta_loss_post",
                "delta_loss_post_ablation_with_frozen_unigram": "delta_loss_post_frozen",
            }
        else:
            ranked_neurons = final_df.sort_values(by="abs_kl_from_unigram_diff", ascending=False)
            # Define header dictionary
            header_dict = {
                "component_name": "top_neurons",
                "kl_from_unigram_diff": "kl_diff",
                "delta_loss_post_ablation": "delta_loss_post",
                "delta_loss_post_ablation_with_frozen_unigram": "delta_loss_post_frozen",
            }

        return self._create_stats_dataframe(ranked_neurons, header_dict)

    def select_by_prob(self, effect: str, final_df: pd.DataFrame) -> pd.DataFrame | None:
        """Select neurons by mediation effect and prob variations."""
        # Apply effect filtering
        if effect == "suppress":
            # Filter the neurons that push towards the unigram freq
            final_df = final_df[final_df["delta_loss_post_ablation"] < 0]
        elif effect == "boost":
            # Filter the neurons that push away from the unigram freq
            final_df = final_df[final_df["delta_loss_post_ablation"] > 0]

        # Rank neurons
        if self.sel_by_med:
            ranked_neurons = final_df.sort_values(
                by=["mediation_effect", "abs_delta_loss_post_ablation_with_frozen_unigram"], ascending=[False, False]
            )
            # Define header dictionary
            header_dict = {
                "component_name": "top_neurons",
                "mediation_effect": "med_effect",
                "delta_loss_post_ablation": "delta_loss_post",
                "delta_loss_post_ablation_with_frozen_unigram": "delta_loss_post_frozen",
            }
        else:
            ranked_neurons = final_df.sort_values(by="abs_delta_loss_post_ablation", ascending=False)
            # Define header dictionary
            header_dict = {
                "component_name": "top_neurons",
                "delta_loss_post_ablation": "delta_loss_post",
                "delta_loss_post_ablation_with_frozen_unigram": "delta_loss_post_frozen",
            }
        return self._create_stats_dataframe(ranked_neurons, header_dict)

    def _create_stats_dataframe(self, ranked_neurons: pd.DataFrame, header_dict: dict[str, str]) -> pd.DataFrame:
        """Create a statistics DataFrame from ranked neurons."""
        df_lst = []
        for sel_header, _ in header_dict.items():
            if self.top_n != -1:
                df_lst.append([ranked_neurons[sel_header].head(self.top_n).tolist()])
            else:
                df_lst.append([ranked_neurons[sel_header].tolist()])
        stat_df = pd.DataFrame(df_lst).T
        stat_df.columns = header_dict.values()
        stat_df.insert(0, "step", self.step)
        return stat_df

    def run_pipeline(self, heuristic: str, effect: str):
        """Run the selection pipeline."""
        final_df = self.load_and_filter_df()
        final_df = self._prepare_dataframe(final_df)
        if heuristic == "KL":
            return self.select_by_KL(final_df=final_df, effect=effect)
        if heuristic == "prob":
            return self.select_by_prob(final_df=final_df, effect=effect)
        return None


#######################################################################################################
# Function to generate random neuron indices


def generate_random_indices(
    all_neuron_indices: list[int], special_indices: list[int], group_size: int, num_random_groups: int
) -> list[int]:
    """Generate non-overlapping random indices that don't overlap with given list."""
    non_special_indices = [idx for idx in all_neuron_indices if idx not in special_indices]
    random_indices = []
    # Ensure we have enough neurons for random groups
    if len(non_special_indices) < 2 * group_size:
        logger.warning(f"Not enough neurons for non-overlapping random groups of size {group_size}.")
        # Split available neurons into two groups
        np.random.shuffle(non_special_indices)
        split_point = len(non_special_indices) // 2
        random_indices.append(non_special_indices[:split_point])
        random_indices.append(non_special_indices[split_point:])
    else:
        # Create non-overlapping random groups
        np.random.shuffle(non_special_indices)
        for i in range(num_random_groups):
            start_idx = i * group_size
            end_idx = (i + 1) * group_size
            if end_idx <= len(non_special_indices):
                random_indices.append(non_special_indices[start_idx:end_idx])
            else:
                # If we don't have enough neurons, just use what's left
                random_indices.append(non_special_indices[start_idx:])
    return random_indices
