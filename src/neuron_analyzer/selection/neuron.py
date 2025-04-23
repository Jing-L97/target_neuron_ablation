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
        sel_longtail: bool = False,
        sel_by_med: bool = False,
    ):
        """Initialize the NeuronSelector."""
        self.feather_path = feather_path
        self.top_n = top_n
        self.step = step
        self.debug = debug
        self.sel_longtail = sel_longtail
        self.sel_by_med = sel_by_med
        # only load the arguments when needing longtail
        if self.sel_longtail:
            self.unigram_analyzer = UnigramAnalyzer(model_name=tokenizer_name, device=device)
            self.threshold_path = threshold_path

    def _prepare_dataframe(self, final_df) -> pd.DataFrame | None:
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

    def load_and_filter_df(self):
        """Load and filter df by frequency if required."""
        final_df = pd.read_feather(self.feather_path)
        logger.info(f"Analyzing file from {self.feather_path}")
        if self.debug:
            first_n_rows = 5000
            final_df = final_df.head(first_n_rows)
            logger.info(f"Entering debugging mode. Loading first {first_n_rows} rows.")
        # filter the df by stats
        if self.sel_longtail:
            # get word freq
            final_df["freq"] = final_df["str_tokens"].apply(self._extract_freq)
            logger.info(f"{final_df.shape[0]} words before filtering")
            # filter by the threshold
            prob_threshold = load_tail_threshold_stat(self.threshold_path)
            final_df = final_df[final_df["freq"] < prob_threshold]
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

    def select_by_KL(self, effect: str, final_df=None) -> pd.DataFrame | None:
        """Select neurons by mediation effect and KL."""
        if final_df is None:
            final_df = self._prepare_dataframe()

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

    def select_by_prob(self, effect: str, final_df=None) -> pd.DataFrame | None:
        """Select neurons by mediation effect and prob variations."""
        if final_df is None:
            final_df = self._prepare_dataframe()

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
