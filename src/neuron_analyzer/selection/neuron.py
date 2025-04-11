#!/usr/bin/env python
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from neuron_analyzer.analysis.freq import UnigramAnalyzer
from neuron_analyzer.load_util import JsonProcessor

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
        effect: str,
        tokenizer_name: str,
        threshold_path: Path,
        device: str,
        debug: bool,
        sel_longtail: bool = False,
    ):
        """Initialize the NeuronSelector."""
        self.feather_path = feather_path
        self.top_n = top_n
        self.step = step
        self.effect = effect
        self.debug = debug
        self.sel_longtail = sel_longtail
        # only load the arguments when needing longtail
        if self.sel_longtail:
            self.unigram_analyzer = UnigramAnalyzer(model_name=tokenizer_name, device=device)
            self.threshold_path = threshold_path

    def _prepare_dataframe(self) -> pd.DataFrame | None:
        """Common preprocessing for the DataFrame."""
        self.final_df = pd.read_feather(self.feather_path)
        logger.info(f"Analyzing file from {self.feather_path}")
        if self.debug:
            self.final_df = self.final_df.head(500)
            logger.info("Entering debugging mode. Loading first 500 rows.")
        # filter the df by stats
        if self.sel_longtail:
            self.final_df = self._filter_df()
        # Calculate delta loss metrics
        self.final_df["abs_delta_loss_post_ablation"] = np.abs(
            self.final_df["loss_post_ablation"] - self.final_df["loss"]
        )
        self.final_df["abs_delta_loss_post_ablation_with_frozen_unigram"] = np.abs(
            self.final_df["loss_post_ablation_with_frozen_unigram"] - self.final_df["loss"]
        )
        self.final_df["delta_loss_post_ablation"] = self.final_df["loss_post_ablation"] - self.final_df["loss"]
        self.final_df["delta_loss_post_ablation_with_frozen_unigram"] = (
            self.final_df["loss_post_ablation_with_frozen_unigram"] - self.final_df["loss"]
        )

        # Group by neuron idx
        self.final_df = self.final_df.groupby("component_name").mean(numeric_only=True).reset_index()

        # Calculate the mediation effect
        self.final_df["mediation_effect"] = (
            1
            - self.final_df["abs_delta_loss_post_ablation_with_frozen_unigram"]
            / self.final_df["abs_delta_loss_post_ablation"]
        )

        return self.final_df

    def _filter_df(self):
        """Filter df by frequency."""
        # get word freq
        self.final_df["freq"] = self.final_df["str_tokens"].apply(self._extract_freq)
        logger.info(f"{self.final_df.shape[0]} words before filtering")
        # filter by the threshold
        prob_dict = JsonProcessor.load_json(self.threshold_path)
        prob_threshold = prob_dict["threshold_info"]["probability"]
        print(self.final_df["freq"])
        self.final_df = self.final_df[self.final_df["freq"] < prob_threshold]
        logger.info(f"{self.final_df.shape[0]} words after filtering.")
        return self.final_df

    def _extract_freq(self, word):
        """Extract frequency from the unifram analyzer."""
        stat = self.unigram_analyzer.get_unigram_freq(word)
        return stat[0][2]

    def _create_stats_dataframe(self, ranked_neurons: pd.DataFrame, header_dict: dict[str, str]) -> pd.DataFrame:
        """Create a statistics DataFrame from ranked neurons."""
        df_lst = []
        for sel_header, _ in header_dict.items():
            df_lst.append([ranked_neurons[sel_header].head(self.top_n).tolist()])

        stat_df = pd.DataFrame(df_lst).T
        stat_df.columns = header_dict.values()
        stat_df.insert(0, "step", self.step)
        return stat_df

    def select_by_KL(self) -> pd.DataFrame | None:
        """Select neurons by mediation effect and KL."""
        final_df = self._prepare_dataframe()
        if final_df is None:
            return None

        if "kl_divergence_before" in final_df.columns:
            final_df["kl_from_unigram_diff"] = final_df["kl_divergence_after"] - final_df["kl_divergence_before"]
            final_df["abs_kl_from_unigram_diff"] = np.abs(
                final_df["kl_divergence_after"] - final_df["kl_divergence_before"]
            )

        # Apply effect filtering
        if self.effect == "suppress":
            # Filter the neurons that push towards the unigram freq
            final_df = final_df[final_df["kl_from_unigram_diff"] < 0]
        elif self.effect == "boost":
            # Filter the neurons that push away from the unigram freq
            final_df = final_df[final_df["kl_from_unigram_diff"] > 0]

        # Rank neurons
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

        return self._create_stats_dataframe(ranked_neurons, header_dict)

    def select_by_prob(self) -> pd.DataFrame | None:
        """Select neurons by mediation effect and prob variations."""
        final_df = self._prepare_dataframe()
        if final_df is None:
            return None

        # Apply effect filtering
        if self.effect == "suppress":
            # Filter the neurons that push towards the unigram freq
            final_df = final_df[final_df["delta_loss_post_ablation_with_frozen_unigram"] < 0]
        elif self.effect == "boost":
            # Filter the neurons that push away from the unigram freq
            final_df = final_df[final_df["delta_loss_post_ablation_with_frozen_unigram"] > 0]

        # Rank neurons
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

        return self._create_stats_dataframe(ranked_neurons, header_dict)
