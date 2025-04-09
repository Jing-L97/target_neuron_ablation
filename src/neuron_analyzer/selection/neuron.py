#!/usr/bin/env python
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NeuronSelector:
    """Class for selecting neurons based on various criteria."""

    def __init__(self, top_n: int, step: int, effect: str = ""):
        """Initialize the NeuronSelector."""
        self.top_n = top_n
        self.step = step
        self.effect = effect

    def _prepare_dataframe(self, feather_path: Path) -> pd.DataFrame | None:
        """Common preprocessing for the DataFrame."""
        if not feather_path.is_file():
            return None

        final_df = pd.read_feather(feather_path)
        logger.info(f"Analyzing file from {feather_path}")

        # Calculate delta loss metrics
        final_df["abs_delta_loss_post_ablation"] = np.abs(final_df["loss_post_ablation"] - final_df["loss"])
        final_df["abs_delta_loss_post_ablation_with_frozen_unigram"] = np.abs(
            final_df["loss_post_ablation_with_frozen_unigram"] - final_df["loss"]
        )
        final_df["delta_loss_post_ablation"] = final_df["loss_post_ablation"] - final_df["loss"]
        final_df["delta_loss_post_ablation_with_frozen_unigram"] = (
            final_df["loss_post_ablation_with_frozen_unigram"] - final_df["loss"]
        )

        # Group by neuron idx
        final_df = final_df.groupby("component_name").mean(numeric_only=True).reset_index()

        # Calculate the mediation effect
        final_df["mediation_effect"] = (
            1 - final_df["abs_delta_loss_post_ablation_with_frozen_unigram"] / final_df["abs_delta_loss_post_ablation"]
        )

        return final_df

    def _create_stats_dataframe(self, ranked_neurons: pd.DataFrame, header_dict: dict[str, str]) -> pd.DataFrame:
        """Create a statistics DataFrame from ranked neurons."""
        df_lst = []
        for sel_header, _ in header_dict.items():
            df_lst.append([ranked_neurons[sel_header].head(self.top_n).tolist()])

        stat_df = pd.DataFrame(df_lst).T
        stat_df.columns = header_dict.values()
        stat_df.insert(0, "step", self.step)
        return stat_df

    def select_by_KL(self, feather_path: Path) -> pd.DataFrame | None:
        """Select neurons by mediation effect and KL."""
        final_df = self._prepare_dataframe(feather_path)
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

    def select_by_prob(self, feather_path: Path) -> pd.DataFrame | None:
        """Select neurons by mediation effect and prob variations."""
        final_df = self._prepare_dataframe(feather_path)
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
