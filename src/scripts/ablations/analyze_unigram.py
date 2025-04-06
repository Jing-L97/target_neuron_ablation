#!/usr/bin/env python
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from neuron_analyzer import settings

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract word surprisal across different training steps.")

    parser.add_argument("-m", "--model", type=str, default="EleutherAI/pythia-70m-deduped", help="Target model name")
    parser.add_argument("--vector", choices=["mean", "longtail"],default="longtail")
    parser.add_argument("--effect", type=str, choices=["boost", "suppress"],default="suppress", help="boost or suppress long-tail prob")
    parser.add_argument("--top_n", type=int, default=10, help="use_bos_only if enabled")
    parser.add_argument("--data_range_end", type=int, default=500, help="the selected datarange")
    parser.add_argument("--k", type=int, default=10, help="use_bos_only if enabled")
    return parser.parse_args()


def select_top_token_frequency_neurons(feather_path: Path, top_n: int, step:int,effect:str) -> pd.DataFrame:
    if not feather_path.is_file():
        return

    final_df = pd.read_feather(feather_path)
    logger.info(f"Analyzing file from {feather_path}")
    final_df["abs_delta_loss_post_ablation"] = np.abs(final_df["loss_post_ablation"] - final_df["loss"])
    final_df["abs_delta_loss_post_ablation_with_frozen_unigram"] = np.abs(
        final_df["loss_post_ablation_with_frozen_unigram"] - final_df["loss"]
    )
    final_df["delta_loss_post_ablation"] = final_df["loss_post_ablation"] - final_df["loss"]
    final_df["delta_loss_post_ablation_with_frozen_unigram"] = final_df["loss_post_ablation_with_frozen_unigram"] - final_df["loss"]

    if "kl_divergence_before" in final_df.columns:
        final_df["kl_from_unigram_diff"] = final_df["kl_divergence_after"] - final_df["kl_divergence_before"]
        final_df["abs_kl_from_unigram_diff"] = np.abs(final_df["kl_divergence_after"] - final_df["kl_divergence_before"])

    # group by neuron idx
    final_df = final_df.groupby("component_name").mean(numeric_only=True).reset_index()

    if effect == "suppress":
        # filter the neurons that push towards the unigram freq
        final_df = final_df[final_df["kl_from_unigram_diff"] < 0]
    if effect == "boost":
        # filter the neurons that push away from the unigram freq
        final_df = final_df[final_df["kl_from_unigram_diff"] > 0]

    # Calculate the mediation effect
    final_df["mediation_effect"] = (
        1 - final_df["abs_delta_loss_post_ablation_with_frozen_unigram"] / final_df["abs_delta_loss_post_ablation"]
    )
    ranked_neurons = final_df.sort_values(by=["mediation_effect", "abs_kl_from_unigram_diff"], ascending=[False, False])

    # Select top N neurons, preserving the original sorting
    header_dict = {
        "component_name":"top_neurons",
        "mediation_effect":"med_effect",
        "kl_from_unigram_diff":"kl_diff",
        "delta_loss_post_ablation": "delta_loss_post",
        "delta_loss_post_ablation_with_frozen_unigram": "delta_loss_post_frozen"
        }
    df_lst = []
    for sel_header,_ in header_dict.items():
        df_lst.append([ranked_neurons[sel_header].head(top_n).tolist()])

    stat_df = pd.DataFrame(df_lst).T
    stat_df.columns = header_dict.values()
    stat_df.insert(0,"step",step)
    return stat_df




def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()

    # loop over different steps
    abl_path = settings.PATH.result_dir / "ablations" / args.vector / args.model
    save_path = settings.PATH.result_dir / "token_freq" / args.effect/args.vector /args.model / f"{args.data_range_end}_{args.top_n}.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.is_file():
        logger.info(f"{save_path} already exists, skip!")
    else:
        neuron_df = pd.DataFrame()
        # check whether the target file has been created
        for step in abl_path.iterdir():
            feather_path = abl_path / str(step) / str(args.data_range_end) / f"k{args.k}.feather"
            frame = select_top_token_frequency_neurons(feather_path, args.top_n,step.name,args.effect)
            neuron_df = pd.concat([neuron_df, frame])
        # assign col headers
        neuron_df.to_csv(save_path)
        logger.info(f"Save file to {save_path}")


if __name__ == "__main__":
    main()
