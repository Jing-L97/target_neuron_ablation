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
    parser.add_argument("--ablation_mode", choices=["mean", "longtail"],default="longtail")
    parser.add_argument("--top_n", type=int, default=10, help="use_bos_only if enabled")
    parser.add_argument("--data_range_end", type=int, default=500, help="use_bos_only if enabled")
    parser.add_argument("--k", type=int, default=10, help="use_bos_only if enabled")
    return parser.parse_args()


def select_top_token_frequency_neurons(feather_path: Path, top_n: int, step:int) -> pd.DataFrame:
    if not feather_path.is_file():
        return

    final_df = pd.read_feather(feather_path)
    logger.info(f"Analyzing file from {feather_path}")
    final_df["abs_delta_loss_post_ablation"] = np.abs(final_df["loss_post_ablation"] - final_df["loss"])
    final_df["abs_delta_loss_post_ablation_with_frozen_unigram"] = np.abs(
        final_df["loss_post_ablation_with_frozen_unigram"] - final_df["loss"]
    )
    if "kl_divergence_before" in final_df.columns:
        final_df["kl_from_unigram_diff"] = final_df["kl_divergence_after"] - final_df["kl_divergence_before"]
    # filter the neurons that push towards the unigram freq
    final_df = final_df[final_df["kl_from_unigram_diff"] > 0]
    # TODO: 
    """
    add the supressing neurons
    
    boost the activation
    
    """
    # Calculate the mediation effect
    final_df["mediation_effect"] = (
        1 - final_df["abs_delta_loss_post_ablation_with_frozen_unigram"] / final_df["abs_delta_loss_post_ablation"]
    )

    ranked_neurons = final_df.sort_values(by=["mediation_effect", "kl_from_unigram_diff"], ascending=[False, False])
    # Select top N neurons, preserving the original sorting
    top_neurons = ranked_neurons["component_name"].head(top_n).tolist()
    med_effect = ranked_neurons["mediation_effect"].head(top_n).tolist()
    kl_diff = ranked_neurons["kl_from_unigram_diff"].head(top_n).tolist()

    return pd.DataFrame([step,top_neurons,med_effect,kl_diff]).T





def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()

    # loop over different steps
    abl_path = settings.PATH.result_dir / "ablations" / args.ablation_mode / args.model
    save_path = settings.PATH.result_dir / "token_freq" / args.ablation_mode /args.model / f"{args.data_range_end}_{args.top_n}.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.is_file():
        logger.info(f"{save_path} already exists, skip!")
    else:
        neuron_df = pd.DataFrame()
        # check whether the target file has been created
        for step in abl_path.iterdir():
            feather_path = abl_path / str(step) / str(args.data_range_end) / f"k{args.k}.feather"
            frame = select_top_token_frequency_neurons(feather_path, args.top_n, step.name)
            neuron_df = pd.concat([neuron_df, frame])
        # assign col headers
        neuron_df.columns = ["step", "top_neurons", "med_effect", "kl_diff"]
        neuron_df.to_csv(save_path)
        logger.info(f"Save file to {save_path}")


if __name__ == "__main__":
    main()
