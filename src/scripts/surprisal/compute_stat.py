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
    parser.add_argument(
        "--eval_set",
        type=str,
        default="oxford-understand",
        help="Target eval word set",
    )
    parser.add_argument(
        "--freq_group",
        type=int,
        default=6,
        help="Number of freq bands",
    )
    return parser.parse_args()



def d_stats(x):
    """Descriptive stats for an array of values."""
    stats = {
        "mean": np.mean(x),
        "median": np.median(x),
        "min": np.min(x),
        "max": np.max(x),
        "stdev": np.std(x, ddof=1),
        "first": np.percentile(x, 25),
        "third": np.percentile(x, 75),
    }
    return stats


def bin_stats(x, N):
    """Divide the array x into N bins and compute stats for each."""
    # Sort the array
    x_sorted = np.sort(x)
    # Calculate the number of elements in each bin
    n = len(x_sorted) // N
    bins = [x_sorted[i : i + n] for i in range(0, len(x_sorted), n)]
    # Ensure we use all elements (important if len(x) is not perfectly divisible by N)
    if len(x_sorted) % N:
        bins[-2] = np.concatenate((bins[-2], bins[-1]))
        bins.pop()
    # Compute stats for each bin using get_stats
    stats_list = [d_stats(bin) for bin in bins]
    # Create DataFrame from the list of stats dictionaries
    df = pd.DataFrame(stats_list)
    idx_lst = []
    idx = 0
    for word_bin in bins:
        idx_lst.extend([idx]*len(word_bin))
        idx+=1
    return df, idx_lst



def load_group_data(cdi_path,freq_group):
    """load group path """
    # load freq file
    cdi_freq = pd.read_csv(cdi_path)
    cdi_sorted = cdi_freq.sort_values(by="freq_m")
    df, idx_lst = bin_stats(cdi_sorted["freq_m"], freq_group)
    cdi_sorted["bin_nb"] = idx_lst
    return cdi_sorted,df


def load_file(
    surprisal_path:Path
    )->pd.DataFrame:
    """Load and filter data"""
    data = pd.read_csv(surprisal_path)
    # select words from the given list
    data["log_step"] = np.log10(data["step"] + 1e-10)
    data = data[(data["log_step"] > 3.5)]
    return data


def group_word(data,group_data,stat_df):
    """Group data into different bins."""
    data_grouped = group_data.groupby("bin_nb")
    stat_df = pd.DataFrame()
    for bin_idx,data_group in data_grouped:
        median_freq_raw = stat_df["median"].tolist()[bin_idx]
        median_freq = f"{median_freq_raw:.2f}"
        data_sel = data[data["target_word"].isin(data_group["word"])]
        stat = data_sel.groupby("log_step")["surprisal"].mean().reset_index()
        stat["median_freq"] = median_freq 
        stat_df = pd.concat([stat_df,stat])
    return stat_df


def read_file(neuron_setting,file_path):
    if neuron_setting == "base":
        return True
    else:
        if int(file_path.name.split("_")[1].split(".")[0]) < 500:
            return True
        else:
            return False




def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()

    ###################################
    # load paths
    ###################################
    cdi_path = settings.PATH.dataset_root / "freq/EleutherAI/pythia-410m" / f"{args.eval_set}.csv"
    logger.info(f"Load tragte data from {cdi_path}")
    cdi_sorted,stat_df = load_group_data(cdi_path,args.freq_group)

    ###################################
    # aggregate stat by groups
    ###################################
    neuron_setting_lst = ["base","zero","random"]
    stat = pd.DataFrame()
    for neuron_setting in neuron_setting_lst:
        logger.info(f"Load {neuron_setting} surprisal data ")
        surprisal_path = settings.PATH.result_dir / "surprisal" / neuron_setting / args.eval_set / "EleutherAI"
        for file_path in surprisal_path.iterdir():

            # only select part odf them
            if read_file(neuron_setting,file_path):
                data = load_file(file_path)
                logger.info(f"Load surprisal data from {file_path}")

                # divide into differnet bands based on annotation
                df = group_word(data,cdi_sorted,stat_df)
                df["eval"] = args.eval_set
                df["para"] = file_path.name.split("-")[1]
                df["setting"] = neuron_setting
                if neuron_setting != "base":
                    df["neuron"] = file_path.name.split("_")[1].split(".")[0]
                else:
                    df["neuron"] = 0
            stat = pd.concat([df,stat])

    ###################################
    # Save the target results
    ###################################
    out_path = settings.PATH.result_dir / "surprisal" / "stat" / f"{args.eval_set}_{args.freq_group}.csv"
    stat.to_csv(out_path)
    logger.info(f"Save data to {out_path}")


if __name__ == "__main__":
    main()
