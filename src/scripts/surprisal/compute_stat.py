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

def load_file(file_path: Path) -> pd.DataFrame:
    # Load the data
    data = pd.read_csv(file_path)
    # Create a new DataFrame for the result
    result = data[["target_word"]]
    # Then process numeric columns
    for col in data.columns:
        if col.isdigit():
            # Convert column name to log scale
            log_value = np.log10(float(col) + 1e-10)
            
            # Only keep columns with log value > 3.5
            if log_value > 3.5:
                # Use the log value as the new column name
                result[f"{log_value:.4f}"] = data[col]
    
    return result

def load_group_data(cdi_path):
    """load group path """
    # load freq file
    cdi_freq = pd.read_csv(cdi_path)
    cdi_sorted = cdi_freq.sort_values(by="freq_m")
    df, idx_lst = bin_stats(cdi_sorted["freq_m"], 6)
    cdi_sorted["bin_nb"] = idx_lst
    return cdi_sorted,df

def group_word(data,group_data,stat_df)->dict:
    """Group data into different bins."""
    data_grouped = group_data.groupby("bin_nb")
    group_dict = {}
    for bin_idx,data_group in data_grouped:
        median_freq = f"{stat_df["median"].tolist()[bin_idx]:.2f}"
        df_group = data[data["target_word"].isin(data_group["word"])]
        # remove the word column
        df_group = df_group.drop(columns=["target_word"])
        group_dict[median_freq] = df_group.mean()
    df = pd.DataFrame(group_dict)
    df = df.reset_index().rename(columns={'index': 'log_step'})
    return df






def main() -> None:
    """Main function demonstrating usage."""
    args = parse_args()

        
    # load group file
    # load file as a dict
    cdi_path = freq_path/"cdi_childes.csv"
    group_data,stat_df = load_group_data(cdi_path)
    eval_set = "cdi_childes"


    ablation_lst = ["base","mean","zero","random"]

    model_lst = ["70m","410m"]
    neuron_lst = [10,50,100,500]

    stat_frame = pd.DataFrame()
    # loop ablation conditions
    for ablation in ablation_lst:
        for model in model_lst:
            if ablation == "base":
                file_path = surprisal_path/ablation/eval_set/"EleutherAI"/f"pythia-{model}-deduped.csv"
                data = load_file(file_path)
                # group data
                stat = group_word(data,group_data,stat_df)
                for header,col_val in {"neuron":0,"model":model, "ablation":ablation}.items():
                    stat[header]=col_val
                stat_frame = pd.concat([stat_frame,stat])

            else:
                for neuron in neuron_lst:
                    file_path = surprisal_path/ablation/eval_set/"EleutherAI"/f"pythia-{model}-deduped_{neuron}.csv"
                    data = load_file(file_path)
                    # group data
                    stat = group_word(data,group_data,stat_df)
                    for header,col_val in {"neuron":neuron,"model":model, "ablation":ablation}.items():
                        stat[header]=col_val
                    stat_frame = pd.concat([stat_frame,stat])



    ###################################
    # Save the target results
    ###################################
    out_path = settings.PATH.result_dir / "surprisal" / "stat" / f"{args.eval_set}_{args.freq_group}.csv"
    stat.to_csv(out_path)
    logger.info(f"Save data to {out_path}")


if __name__ == "__main__":
    main()
