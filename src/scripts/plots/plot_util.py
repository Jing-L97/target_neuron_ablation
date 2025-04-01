import ast
import typing as t
from pathlib import Path

import numpy as np
import pandas as pd

#######################################################
# Util in plotting styles

neuron_colors = {
    0: "#1f77b4",  # blue for baseline
    1: "#4589b9",  # interpolated between 0 and 10
    2: "#6b9bbe",  # interpolated between 0 and 10
    5: "#a5bec6",  # interpolated between 0 and 10
    10: "#ff7f0e",  # orange (unchanged)
    25: "#c89f1d",  # interpolated between 10 and 50
    50: "#2ca02c",  # green (unchanged)
    500: "#9467bd",  # purple (unchanged)
}


#######################################################
# Util func to set up steps

class SurprisalLoader:
    """Loader for processing and analyzing surprisal data."""

    def __init__(
        self,
        base_path: Path,
        eval_sets: list[str] = ["merged", "longtail_words"],
        ablations: list[str] = ["mean", "zero", "random", "scaled", "full"],
        effects: list[str] = ["boost", "supress"],
        models: list[str] = ["70m", "410m"],
        neurons: list[int] = [1, 2, 5, 10, 25, 50, 500],
        vectors: list[str] = ["base", "mean", "longtail"],
        min_step: float = 3.5,
    ):
        """Initialize the SurprisalAnalyzer with configurable parameters."""
        self.base_path = base_path
        self.eval_sets = eval_sets
        self.ablations = ablations
        self.effects = effects
        self.models = models
        self.neurons = neurons
        self.vectors = vectors
        self.min_step = min_step

    def load_file(self, file_path: Path) -> pd.DataFrame:
        """Load and process a CSV file, filtering columns based on log scale."""
        # Load the data
        data = pd.read_csv(file_path)

        # Create columns dictionary to build final DataFrame efficiently
        columns_dict = {"target_word": data["target_word"]}

        # Process numeric columns and collect them in a dictionary
        for col in data.columns:
            if col.isdigit():
                # Convert column name to log scale
                log_value = np.log10(float(col) + 1e-10)
                # Only keep columns with log value > min_step
                if log_value > self.min_step:
                    # Add to columns dictionary with log value as key
                    columns_dict[f"{log_value:.4f}"] = data[col]

        # Create DataFrame from dictionary at once (avoids fragmentation)
        result = pd.DataFrame(columns_dict)

        return result

    def group_word_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean values across all words, dropping the target_word column."""
        df_group = data.drop(columns=["target_word"])
        result = df_group.mean()

        # Create DataFrame efficiently using a dictionary
        df = pd.DataFrame({"surprisal": result})
        df = df.reset_index().rename(columns={"index": "log_step"})

        return df

    def process_all(self, file_path: Path, header_dict: dict[str, t.Any]) -> pd.DataFrame:
        """Process a file and add metadata columns."""
        data = self.load_file(file_path)
        # Group data
        stat = self.group_word_all(data)
        # Add all metadata columns at once by creating a new DataFrame with
        # repeated metadata values and concatenating horizontally
        metadata_df = pd.DataFrame({key: [val] * len(stat) for key, val in header_dict.items()})
        # Concatenate horizontally to avoid fragmentation
        stat = pd.concat([stat, metadata_df], axis=1)
        return stat

    def get_stat_all(self) -> pd.DataFrame:
        """Process all data files according to the configured parameters."""
        # Use a list to collect DataFrames and concatenate once at the end
        stat_frames = []

        # Loop through all parameter combinations
        for effect in self.effects:
            for eval_set in self.eval_sets:
                for model in self.models:
                    for vec in self.vectors:
                        if vec == "base":
                            # Base vector case is handled differently
                            header_dict = {
                                "vec": vec,
                                "neuron": 0,
                                "model": model,
                                "ablation": "base",
                                "eval": eval_set,
                                "effect": effect,
                            }

                            file_path = self.base_path / vec / eval_set / "EleutherAI" / f"pythia-{model}-deduped.csv"

                            if file_path.is_file():
                                stat = self.process_all(file_path, header_dict)
                                stat_frames.append(stat)

                        else:
                            # Process all ablation and neuron combinations
                            for ablation in self.ablations:
                                for neuron in self.neurons:
                                    file_path = (
                                        self.base_path
                                        / effect
                                        / vec
                                        / ablation
                                        / eval_set
                                        / "EleutherAI"
                                        / f"pythia-{model}-deduped_{neuron}.csv"
                                    )

                                    header_dict = {
                                        "vec": vec,
                                        "neuron": neuron,
                                        "model": model,
                                        "ablation": ablation,
                                        "eval": eval_set,
                                        "effect": effect,
                                    }

                                    if file_path.is_file():
                                        stat = self.process_all(file_path, header_dict)
                                        stat_frames.append(stat)

        # Concatenate all frames at once (more efficient than incremental concatenation)
        stat_frame = pd.concat(stat_frames, ignore_index=True) if stat_frames else pd.DataFrame()

        # Save the result
        output_path = self.base_path / "stat_all.csv"
        stat_frame.to_csv(output_path)
        print(f"Stat file has been saved to {output_path}")

        return stat_frame




def load_kl(file_path: Path) -> list[float]:
    """Load the kl_diff column from a CSV file and convert it to a flat list of floats."""
    df = pd.read_csv(file_path)
    kl_diff_strings = df["kl_diff"].tolist()
    # Convert each string representation of a list to an actual list
    kl_diff_lists = [ast.literal_eval(kl_diff_str) for kl_diff_str in kl_diff_strings]
    # Flatten the list of lists into a single list of floats
    flat_kl_diff = [float(val) for sublist in kl_diff_lists for val in sublist]
    # get the stats o fthe given list
    kl_stat = d_stats(flat_kl_diff)
    stat_df = pd.DataFrame(kl_stat, index=[0])
    return flat_kl_diff, stat_df



#######################################################
# Util func in stat

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
