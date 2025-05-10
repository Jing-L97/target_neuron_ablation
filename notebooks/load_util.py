import ast
import typing as t
from pathlib import Path

import matplotlib.pyplot as plt
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
        effects: list[str] = ["boost", "suppress"],
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
        return pd.DataFrame(columns_dict)

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
        stat_frame["log_step"] = stat_frame["log_step"].astype(float)
        # Save the result
        output_path = self.base_path / "stat_all.csv"
        stat_frame.to_csv(output_path)
        print(f"Stat file has been saved to {output_path}")

        return stat_frame


class GeometryLoader:
    """Class for loading and processing geometric metric data."""

    def __init__(self, min_step: float = 3.5):
        self.min_step = min_step

    @staticmethod
    def convert_log(step: float) -> float:
        return np.log10(step + 1e-10)

    def convert_log_step(self, file_path: Path) -> pd.DataFrame:
        # Load the data
        data = pd.read_csv(file_path)
        # Apply log conversion
        data["log_step"] = data["step"].apply(self.convert_log)
        # Filter by minimum log step
        return data[data["log_step"] > self.min_step].copy()

    def load_subspace(self, data: pd.DataFrame, neuron_type_lst=None) -> pd.core.groupby.DataFrameGroupBy:
        """Process subspace data with string replacements and filtering."""
        # Create a copy to avoid SettingWithCopyWarning
        df = data.copy()
        # Replace neuron type strings
        df["neuron"] = df["neuron"].str.replace("sampled_common", "random", regex=False)
        df["neuron"] = df["neuron"].str.replace("common", "all", regex=False)
        # Filter by neuron type if list provided
        if neuron_type_lst:
            df = df[~df["neuron"].isin(neuron_type_lst)]
        # Calculate dimension proportion
        df["dim_prop"] = df["effective_dim"] / df["total_dim"]
        return df.groupby("neuron")

    def load_orthogonality(
        self, data: pd.DataFrame, neuron_type_lst = None
    ) -> pd.core.groupby.DataFrameGroupBy:
        """Process orthogonality data with string replacements and filtering."""
        # Create a copy to avoid SettingWithCopyWarning
        df = data.copy()
        # Replace pair strings (order matters - replace longer pattern first)
        df["pair"] = df["pair"].str.replace("sampled_common", "random", regex=False)
        df["pair"] = df["pair"].str.replace("common", "all", regex=False)
        # Filter by neuron type if list provided
        if neuron_type_lst:
            # Exclude pairs containing any item from neuron_type_lst
            df = df[~df["pair"].apply(lambda pair: any(item in pair for item in neuron_type_lst))]
        return df.groupby("pair")

    def load_file(self, file_path: Path, metric: str, neuron_type_lst = None) :
        """Main function to load and process metric data files."""
        # Load data with log step conversion
        data = self.convert_log_step(file_path)
        # Route to appropriate processing function based on metric
        if metric == "subspace":
            return self.load_subspace(data, neuron_type_lst=neuron_type_lst)
        if metric == "orthogonality":
            return self.load_orthogonality(data, neuron_type_lst=neuron_type_lst)
        return None



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




#######################################################
# Util func in plotting surprisal

class SurprisalPlotter:
    """Class for plotting model effect data with various filtering options."""

    def __init__(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        neuron_colors: dict[int, str],
        ylim_dict: dict[str, dict[str, tuple[float, float]]],
        models=None,
        effect_lst=None,
        vec_lst=None,
        ablations=None,
        neurons=None,
    ):
        """Initialize the plotter with data and configuration."""
        self.df = df
        self.output_dir = Path(output_dir)
        self.neuron_colors = neuron_colors
        self.ylim_dict = ylim_dict

        # Set models or compute from df if None
        self.models = models if models is not None else df["model"].unique().tolist()
        self.effect_lst = effect_lst if effect_lst is not None else df["effect"].unique().tolist()
        self.vec_lst = vec_lst if vec_lst is not None else df["vec"].unique().tolist()
        # Make sure "base" is not included in ablations to avoid creating "base_*.png" files
        if ablations is not None:
            self.ablations = [a for a in ablations if a != "base"]
        else:
            self.ablations = [a for a in df["ablation"].unique() if a != "base"]
        self.neurons = neurons if neurons is not None else df["neuron"].unique().tolist()

    def _plot_line(self, data: pd.DataFrame, label: str) -> bool:
        """ Plot a single line for the given data."""
        if data.empty:
            return False

        baseline_grouped = data.groupby("log_step")
        x_values = sorted(data["log_step"].unique())

        # Extract surprisal values
        y_values = [
            baseline_grouped.get_group(log_step)["surprisal"].values[0]
            for log_step in x_values
            if log_step in baseline_grouped.groups
        ]

        if not y_values:  # Skip if no values to plot
            return False

        # Determine color based on label
        if label == "baseline":
            color = self.neuron_colors.get(0, "black")
        else:
            try:
                neuron_id = int(label)
                color = self.neuron_colors.get(neuron_id, "black")
            except ValueError:
                color = "black"

        # Plot the line
        plt.plot(x_values, y_values, color=color, linewidth=2, label=label)
        return True

    def plot_all(self, eval_set: str, figure_size: tuple[int, int] = (10, 8)) -> list[Path]:
        """Plot the overall development using the configuration from initialization. """

        # Process each model and ablation type using class attributes
        for effect in self.effect_lst:
            for vec in self.vec_lst:
                for model in self.models:
                    for ablation in self.ablations:
                        # Create a new figure
                        plt.figure(figsize=figure_size)

                        # Get baseline data (always include baseline for comparison)
                        baseline_data = self.df[
                            (self.df["model"] == model)
                            & (self.df["ablation"] == "base")
                            & (self.df["eval"] == eval_set)
                            & (self.df["effect"] == effect)
                        ]

                        # Plot baseline data first
                        baseline_plotted = self._plot_line(baseline_data, "baseline")

                        # Count how many lines we've plotted
                        lines_plotted = 1 if baseline_plotted else 0

                        # Filter data for this model and configuration
                        model_data = self.df[
                            (self.df["model"] == model)
                            & (self.df["vec"] == vec)
                            & (self.df["eval"] == eval_set)
                            & (self.df["effect"] == effect)
                            & (self.df["ablation"] == ablation)
                        ]

                        if model_data.empty and not baseline_plotted:
                            plt.close()
                            continue

                        # Process each neuron condition for this ablation
                        for neuron in self.neurons:
                            # Filter data for this neuron and ablation combination
                            condition_data = model_data[(model_data["neuron"] == neuron)]

                            # Plot neuron data
                            if self._plot_line(condition_data, str(neuron)):
                                lines_plotted += 1

                        # Check if we have any plotted data or if it's a base ablation
                        if lines_plotted == 0 or ablation == "base":
                            plt.close()
                            continue

                        # Style the plot
                        plt.xlabel("Log step", fontsize=12)
                        plt.ylabel("Surprisal", fontsize=12)
                        plt.title(f"neuron={effect}, vec={vec}, intervention={ablation}", fontsize=13)
                        plt.grid(alpha=0.2)

                        # Create legend with baseline first
                        handles, labels = plt.gca().get_legend_handles_labels()

                        if handles:  # Only create legend if we have items to show
                            # If baseline is in the legend, make sure it comes first
                            if "baseline" in labels:
                                base_idx = labels.index("baseline")
                                # Move baseline to front
                                handles = [handles[base_idx]] + [h for i, h in enumerate(handles) if i != base_idx]
                                labels = [labels[base_idx]] + [l for i, l in enumerate(labels) if i != base_idx]

                            plt.legend(handles, labels, loc="lower left")

                        # Set y-axis limits if provided
                        if eval_set in self.ylim_dict and model in self.ylim_dict[eval_set]:
                            plt.ylim(self.ylim_dict[eval_set][model])

                        # Save the figure
                        plt.tight_layout()

                        # Create output directory if it doesn't exist
                        output_path = self.output_dir / effect / eval_set
                        output_path.mkdir(parents=True, exist_ok=True)

                        # Final check to absolutely make sure we never save any files with "base" in the name
                        if vec != "base":
                            output_file = output_path / f"{vec}_{model}_{ablation}.png"
                            plt.savefig(output_file, dpi=300, bbox_inches="tight")

                        plt.close()




def plot_geometry_step(geometry_path,output_path, metric, model_lst, neuron_lst, neuron_type_lst, ylim_dict, metric_dict):
    for model in model_lst:
        for neuron in neuron_lst:
            file_path = geometry_path / f"pythia-{model}-deduped" / metric / f"500_{neuron}.csv"
            geometry_loader = GeometryLoader()
            data_grouped = geometry_loader.load_file(file_path, metric=metric, neuron_type_lst=neuron_type_lst)
            for metric_val in metric_dict[metric]:
                for neuron_type, data_group in data_grouped:
                    plt.grid(alpha=0.2)
                    plt.plot(data_group["log_step"], data_group[metric_val], label=neuron_type)
                    plt.title(f"{metric_val}: #neuron={neuron}, model={model}")
                plt.ylim(ylim_dict[metric_val])
                plt.xlabel("Log step", fontsize=12)
                plt.ylabel(metric_val, fontsize=12)
                plt.legend()
                output_file = output_path / metric_val / f"{model}_{neuron}.png"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_file, dpi=300, bbox_inches="tight")
                plt.close()
