import logging
import typing as t
from pathlib import Path

import numpy as np
import pandas as pd

from neuron_analyzer.analysis.geometry_util import NeuronGroupAnalyzer
from neuron_analyzer.load_util import JsonProcessor
from neuron_analyzer.selection.neuron import NeuronSelector

T = t.TypeVar("T")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

#######################################################################################################
# Extract neuron features from activation


class NeuronFeatureExtractor:
    """Class for loading and extracting neuron features from raw data."""

    def __init__(
        self,
        args: t.Any,
        abl_path: Path,
        step_path: Path,
        out_dir: Path,
        step_num: str,
        fea_dim: int,
        sel_freq: str,
        device: str,
    ):
        """Initialize the NeuronFeatureExtractor."""
        self.args = args
        self.abl_path = abl_path
        self.step_path = step_path
        self.device = device
        self.sel_freq = sel_freq
        self.step_num = step_num
        self.fea_dim = fea_dim
        # configure out path dir
        self.out_dir = out_dir / self.step_num / str(self.args.data_range_end)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def load_delta_loss(self) -> pd.DataFrame:
        """Load and filter feather data."""
        out_path = self.out_dir / f"k{self.args.k}.feather"
        if out_path.is_file() and self.args.resume:
            self.loss_data = pd.read_feather(out_path)
            return self.loss_data

        feather_path = self.abl_path / self.step_num / str(self.args.data_range_end) / f"k{self.args.k}.feather"
        if feather_path.is_file():
            group_analyzer = NeuronGroupAnalyzer(
                args=self.args,
                feather_path=feather_path,
                step_path=self.step_path,
                abl_path=self.abl_path,
                device=self.device,
            )
            self.loss_data = group_analyzer.load_activation_df()

        # Save the selected intermediate data
        self.loss_data.reset_index(drop=True).to_feather(out_path)
        return self.loss_data

    def load_fea(self) -> pd.DataFrame:
        """Load and filter feather data."""
        out_path = self.out_dir / "entropy_df.csv"
        if out_path.is_file() and self.args.resume:
            self.fea_data = pd.read_csv(out_path)
            return self.fea_data

        entropy_path = self.abl_path / self.step_num / str(self.args.data_range_end) / "entropy_df.csv"
        if entropy_path.is_file():
            # filter file
            df_filter = NeuronSelector(feather_path=entropy_path, sel_freq=self.sel_freq, device="cpu")
            self.activation_data = df_filter.filter_df_by_freq()
            # load file by freq
            self.fea_data = self._filter_token(groupby_col="str_tokens", sort_by_col="freq")
        # Save the selected intermediate data
        self.activation_data.reset_index(drop=True).to_csv(out_path)
        return self.fea_data

    def _filter_token(self, groupby_col: str, sort_by_col: str) -> pd.DataFrame:
        """Filter top-n tokens together with the rows."""
        first_n_groups = self.df[groupby_col].drop_duplicates().head(self.fea_dim)
        filtered_df = self.df[self.df[groupby_col].isin(first_n_groups)]
        if "longtail" in self.sel_freq:
            return (
                filtered_df.sort_values(by=sort_by_col, ascending=True).groupby(groupby_col, group_keys=False).head(1)
            )
        return filtered_df.sort_values(by=sort_by_col, ascending=False).groupby(groupby_col, group_keys=False).head(1)

    def build_vector(
        self, loss_data: pd.DataFrame, fea_data: pd.DataFrame
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Load and process data from the filtered dataframe."""
        # Group by neuron index (component_name)
        neuron_indices = loss_data["component_name"].unique()
        # Dictionary to store feature vector for each neuron
        neuron_features: dict[str, np.ndarray] = {}
        # Dictionary to store delta losses for each neuron
        delta_losses: dict[str, float] = {}
        for neuron_idx in neuron_indices:
            # Extract activation values as features
            col_header = self._get_column_name(fea_data.columns, neuron_idx)
            if col_header:
                neuron_features[neuron_idx] = fea_data[col_header].to_list()
                # Get all delta loss rows for this neuron
                neuron_data = loss_data[loss_data["component_name"] == neuron_idx]
                delta_losses[neuron_idx] = float(neuron_data["delta_loss_post_ablation"].values.mean())
        return neuron_features, delta_losses

    def _get_column_name(self, columns, idx):
        """Filter column header given part of the index."""
        for col in columns:
            try:
                decimal_part = int(col.split(".")[1].split("_")[0])
                if decimal_part == idx:
                    return col
            except (IndexError, ValueError):
                continue
        return None

    def run_pipeline(self) -> dict:
        """Extract and save features and delta losses for a single step."""
        # resume logic
        out_path = self.out_dir / "features.json"
        if self.args.resume and out_path.is_file():
            # load file for the optimal selection
            logger.info(f"Load existing file from {out_path}")
            return JsonProcessor.load_json(out_path)

        # Load data if not already loaded
        loss_data = self.load_delta_loss()
        fea_data = self.load_fea()
        # Calculate features and losses if not provided
        neuron_features, delta_losses = self.build_vector(loss_data, fea_data)
        # Prepare the results dictionary
        results = {
            "step_num": self.step_num,
            "neuron_features": {k: v.tolist() for k, v in neuron_features.items()},
            "delta_losses": delta_losses,
            "metadata": {
                "feature_count": len(next(iter(neuron_features.values()))),
                "neuron_count": len(neuron_features),
            },
        }
        JsonProcessor.save_json(results, out_path)
        logger.info(f"Save file to {out_path}")
        return results
