import logging
import typing as t
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
        feather_path: Path,
        entropy_path: Path,
        step_path: Path,
        out_dir: Path,
        step_num: str,
        device: str,
        unigram_analyzer,
    ):
        """Initialize the NeuronFeatureExtractor."""
        self.args = args
        self.feather_path = feather_path
        self.entropy_path = entropy_path
        self.step_path = step_path
        self.device = device
        self.unigram_analyzer = unigram_analyzer
        self.step_num = str(step_num)
        # configure out path dir
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def load_delta_loss(self) -> pd.DataFrame:
        """Load and filter feather data."""
        out_path = self.out_dir / f"k{self.args.k}.feather"
        if out_path.is_file() and self.args.resume:
            logger.info(f"Resuming file from {out_path}")
            self.loss_data = pd.read_feather(out_path)
            return self.loss_data

        feather_path = self.feather_path / self.step_num / str(self.args.data_range_end) / f"k{self.args.k}.feather"
        if feather_path.is_file():
            group_analyzer = NeuronGroupAnalyzer(
                args=self.args,
                feather_path=feather_path,
                unigram_analyzer=self.unigram_analyzer,
                step_path=self.step_path,
                abl_path=self.entropy_path,
                device=self.device,
            )
            self.loss_data = group_analyzer.load_activation_df()

        # Save the selected intermediate data
        self.loss_data.reset_index(drop=True).to_feather(out_path)
        return self.loss_data

    def load_fea(self) -> pd.DataFrame:
        """Load and filter feather data."""
        out_path = self.out_dir / "entropy_df.feather"
        if out_path.is_file() and self.args.resume:
            logger.info(f"Resuming file from {out_path}")
            fea_data = pd.read_feather(out_path)
            return fea_data

        entropy_path = self.entropy_path / self.step_num / str(self.args.data_range_end) / "entropy_df.csv"
        if entropy_path.is_file():
            logger.info(f"Loading file from {entropy_path}")
            # filter file
            df_filter = NeuronSelector(
                feather_path=entropy_path,
                sel_freq=self.args.sel_freq,
                unigram_analyzer=self.unigram_analyzer,
                debug=self.args.debug,
                threshold_path=self.entropy_path / self.args.stat_file,
            )
            fea_data = df_filter.filter_df_by_freq()

            # load file by freq
            fea_data = self._filter_token(fea_data, groupby_col="str_tokens", sort_by_col="freq")
            # Save the selected intermediate data
            fea_data.reset_index(drop=True).to_feather(out_path)
        else:
            logger.info(f"No file from {entropy_path}")
        return fea_data

    def _filter_token(self, fea_data, groupby_col: str, sort_by_col: str) -> pd.DataFrame:
        """Filter top-n tokens together with the rows."""
        first_n_groups = fea_data[groupby_col].drop_duplicates().head(self.args.fea_dim)
        filtered_df = fea_data[fea_data[groupby_col].isin(first_n_groups)]
        if "longtail" in self.args.sel_freq:
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
                delta_losses[neuron_idx] = float(neuron_data["delta_loss_post_ablation"].mean())
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
        fea_data = self.load_fea()
        logger.info("Finish loading fea data.")
        loss_data = self.load_delta_loss()
        logger.info("Finish loading loss data.")
        # Calculate features and losses if not provided
        neuron_features, delta_losses = self.build_vector(loss_data, fea_data)
        # Prepare the results dictionary
        results = {
            "step_num": self.step_num,
            "neuron_features": neuron_features,
            "delta_losses": delta_losses,
            "metadata": {
                "feature_count": len(next(iter(neuron_features.values()))),
                "neuron_count": len(neuron_features),
            },
        }
        # TDOO: add additional info: context, original_loss, ablation_loss,
        JsonProcessor.save_json(results, out_path)
        logger.info(f"Save file to {out_path}")
        return results


#######################################################################################################
# Class to load neuron features


class FeatureLoader:
    """Class for loading neuron features."""

    def __init__(self, data_path: Path):
        """Initialize the LabelAnnotator."""
        self.data_path = data_path

    def filter_data(self) -> dict:
        """Filter dictionary to keep only features with the most common length."""
        # Load step data
        data = JsonProcessor.load_json(self.data_path)
        # Extract lengths and indices
        len_lst = []
        index_lst = []
        for index, fea in data["neuron_features"].items():
            len_lst.append(len(fea))
            index_lst.append(index)

        # Count occurrences of each length
        length_counts = Counter(len_lst)
        logger.info(f"Feature length distribution: {dict(length_counts)}")

        # Find the most common length
        most_common_length = length_counts.most_common(1)[0][0]
        logger.info(f"Most common feature length: {most_common_length}")

        # Filter the data to include only features with the most common length
        self.data = {"neuron_features": {}, "delta_losses": {}}
        for index, fea in data["neuron_features"].items():
            if len(fea) == most_common_length:
                self.data["neuron_features"][index] = fea
                # Also get the corresponding delta loss if it exists
                if index in data["delta_losses"]:
                    self.data["delta_losses"][index] = data["delta_losses"][index]

        logger.info(f"Original data had {len(data['neuron_features'])} features")
        logger.info(f"Filtered data has {len(self.data['neuron_features'])} features")
        return self.data

    def load_fea(self) -> list:
        """Load feature vectors."""
        return np.array(list(self.data["neuron_features"].values()))

    def run_pipeline(self) -> tuple[dict, np.array]:
        """Run pipeline of feature loading."""
        self.data = self.filter_data()
        fea = self.load_fea()
        return self.data, fea


#######################################################################################################
# Class to label neuron classes by thresholds


class ThresholdLabeler:
    """Class for annotating neuron data with labels based on thresholds."""

    def __init__(self, threshold: float, data: dict):
        """Initialize the LabelAnnotator."""
        self.threshold = threshold
        self.data = data

    def run_pipeline(self) -> list:
        """Annotate labels for each neuron."""
        self.labels = []
        for delta_loss in self.data["delta_losses"].values():
            self.labels.append(self._generate_labels(delta_loss))
        return np.array(self.labels), list(self.data["delta_losses"].keys())

    def _generate_labels(self, delta_loss: float) -> int:
        """Generate class labels based on delta loss values and threshold; maximize class info."""
        if abs(delta_loss) < self.threshold:
            return -1  # Common neuron
        if abs(delta_loss) > self.threshold and delta_loss > 0:
            return 1  # Boost neuron
        if abs(delta_loss) > self.threshold and delta_loss < 0:
            return 2  # Suppress neuron
        return None


def get_threshold(data_path: Path, threshold_mode: str) -> float:
    """Get threshold from the mode."""
    threshold_dict = JsonProcessor.load_json(data_path)
    return threshold_dict[threshold_mode]["threshold"]


#######################################################################################################
# Class to label neuron classes by given indices


class FixedLabeler:
    """Class for annotating neuron data with predefined labels; maximize class info."""

    def __init__(self, data: dict, class_indices: dict, run_baseline: bool):
        """Initialize the LabelAnnotator."""
        self.data = data
        self.class_indices = class_indices
        self.run_baseline = run_baseline

    def run_pipeline(self) -> list:
        """Annotate labels for each neuron."""
        self.labels = []
        self.feas = []
        self.indices = []
        for neuron_index, fea in self.data["neuron_features"].items():
            if self._annotate_labels(neuron_index):
                self.labels.append(self._annotate_labels(neuron_index))
                self.feas.append(fea)
                self.indices.append(neuron_index)
        return np.array(self.feas), np.array(self.labels), self.indices

    def _annotate_labels(self, neuron_index: int):
        """Generate class labels based on differnet conditions."""
        if self.run_baseline:
            return self._generate_baseline_labels(neuron_index)
        return self._generate_labels(neuron_index)

    def _generate_labels(self, neuron_index: int) -> int:
        """Generate class labels based on predefined class dict."""
        if int(neuron_index) in self.class_indices["random"]:
            return -1
        if int(neuron_index) in self.class_indices["boost"]:
            return 1
        if int(neuron_index) in self.class_indices["suppress"]:
            return 2
        return None

    def _generate_baseline_labels(self, neuron_index: int) -> int:
        """Generate class labels based on predefined class dict for baseline condition."""
        if int(neuron_index) in self.class_indices["random"]:
            return -1
        if str(neuron_index) in self.class_indices["baseline"]:
            return 1
        return None


#######################################################################################################
# Class to label neuron classes by thredholds


class DataLoader:
    """Class to load dataset."""

    def __init__(self, X: np.array, y: np.array, neuron_indices: list, out_path: Path, normalize: bool = True):
        """Initialize the LabelAnnotator."""
        self.out_path = out_path
        self.normalize = normalize
        self.indices = neuron_indices
        self.X = X
        self.y = y

    def run_pipeline(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Prepare data for machine learning from annotated step data."""
        # Normalize if requested
        if self.normalize:
            scaler = StandardScaler()
            self.X = scaler.fit_transform(self.X)
        # save the results
        self._save_dataset()
        return self.X, self.y, self.indices

    def _save_dataset(self) -> None:
        """Save ML dataset to disk."""
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"X": self.X, "y": self.y, "neuron_indices": self.indices}
        JsonProcessor.save_json(data, self.out_path)
        logger.info(f"Save labeled dataset to {self.out_path}")
