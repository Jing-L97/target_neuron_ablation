import logging
import typing as t
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from neuron_analyzer.load_util import JsonProcessor

T = t.TypeVar("T")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


#######################################################################################################
# Class to load neuron features


class FeatureLoader:
    """Class for loading neuron features."""

    def __init__(self, data_dir: Path):
        """Initialize the LabelAnnotator."""
        self.data_dir = data_dir

    def filter_data(self) -> dict:
        """Filter dictionary to keep only features with the most common length."""
        # Load step data
        data = JsonProcessor.load_json(self.data_dir / "features.json")
        # Extract lengths and indices
        len_lst = []
        index_lst = []
        for index, fea in data["neuron_features"].items():
            len_lst.append(len(fea))
            index_lst.append(index)

        # Count occurrences of each length
        length_counts = Counter(len_lst)
        # logger.info(f"Feature length distribution: {dict(length_counts)}")

        # Find the most common length
        most_common_length = length_counts.most_common(1)[0][0]
        # logger.info(f"Most common feature length: {most_common_length}")

        # Filter the data to include only features with the most common length
        self.data = {"neuron_features": {}, "delta_losses": {}}
        for index, fea in data["neuron_features"].items():
            if len(fea) == most_common_length:
                self.data["neuron_features"][index] = fea
                # Also get the corresponding delta loss if it exists
                if index in data["delta_losses"]:
                    self.data["delta_losses"][index] = data["delta_losses"][index]

        # logger.info(f"Original data had {len(data['neuron_features'])} features")
        # logger.info(f"Filtered data has {len(self.data['neuron_features'])} features")
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

    def __init__(self, data: dict, class_indices: dict):
        """Initialize the LabelAnnotator."""
        self.data = data
        self.class_indices = class_indices

    def run_pipeline(self) -> list:
        """Annotate labels for each neuron."""
        self.labels = []
        self.feas = []
        self.indices = []
        for neuron_index, fea in self.data["neuron_features"].items():
            if self._generate_labels(neuron_index):
                self.labels.append(self._generate_labels(neuron_index))
                self.feas.append(fea)
                self.indices.append(neuron_index)
        return np.array(self.feas), np.array(self.labels), self.indices

    def _generate_labels(self, neuron_index: int) -> int:
        """Generate class labels based on predefined class dict."""
        if int(neuron_index) in self.class_indices["random"]:
            return -1
        if int(neuron_index) in self.class_indices["boost"]:
            return 1
        if int(neuron_index) in self.class_indices["suppress"]:
            return 2
        return None


#######################################################################################################
# Class to label neuron classes by thredholds


class DataLoader:
    """Class to load dataset."""

    def __init__(
        self, X: np.array, y: np.array, neuron_indices: list, resume: bool, out_path: Path, normalize: bool = True
    ):
        """Initialize the LabelAnnotator."""
        self.out_path = out_path
        self.resume = resume
        self.normalize = normalize
        self.indices = neuron_indices
        self.X = X
        self.y = y

    def run_pipeline(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Prepare data for machine learning from annotated step data."""
        # Load data if path provided
        if self.resume and self.out_path.is_file():
            logger.info(f"Resume existing dataset from {self.out_path}")
            return JsonProcessor.load_json(self.out_path)
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
