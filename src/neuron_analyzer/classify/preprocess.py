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


def get_threshold(data_path: Path, threshold_mode: str) -> float:
    """Get threshold from the mode."""
    # load json
    threshold_dict = JsonProcessor.load_json(data_path)
    return threshold_dict[threshold_mode]["threshold"]


class LabelAnnotator:
    """Class for annotating neuron data with labels based on thresholds."""

    def __init__(self, resume: bool, threshold: float, threshold_mode: str, data_dir: Path, normalize: bool = True):
        """Initialize the LabelAnnotator."""
        self.threshold_mode = threshold_mode
        self.data_dir = data_dir
        self.threshold = threshold
        self.resume = resume
        self.normalize = normalize
        self.data = self.filter_data()

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
        """
        for index, fea in self.data["neuron_features"].items():
            logger.info(f"{index}:{len(fea)}")
        """
        values = list(self.data["neuron_features"].values())
        logger.info(set(values))
        # return np.array(list(self.data["neuron_features"].values()))
        return list(self.data["neuron_features"].values())

    def annotate_label(self) -> list:
        """Annotate labels for each neuron."""
        self.labels = []
        for delta_loss in self.data["delta_losses"].values():
            self.labels.append(self._generate_labels(delta_loss))
        # return np.array(self.labels)
        return self.labels

    def _generate_labels(self, delta_loss: float) -> dict[str, int]:
        """Generate class labels based on delta loss values and threshold."""
        if abs(delta_loss) < self.threshold:
            return 0  # Common neuron
        if abs(delta_loss) > self.threshold and delta_loss > 0:
            return 1  # Boost neuron
        if abs(delta_loss) > self.threshold and delta_loss < 0:
            return 2  # Suppress neuron
        return -1

    def run_pipeline(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Prepare data for machine learning from annotated step data."""
        # Load data if path provided
        out_path = self.data_dir / f"{self.threshold_mode}.json"
        if self.resume and out_path.is_file():
            logger.info(f"Resume existing dataset from {out_path}")
            return JsonProcessor.load_json(out_path)
        # load features and labels
        X = self.load_fea()
        y = self.annotate_label()
        # Normalize if requested
        if self.normalize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        # save the results
        return self.save_dataset(X, y, out_path)

    def save_dataset(self, X: np.ndarray, y: np.ndarray, out_path: Path) -> None:
        """Save ML dataset to disk."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"threshold": self.threshold, "threshold_mode": self.threshold_mode}
        neuron_indices = list(self.data["neuron_features"].keys())
        data = {"X": X, "y": y, "neuron_indices": neuron_indices, "metadata": metadata}
        JsonProcessor.save_json(data, out_path)
        logger.info(f"Save labeled dataset to {out_path}")
        return X, y, neuron_indices, metadata
