import typing as t
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from neuron_analyzer.load_util import JsonProcessor

T = t.TypeVar("T")


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
        # Load step data
        self.data = JsonProcessor.load_json(self.data_dir / "features.json")

    def load_feal(self) -> list:
        """Load feature vectors."""
        return np.array(self.data["neuron_features"].values())

    def annotate_label(self) -> list:
        """Annotate labels for each neuron."""
        self.labels = []
        for _, delta_loss in self.data.items():
            self.labels.append(self._generate_labels(delta_loss))
        return np.array(self.labels)

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
        return X, y, neuron_indices, metadata
