import json
import pickle
import typing as t
from pathlib import Path

import numpy as np

T = t.TypeVar("T")


class LabelAnnotator:
    """Class for annotating neuron data with labels based on thresholds."""

    def __init__(self, threshold: float | None = None):
        """Initialize the LabelAnnotator.

        Args:
            threshold: Threshold value to use for classification

        """
        self.threshold = threshold

    def set_threshold(self, threshold: float) -> None:
        """Set the threshold value.

        Args:
            threshold: Threshold value to use

        """
        self.threshold = threshold

    def load_threshold_analysis(self, file_path: Path) -> float:
        """Load threshold analysis and set the best threshold.

        Args:
            file_path: Path to threshold analysis file

        Returns:
            The best threshold value

        """
        with open(file_path) as f:
            analysis = json.load(f)

        self.threshold = analysis["best_threshold"]
        return self.threshold

    def generate_labels(
        self, delta_losses: dict[str, float], threshold: float | None = None, mode: str = "triclass"
    ) -> dict[str, int]:
        """Generate class labels based on delta loss values and threshold.

        Args:
            delta_losses: Dictionary mapping neuron indices to delta loss values
            threshold: Threshold value for classification (overrides instance threshold if provided)
            mode: Classification mode - "triclass" (0: common, 1: boost, 2: suppress) or
                 "binary" (0: common, 1: special)

        Returns:
            Dictionary mapping neuron indices to class labels

        """
        if threshold is None:
            threshold = self.threshold

        if threshold is None:
            raise ValueError("Threshold not set. Call set_threshold or provide a threshold value.")

        labels: dict[str, int] = {}

        for neuron_idx, loss in delta_losses.items():
            if abs(loss) < threshold:
                # Common neuron
                labels[neuron_idx] = 0
            elif loss > threshold:
                # Boost neuron
                labels[neuron_idx] = 1 if mode == "triclass" else 1
            else:
                # Suppress neuron
                labels[neuron_idx] = 2 if mode == "triclass" else 1

        return labels

    def annotate_step_data(
        self, step_file: Path, output_path: Path | None = None, threshold: float | None = None, mode: str = "triclass"
    ) -> dict:
        """Load step data, annotate it with labels, and save the result.

        Args:
            step_file: Path to step data file
            output_path: Path to save annotated data (if None, returns without saving)
            threshold: Threshold value (overrides instance threshold if provided)
            mode: Classification mode

        Returns:
            Dictionary with annotated data

        """
        # Use provided threshold or instance threshold
        if threshold is not None:
            effective_threshold = threshold
        elif self.threshold is not None:
            effective_threshold = self.threshold
        else:
            raise ValueError("Threshold not set. Call set_threshold or provide a threshold value.")

        # Load step data
        with open(step_file) as f:
            step_data = json.load(f)

        # Generate labels
        delta_losses = step_data["delta_losses"]
        labels = self.generate_labels(delta_losses, effective_threshold, mode)

        # Add labels to step data
        step_data["labels"] = labels
        step_data["threshold_used"] = effective_threshold
        step_data["label_mode"] = mode

        # Save if output path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(step_data, f, indent=2)

        return step_data

    def prepare_ml_dataset(
        self, annotated_data: dict | Path, normalize: bool = True
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Prepare data for machine learning from annotated step data.

        Args:
            annotated_data: Dictionary with annotated data or path to file
            normalize: Whether to normalize features

        Returns:
            Tuple of (X, y, neuron_indices)

        """
        # Load data if path provided
        if isinstance(annotated_data, Path):
            with open(annotated_data) as f:
                data = json.load(f)
        else:
            data = annotated_data

        # Extract features, labels and indices
        neuron_indices = list(data["neuron_features"].keys())

        # Convert features from lists to arrays if needed
        X = np.array([np.array(data["neuron_features"][idx]) for idx in neuron_indices])

        y = np.array([data["labels"][idx] for idx in neuron_indices])

        # Normalize if requested
        if normalize:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        return X, y, neuron_indices

    def save_ml_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        neuron_indices: list[str],
        output_path: Path,
        metadata: dict | None = None,
        format: str = "npz",
    ) -> None:
        """Save ML dataset to disk.

        Args:
            X: Feature matrix
            y: Labels
            neuron_indices: List of neuron indices
            output_path: Path to save dataset
            metadata: Additional metadata to save
            format: Output format ("npz" or "pickle")

        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "npz":
            # Convert neuron_indices to numpy array of strings
            indices_arr = np.array(neuron_indices, dtype=object)

            # Create a dictionary with all data to save
            data_dict = {"X": X, "y": y, "neuron_indices": indices_arr}

            # Add metadata if provided
            if metadata:
                for k, v in metadata.items():
                    # Convert numpy values to native Python types
                    if isinstance(v, (np.number, np.ndarray)):
                        v = v.item() if hasattr(v, "item") else v.tolist()
                    data_dict[k] = v

            np.savez(output_path, **data_dict)

        elif format == "pickle":
            data = {"X": X, "y": y, "neuron_indices": neuron_indices, "metadata": metadata}

            with open(output_path, "wb") as f:
                pickle.dump(data, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
