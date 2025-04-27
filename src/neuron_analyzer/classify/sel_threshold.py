import json
import logging
import typing as t
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from neuron_analyzer.analysis.geometry_util import NeuronGroupAnalyzer
from neuron_analyzer.load_util import JsonProcessor

T = t.TypeVar("T")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class NeuronData:
    """Data class to store neuron features and labels."""

    features: np.ndarray
    delta_loss: float
    label: int | None = None


class NeuronFeatureExtractor:
    """Class for loading and extracting neuron features from raw data."""

    def __init__(
        self,
        args: t.Any,
        abl_path: Path,
        step_path: Path,
        out_dir: Path,
        step_num: str,
        device: str,
        resume: bool = False,
    ):
        """Initialize the NeuronFeatureExtractor."""
        self.args = args
        self.abl_path = abl_path
        self.step_path = step_path
        self.device = device
        self.resume = resume
        self.step_num = step_num
        # configure out path dir
        self.out_dir = out_dir / self.step_num / str(self.args.data_range_end)

    def load_data(self) -> pd.DataFrame:
        """Load and filter feather data."""
        out_path = self.out_dir / f"k{self.args.k}.feather"
        if out_path.is_file() and self.resume:
            self.activation_data = pd.read_feather(self.out_path)
            return self.activation_data

        feather_path = self.abl_path / self.step_num / str(self.args.data_range_end) / f"k{self.args.k}.feather"
        if feather_path.is_file():
            group_analyzer = NeuronGroupAnalyzer(
                args=self.args,
                feather_path=feather_path,
                step_path=self.step_path,
                abl_path=self.abl_path,
                device=self.device,
            )
            activation_data = group_analyzer.load_activation_df()

        # Save the selected intermediate data
        activation_data.to_feather(out_path)
        return activation_data

    def build_vector(self, df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Load and process data from the filtered dataframe."""
        # Group by neuron index (component_name)
        neuron_indices = df["component_name"].unique()
        # Dictionary to store feature vector for each neuron
        neuron_features: dict[str, np.ndarray] = {}
        # Dictionary to store delta losses for each neuron
        delta_losses: dict[str, float] = {}
        for neuron_idx in neuron_indices:
            # Get all rows for this neuron
            neuron_data = df[df["component_name"] == neuron_idx]
            # Extract activation values as features
            activations = neuron_data["activation"].values
            # Store in dictionaries
            neuron_features[neuron_idx] = activations
            delta_losses[neuron_idx] = float(neuron_data["delta_loss"].values.mean())
        return neuron_features, delta_losses

    def run_pipeline(self) -> dict:
        """Extract and save features and delta losses for a single step."""
        # resume logic
        out_path = self.out_dir / "features.json"
        if self.resume and out_path.is_file():
            # load file for the optimal selection
            logger.info(f"Load existing file from {out_path}")
            return JsonProcessor.load_json(out_path)

        # Load data if not already loaded
        activation_data = self.load_data()
        # Calculate features and losses if not provided
        neuron_features, delta_losses = self.build_vector(activation_data)
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
        # Save as JSON
        JsonProcessor.save_json(results, out_path)
        logger.info(f"Save file to {out_path}")
        return results


class SingleThresholdCalculator:
    """Class for calculating thresholds from data."""

    def __init__(
        self,
        data: dict = None,
        out_dir: Path = None,
        min_percentile: int = 5,
        max_percentile: int = 95,
    ):
        """Initialize the SingleThresholdCalculator."""
        self.data, self.losses, self.abs_losses = self.load_data(data, out_dir)
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile

    def load_data(self, data: dict, out_dir: Path) -> tuple[float, float]:
        """Load (absolute) delta loss."""
        if not data:
            data = JsonProcessor.load_json(out_dir / "features.json")
        delta_losses = data["delta_loss"]
        losses = np.array(list(delta_losses.values()))
        return data, losses, np.abs(losses)

    def calculate_thresholds(self) -> dict[int, float]:
        """Calculate threshold values at different percentiles."""
        # Calculate percentiles
        percentiles = list(range(self.min_percentile, 50)) + list(range(50, self.max_percentile + 1))
        # Compute thresholds at each percentile
        threshold_values: dict[int, float] = {}
        for p in percentiles:
            threshold = float(np.percentile(self.abs_losses, p))
            threshold_values[p] = threshold
        return threshold_values

    def analyze_class_distribution(self, threshold: float) -> dict[str, t.Any]:
        """Analyze class distribution for a given threshold."""
        # Convert delta losses to numpy array
        # Calculate class sizes with this threshold
        common_mask = self.abs_losses < threshold
        boost_mask = self.losses > threshold
        suppress_mask = self.losses < -threshold

        common_count = int(np.sum(common_mask))
        boost_count = int(np.sum(boost_mask))
        suppress_count = int(np.sum(suppress_mask))
        special_count = boost_count + suppress_count

        # Calculate percentages
        total_count = len(self.losses)
        common_percent = common_count / total_count * 100
        boost_percent = boost_count / total_count * 100
        suppress_percent = suppress_count / total_count * 100
        special_percent = special_count / total_count * 100

        # Return distribution information
        return {
            "common_count": common_count,
            "boost_count": boost_count,
            "suppress_count": suppress_count,
            "special_count": special_count,
            "common_percent": common_percent,
            "boost_percent": boost_percent,
            "suppress_percent": suppress_percent,
            "special_percent": special_percent,
            "min_binary_percent": min(common_percent, special_percent),
            "min_triclass_percent": min(common_percent, boost_percent, suppress_percent),
            "unified_min_percent": min(common_percent, boost_percent, suppress_percent),
        }

    def calculate_stat(self) -> dict[str, float]:
        """Calculate basic statistics for a step's delta losses."""
        # Calculate basic statistics
        return {
            "mean": float(np.mean(self.losses)),
            "std": float(np.std(self.losses)),
            "abs_mean": float(np.mean(self.abs_losses)),
            "abs_std": float(np.std(self.abs_losses)),
            "min": float(np.min(self.losses)),
            "max": float(np.max(self.losses)),
            "count": int(len(self.losses)),
        }

    def run_pipeline(self) -> dict[str, t.Any]:
        """Analyze all thresholds for a single step."""
        # Extract step number and delta losses
        out_path = self.out_dir / "threshold_stat.json"
        step_num = self.data["step_num"]

        if self.resume and out_path.is_file():
            # load file for the optimal selection
            logger.info(f"Load existing file from {out_path}")
            return JsonProcessor.load_json(out_path)

        # Calculate basic statistics
        statistics = self.calculate_stat()

        # Calculate thresholds at different percentiles
        threshold_values = self.calculate_thresholds()

        # Analyze class distribution for each threshold
        class_sizes = {}
        for p, threshold in threshold_values.items():
            class_sizes[p] = self.analyze_class_distribution(threshold)

        # Find the percentile with the most balanced classes (highest min percentage)
        best_percentile = None
        best_balance = 0.0

        for p, distribution in class_sizes.items():
            min_percent = distribution["unified_min_percent"]
            if best_percentile is None or min_percent > best_balance:
                best_percentile = p
                best_balance = min_percent

        # save the full analysis
        threshold_stat = {
            "step_num": step_num,
            "statistics": statistics,
            "threshold_values": threshold_values,
            "class_sizes": class_sizes,
            "optimal_percentile": best_percentile,
            "optimal_threshold": threshold_values[best_percentile] if best_percentile is not None else None,
            "optimal_balance": best_balance,
        }
        # Save as JSON
        JsonProcessor.save_json(threshold_stat, out_path)
        logger.info(f"Save file to {out_path}")
        return threshold_stat


class ThresholdOptimizer:
    """Class for selecting optimal thresholds across multiple training steps."""

    def __init__(
        self,
        stat_results: dict,
        binary_min_percent: float = 10.0,
        triclass_min_percent: float = 5.0,
        converged_step: str = None,
    ):
        """Initialize the ThresholdOptimizer.

        Args:
            stat_results: Dictionary mapping step names to threshold statistics
            binary_min_percent: Minimum percentage required for binary classification
            triclass_min_percent: Minimum percentage required for triclass classification
            converged_step: Name of step considered as converged model (defaults to last step)

        """
        self.stat_results = stat_results
        self.binary_min_percent = binary_min_percent
        self.triclass_min_percent = triclass_min_percent
        self.training_steps = list(stat_results.keys())

        if converged_step is None:
            # Default to the last step (typically the converged model)
            self.converged_step = self.training_steps[-1]
        else:
            self.converged_step = converged_step

    def collect_threshold_candidates(self) -> list:
        """Collect optimal threshold candidates from each step.

        Returns:
            List of candidate thresholds with their metadata

        """
        candidates = []

        # Collect the optimal threshold from each step
        for step in self.training_steps:
            step_data = self.stat_results[step]

            # Get the optimal threshold for this step
            if "optimal_threshold" in step_data:
                optimal_threshold = step_data["optimal_threshold"]
                optimal_percentile = step_data["optimal_percentile"]

                candidates.append(
                    {
                        "step": step,
                        "threshold": optimal_threshold,
                        "percentile": optimal_percentile,
                    }
                )

        return candidates

    def evaluate_candidate(self, threshold: float) -> dict:
        """Evaluate a threshold candidate across all steps.

        Args:
            threshold: Threshold value to evaluate

        Returns:
            Evaluation metrics for this threshold

        """
        binary_mins = []
        triclass_mins = []
        unified_mins = []

        # Check this threshold on each step
        for step in self.training_steps:
            # Need to analyze the distribution for this specific threshold
            # (which might not be at a standard percentile)
            step_data = self.stat_results[step]
            delta_losses = step_data["delta_losses"]

            # Convert to numpy arrays
            losses = np.array(list(delta_losses.values()))
            abs_losses = np.abs(losses)

            # Calculate class sizes
            common_mask = abs_losses < threshold
            boost_mask = losses > threshold
            suppress_mask = losses < -threshold

            common_count = np.sum(common_mask)
            boost_count = np.sum(boost_mask)
            suppress_count = np.sum(suppress_mask)
            special_count = boost_count + suppress_count

            # Calculate percentages
            total_count = len(losses)
            common_percent = common_count / total_count * 100
            boost_percent = boost_count / total_count * 100
            suppress_percent = suppress_count / total_count * 100
            special_percent = special_count / total_count * 100

            # Calculate minimum percentages
            min_binary = min(common_percent, special_percent)
            min_triclass = min(common_percent, boost_percent, suppress_percent)
            min_unified = min(common_percent, boost_percent, suppress_percent)

            binary_mins.append(min_binary)
            triclass_mins.append(min_triclass)
            unified_mins.append(min_unified)

        # Return the minimum percentages across all steps
        return {
            "binary_min": min(binary_mins),
            "triclass_min": min(triclass_mins),
            "unified_min": min(unified_mins),
        }

    def select_optimal_threshold(self) -> dict:
        """Select the optimal threshold across all training steps.

        Returns:
            Dictionary with threshold recommendations and evaluation data

        """
        # Get candidates from each step's optimal threshold
        candidates = self.collect_threshold_candidates()

        # Evaluate each candidate
        evaluations = []
        for candidate in candidates:
            threshold = candidate["threshold"]
            evaluation = self.evaluate_candidate(threshold)

            evaluations.append(
                {
                    "threshold": threshold,
                    "step": candidate["step"],
                    "percentile": candidate["percentile"],
                    "binary_min": evaluation["binary_min"],
                    "triclass_min": evaluation["triclass_min"],
                    "unified_min": evaluation["unified_min"],
                    "meets_binary": evaluation["binary_min"] >= self.binary_min_percent,
                    "meets_triclass": evaluation["triclass_min"] >= self.triclass_min_percent,
                    "meets_unified": evaluation["unified_min"]
                    >= max(self.binary_min_percent, self.triclass_min_percent),
                }
            )

        # Filter candidates that meet requirements
        valid_binary = [e for e in evaluations if e["meets_binary"]]
        valid_triclass = [e for e in evaluations if e["meets_triclass"]]
        valid_unified = [e for e in evaluations if e["meets_unified"]]

        # Sort by step (descending) to prioritize later steps
        sorted_steps = sorted(self.training_steps, reverse=True)
        step_priority = {step: idx for idx, step in enumerate(sorted_steps)}

        # Select optimal thresholds (prioritizing later steps)
        binary_recommendation = None
        if valid_binary:
            # Sort by step priority (higher is better)
            valid_binary.sort(key=lambda x: step_priority[x["step"]], reverse=True)
            binary_recommendation = valid_binary[0]

        triclass_recommendation = None
        if valid_triclass:
            valid_triclass.sort(key=lambda x: step_priority[x["step"]], reverse=True)
            triclass_recommendation = valid_triclass[0]

        unified_recommendation = None
        if valid_unified:
            valid_unified.sort(key=lambda x: step_priority[x["step"]], reverse=True)
            unified_recommendation = valid_unified[0]

        return {
            "binary_recommendation": binary_recommendation,
            "triclass_recommendation": triclass_recommendation,
            "unified_recommendation": unified_recommendation,
            "all_evaluations": evaluations,
            "valid_binary": valid_binary,
            "valid_triclass": valid_triclass,
            "valid_unified": valid_unified,
        }

    def get_neuron_groups(self, step_name: str, threshold_type: str = "unified") -> dict:
        """Get neuron groups for a specific step using the optimal threshold.

        Args:
            step_name: Name of the step to analyze
            threshold_type: Type of threshold to use ('unified', 'binary', or 'triclass')

        Returns:
            Dictionary with neuron groups and threshold information

        """
        # Get optimal thresholds
        recommendations = self.select_optimal_threshold()

        if threshold_type == "unified":
            recommendation = recommendations["unified_recommendation"]
        elif threshold_type == "binary":
            recommendation = recommendations["binary_recommendation"]
        elif threshold_type == "triclass":
            recommendation = recommendations["triclass_recommendation"]
        else:
            raise ValueError(f"Invalid threshold_type: {threshold_type}")

        if recommendation is None:
            raise ValueError(f"No valid {threshold_type} threshold found")

        threshold = recommendation["threshold"]

        # Load data for the specified step
        step_data = self.stat_results[step_name]
        delta_losses = step_data["delta_losses"]

        # Convert to numpy arrays
        neuron_indices = list(delta_losses.keys())
        losses = np.array(list(delta_losses.values()))
        abs_losses = np.abs(losses)

        # Apply threshold
        common_mask = abs_losses < threshold
        boost_mask = losses > threshold
        suppress_mask = losses < -threshold

        # Map indices to neuron IDs
        common_neurons = [neuron_indices[i] for i in np.where(common_mask)[0]]
        boost_neurons = [neuron_indices[i] for i in np.where(boost_mask)[0]]
        suppress_neurons = [neuron_indices[i] for i in np.where(suppress_mask)[0]]
        special_neurons = boost_neurons + suppress_neurons

        return {
            "common": common_neurons,
            "boost": boost_neurons,
            "suppress": suppress_neurons,
            "special": special_neurons,
            "threshold": threshold,
            "source_step": recommendation["step"],
            "threshold_type": threshold_type,
        }

    def save_results(self, output_path: Path) -> None:
        """Save optimization results to a JSON file.

        Args:
            output_path: Path to save the results

        """
        optimization_results = self.select_optimal_threshold()

        # Convert numpy types to native Python types for JSON serialization
        for key in ["binary_recommendation", "triclass_recommendation", "unified_recommendation"]:
            if optimization_results[key] is not None:
                for k, v in optimization_results[key].items():
                    if isinstance(v, (np.integer, np.floating)):
                        optimization_results[key][k] = float(v)

        # Convert all evaluations
        for i, eval_data in enumerate(optimization_results["all_evaluations"]):
            for k, v in eval_data.items():
                if isinstance(v, (np.integer, np.floating)):
                    optimization_results["all_evaluations"][i][k] = float(v)

        # Save as JSON
        with open(output_path, "w") as f:
            json.dump(optimization_results, f, indent=2)
