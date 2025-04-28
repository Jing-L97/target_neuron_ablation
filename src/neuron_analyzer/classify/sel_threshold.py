import logging
import typing as t
from pathlib import Path

import numpy as np
import pandas as pd

from neuron_analyzer.analysis.geometry_util import NeuronGroupAnalyzer
from neuron_analyzer.load_util import JsonProcessor

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
        device: str,
    ):
        """Initialize the NeuronFeatureExtractor."""
        self.args = args
        self.abl_path = abl_path
        self.step_path = step_path
        self.device = device
        self.step_num = step_num
        # configure out path dir
        self.out_dir = out_dir / self.step_num / str(self.args.data_range_end)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load and filter feather data."""
        out_path = self.out_dir / f"k{self.args.k}.feather"
        if out_path.is_file() and self.args.resume:
            self.activation_data = pd.read_feather(out_path)
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
        activation_data.reset_index(drop=True).to_feather(out_path)
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
            delta_losses[neuron_idx] = float(neuron_data["delta_loss_post_ablation"].values.mean())
        return neuron_features, delta_losses

    def run_pipeline(self) -> dict:
        """Extract and save features and delta losses for a single step."""
        # resume logic
        out_path = self.out_dir / "features.json"
        if self.args.resume and out_path.is_file():
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
        JsonProcessor.save_json(results, out_path)
        logger.info(f"Save file to {out_path}")
        return results


#######################################################################################################
# Util functions for Threshold stat analyzer


def analyze_distribution(abs_losses: list[float], losses: list[float], threshold: float) -> dict[str, t.Any]:
    """Analyze class distribution for a given threshold."""
    common_mask = abs_losses < threshold
    boost_mask = losses > threshold
    suppress_mask = losses < -threshold

    common_count = int(np.sum(common_mask))
    boost_count = int(np.sum(boost_mask))
    suppress_count = int(np.sum(suppress_mask))
    special_count = boost_count + suppress_count

    # Calculate percentages
    total_count = len(losses)
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


def extract_loss(data: dict | None, data_path: Path | None) -> tuple[float, float]:
    """Extract delta loss from the data dict."""
    if not data:
        data = JsonProcessor.load_json(data_path)
    delta_losses = data["delta_losses"]
    losses = np.array(list(delta_losses.values()))
    return data, losses, np.abs(losses)


#######################################################################################################
# Threshold stat analyzer for single step


class SingleThresholdCalculator:
    """Class for calculating thresholds from data."""

    def __init__(
        self,
        args: t.Any,
        step_num: str,
        data: dict = None,
        out_dir: Path = None,
        min_percentile: int = 5,
        max_percentile: int = 95,
    ):
        """Initialize the SingleThresholdCalculator."""
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        self.step_num = step_num
        self.args = args
        # configure out path dir
        self.out_dir = out_dir / self.step_num / str(self.args.data_range_end)
        self.data, self.losses, self.abs_losses = self.load_data(data)

    def load_data(self, data: dict) -> tuple[float, float]:
        """Load (absolute) delta loss."""
        return extract_loss(data, data_path=self.out_dir / "features.json")

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
        return analyze_distribution(self.abs_losses, self.losses, threshold)

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
        # step_num = self.data["step_num"]

        if self.args.resume and out_path.is_file():
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
            # "step_num": step_num,
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


#######################################################################################################
# Util functions for assembled results from of each step


class GlobalThresholdOptimizer:
    """Class for optimizing thresholds across all training steps."""

    def __init__(
        self,
        args,
        stat_results: dict,
        out_dir: Path = None,
        save_path: Path | None = None,
        binary_min_percent: float = 5.0,
        triclass_min_percent: float = 10.0,
    ):
        """Initialize the GlobalThresholdOptimizer."""
        self.stat_results = stat_results
        self.binary_min_percent = binary_min_percent
        self.triclass_min_percent = triclass_min_percent
        self.training_steps = list(stat_results.keys())
        self.save_path = save_path
        self.out_dir = out_dir
        self.args = args
        self.sorted_steps = sorted(self.training_steps, reverse=True)

    def collect_threshold_candidates(self) -> list[dict]:
        """Collect threshold candidates from all steps' optimal thresholds."""
        candidates = []

        # Get thresholds from each step's optimal threshold
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

    def select_optimal_threshold(self, candidates) -> dict:
        """Select the optimal threshold across all training steps."""
        # Evaluate each candidate
        evaluations = []
        for candidate in candidates:
            threshold = candidate["threshold"]
            source_step = candidate["step"]
            evaluation = self._evaluate_candidate(threshold, source_step)
            evaluations.append(evaluation)

        # Filter candidates that meet requirements
        valid_binary = [e for e in evaluations if e["meets_binary"]]
        valid_triclass = [e for e in evaluations if e["meets_triclass"]]
        valid_unified = [e for e in evaluations if e["meets_unified"]]

        # Create mapping from step to priority (later steps get higher priority)
        step_priority = {step: idx for idx, step in enumerate(self.sorted_steps)}

        # Select optimal thresholds (prioritizing later steps)
        binary_recommendation = None
        if valid_binary:
            # Sort by step priority (higher is better)
            valid_binary.sort(key=lambda x: step_priority[x["source_step"]], reverse=True)
            binary_recommendation = valid_binary[0]

        triclass_recommendation = None
        if valid_triclass:
            valid_triclass.sort(key=lambda x: step_priority[x["source_step"]], reverse=True)
            triclass_recommendation = valid_triclass[0]

        unified_recommendation = None
        if valid_unified:
            valid_unified.sort(key=lambda x: step_priority[x["source_step"]], reverse=True)
            unified_recommendation = valid_unified[0]

        return {
            "binary_recommendation": binary_recommendation,
            "triclass_recommendation": triclass_recommendation,
            "unified_recommendation": unified_recommendation,
            "all_evaluations": evaluations,
            "valid_binary": valid_binary,
            "valid_triclass": valid_triclass,
            "valid_unified": valid_unified,
            "binary_min_percent": self.binary_min_percent,
            "triclass_min_percent": self.triclass_min_percent,
        }

    def _evaluate_candidate(self, threshold: float, source_step: str) -> dict:
        """Evaluate a threshold candidate across all steps."""
        # Track minimum percentages across all steps
        binary_mins = []
        triclass_mins = []
        unified_mins = []

        # Store detailed distribution for each step
        step_distributions = {}

        # Check this threshold on each step
        for step in self.training_steps:
            # Analyze class distribution for this step using the threshold
            distribution = self._analyze_class_distribution(threshold, step)
            step_distributions[step] = distribution

            # Track minimum percentages
            binary_mins.append(distribution["min_binary_percent"])
            triclass_mins.append(distribution["min_triclass_percent"])
            unified_mins.append(distribution["unified_min_percent"])

        # Find the minimum percentages across all steps
        binary_min = min(binary_mins)
        triclass_min = min(triclass_mins)
        unified_min = min(unified_mins)

        # Find the worst-case steps (steps with minimum percentages)
        worst_binary_step = self.training_steps[binary_mins.index(binary_min)]
        worst_triclass_step = self.training_steps[triclass_mins.index(triclass_min)]
        worst_unified_step = self.training_steps[unified_mins.index(unified_min)]

        return {
            "threshold": threshold,
            "source_step": source_step,
            "binary_min": binary_min,
            "triclass_min": triclass_min,
            "unified_min": unified_min,
            "worst_binary_step": worst_binary_step,
            "worst_triclass_step": worst_triclass_step,
            "worst_unified_step": worst_unified_step,
            "meets_binary": binary_min >= self.binary_min_percent,
            "meets_triclass": triclass_min >= self.triclass_min_percent,
            "meets_unified": unified_min >= max(self.binary_min_percent, self.triclass_min_percent),
            "step_distributions": step_distributions,
        }

    def _analyze_class_distribution(self, threshold: float, step: int) -> dict:
        """Analyze class distribution for a given threshold on a specific step."""
        # load delta loss data
        data_path = self.out_dir / str(step) / str(self.args.data_range_end) / "features.json"
        _, losses, abs_losses = extract_loss(data=None, data_path=data_path)
        return analyze_distribution(abs_losses, losses, threshold)

    def run_pipeline(self) -> dict:
        """Get the simplified final optimization result."""
        # Get candidates from each step's optimal threshold
        candidates = self.collect_threshold_candidates()
        results = self.select_optimal_threshold(candidates)
        JsonProcessor.save_json(results, self.save_path)
        logger.info(f"Saved optimization results to {self.save_path}")
