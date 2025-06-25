#!/usr/bin/env python
import gc
import json
import logging
import random
import sys
import typing as t
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neuron_analyzer import settings
from neuron_analyzer.ablation.abl_util import get_pile_unigram_distribution

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
random.seed(42)
T = t.TypeVar("T")

#######################################################
# json file tools


class JsonProcessor:
    """Class for handling JSON serialization with NumPy type conversion."""

    @staticmethod
    def convert_numpy_types(obj: t.Any, _seen: set[int] | None = None) -> t.Any:
        """Recursively convert NumPy types and custom objects in a nested structure to standard Python types."""
        # Initialize seen objects set for cycle detection
        if _seen is None:
            _seen = set()

        # Check for cycles using object ID
        obj_id = id(obj)
        if obj_id in _seen:
            # Return a safe representation for circular references
            return f"<circular_reference_to_{type(obj).__name__}>"

        # Handle None
        if obj is None:
            return None

        # Early detection of NetworkX objects to prevent any issues
        if hasattr(obj, "__module__") and obj.__module__ and "networkx" in obj.__module__:
            class_name = obj.__class__.__name__
            if "Graph" in class_name:
                return {
                    "type": "networkx_graph",
                    "graph_type": class_name,
                    "nodes": list(obj.nodes()) if hasattr(obj, "nodes") else [],
                    "edges": list(obj.edges()) if hasattr(obj, "edges") else [],
                    "number_of_nodes": obj.number_of_nodes() if hasattr(obj, "number_of_nodes") else 0,
                    "number_of_edges": obj.number_of_edges() if hasattr(obj, "number_of_edges") else 0,
                }
            return f"<networkx_{class_name}>"

        # Handle SearchResult objects
        if hasattr(obj, "__class__") and obj.__class__.__name__ == "SearchResult":
            # Add to seen set before processing
            _seen.add(obj_id)
            try:
                # Convert SearchResult to dict
                result_dict = {
                    "neurons": JsonProcessor.convert_numpy_types(obj.neurons, _seen),
                    "delta_loss": JsonProcessor.convert_numpy_types(obj.delta_loss, _seen),
                }
                if hasattr(obj, "is_target_size"):
                    result_dict["is_target_size"] = obj.is_target_size
                return result_dict
            finally:
                _seen.discard(obj_id)

        # Handle NumPy arrays
        if isinstance(obj, np.ndarray):
            return JsonProcessor.convert_numpy_types(obj.tolist(), _seen)

        # Handle NumPy scalars
        if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(
            obj, (np.integer, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)
        ):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.complex128):
            return complex(obj)

        # Handle Path objects
        if isinstance(obj, Path):
            return str(obj)

        # Handle NetworkX objects specifically
        if hasattr(obj, "__class__"):
            class_name = obj.__class__.__name__
            # Handle NetworkX graph objects
            if class_name in ["Graph", "DiGraph", "MultiGraph", "MultiDiGraph"]:
                return {
                    "type": "networkx_graph",
                    "graph_type": class_name,
                    "nodes": list(obj.nodes()),
                    "edges": list(obj.edges(data=True)) if hasattr(obj, "edges") else [],
                    "number_of_nodes": obj.number_of_nodes() if hasattr(obj, "number_of_nodes") else 0,
                    "number_of_edges": obj.number_of_edges() if hasattr(obj, "number_of_edges") else 0,
                }
            # Handle NetworkX views (AdjacencyView, etc.)
            if class_name in ["AdjacencyView", "AtlasView", "NodeView", "EdgeView", "DegreeView"]:
                try:
                    return list(obj) if hasattr(obj, "__iter__") else str(obj)
                except:
                    return f"<{class_name}_object>"

        # Add to seen set before processing complex objects
        _seen.add(obj_id)

        try:
            # Handle dictionaries
            if isinstance(obj, dict):
                return {
                    JsonProcessor.convert_numpy_types(k, _seen): JsonProcessor.convert_numpy_types(v, _seen)
                    for k, v in obj.items()
                }

            # Handle lists and tuples
            if isinstance(obj, (list, tuple)):
                converted = [JsonProcessor.convert_numpy_types(item, _seen) for item in obj]
                return converted if isinstance(obj, list) else tuple(converted)

            # Handle sets by converting to list
            if isinstance(obj, set):
                return [JsonProcessor.convert_numpy_types(item, _seen) for item in obj]

            # Handle objects with a to_dict method
            if hasattr(obj, "to_dict") and callable(obj.to_dict):
                return JsonProcessor.convert_numpy_types(obj.to_dict(), _seen)

            # Handle objects with __dict__ attribute
            if hasattr(obj, "__dict__") and not isinstance(obj, type):
                # Special handling for known problematic classes
                class_name = obj.__class__.__name__

                # Skip NetworkX objects entirely (already handled above)
                if any(nx_type in class_name for nx_type in ["Graph", "View", "Atlas", "Node", "Edge", "Degree"]):
                    return f"<{class_name}_skipped>"

                # Handle statistical validation objects carefully
                if class_name in ["ValidationResult", "StatisticalTest", "StatisticalValidator", "BootstrapEstimator"]:
                    # For these classes, extract only essential attributes
                    safe_dict = {}
                    for key, value in obj.__dict__.items():
                        if not key.startswith("_") and key not in [
                            "graph",
                            "data",
                            "samples",
                            "_seen",
                        ]:  # Skip problematic attributes
                            try:
                                safe_dict[key] = JsonProcessor.convert_numpy_types(value, _seen)
                            except (RecursionError, TypeError, AttributeError, ValueError):
                                # Fallback for problematic attributes
                                if isinstance(value, (int, float, str, bool, type(None))):
                                    safe_dict[key] = value
                                else:
                                    safe_dict[key] = str(type(value).__name__)
                    return safe_dict
                return JsonProcessor.convert_numpy_types(obj.__dict__, _seen)

            # Return the object as is if no conversion is needed
            return obj

        finally:
            # Remove from seen set when done processing this object
            _seen.discard(obj_id)

    @classmethod
    def save_json(cls, data: dict, filepath: Path) -> None:
        """Save a nested dictionary with float values to a file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        converted_data = cls.convert_numpy_types(data)
        with open(filepath, "w") as f:
            json.dump(converted_data, f, indent=2)

    @staticmethod
    def load_json(filepath: Path) -> dict:
        """Load a JSON file into a dictionary."""
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)


#######################################################
# Stepwise path managing class


class StepPathProcessor:
    """Process paths and manage steps for resumable processing."""

    def __init__(self, abl_path: Path):
        """Initialize the PathProcessor with the base path."""
        self.abl_path = abl_path
        self.step_dirs: list[tuple[Path, int]] = []

    def sort_paths(self) -> list[tuple[Path, int]]:
        """Get the sorted directory by steps in descending order."""
        step_dirs = []
        for step in self.abl_path.iterdir():
            if step.is_dir():
                # Extract the number from directory name
                try:
                    step_num = int(step.name)
                    step_dirs.append((step, step_num))
                except:
                    logger.info(f"Something wrong with step {step}")

        # Sort directories by step number in descending order
        step_dirs.sort(key=lambda x: x[1], reverse=True)
        self.step_dirs = step_dirs
        return self.step_dirs

    def resume_results(
        self, resume: bool, save_path: Path, file_path: Path = None
    ) -> tuple[dict, list[tuple[Path, int]]]:
        """Resume results from the existing directory list."""
        if not self.step_dirs:
            self.sort_paths()

        if resume and save_path.is_file():
            final_results, remaining_step_dirs = self._get_step_intersection(save_path, self.step_dirs)
            # check whether the target file path exsits
            if file_path and file_path.is_file():
                logger.info(
                    f"Filter steps from existing neuron index file. Steps before filtering: {len(remaining_step_dirs)}"
                )
                _, remaining_step_dirs = self._get_step_intersection(file_path, remaining_step_dirs)
                logger.info(f"Steps after filtering: {len(remaining_step_dirs)}")
            logger.info(f"Resume {len(self.step_dirs) - len(remaining_step_dirs)} states from {save_path}.")
            if len(remaining_step_dirs) == 0:
                logger.info("All steps already processed. Exiting.")
                sys.exit(0)
            return final_results, remaining_step_dirs
        return {}, self.step_dirs

    def _get_step_intersection(
        self, file_path: Path, remaining_step_dirs: list[tuple[Path, int]]
    ) -> list[tuple[Path, int]]:
        """Resume results from the selected indices."""
        # Load JSON file
        final_results = JsonProcessor.load_json(file_path)
        # Get the generated checkpoints
        completed_results = list(final_results.keys())
        # Remove the existing files
        remaining_step_dirs = [p for p in self.step_dirs if p[0].name not in completed_results]
        return final_results, remaining_step_dirs


#######################################################
# load and select eval set


def load_eval(word_path: Path, min_context: int = 2, debug: bool = False) -> tuple[list[str], list[list[str]]]:
    """Load word and context lists from a JSON file."""
    data = JsonProcessor.load_json(word_path)
    # Extract words
    target_words = list(data.keys())
    words = []
    contexts = []
    for word in target_words:
        if len(word) > 1:
            word_contexts = []
            # Handle different JSON structures
            word_data = data[word]
            if len(word_data) >= min_context:
                words.append(word)
                for context_data in word_data:
                    word_contexts.append(context_data["context"])
                contexts.append(word_contexts)
    if debug:
        target_words, contexts = target_words[:5], contexts[:5]
        logger.info("Entering debugging mode. Loading first 5 words")
    if not debug:
        # filter out the single chracters
        logger.info(f"{len(target_words) - len(words)} words are filtered.")
        logger.info(f"Loading {len(words)} words.")

    return words, contexts


def load_df(file_path: Path, col_header: str) -> pd.DataFrame:
    """Load df from the given column."""
    df = pd.read_csv(file_path)
    start_idx = df.columns.tolist().index(col_header)
    return df[start_idx:]


def sel_eval(results_df: pd.DataFrame, eval_path: Path, result_dir: Path, filename):
    """Select the word subset from the eval file."""
    # load eval file
    eval_file = settings.PATH.dataset_root / eval_path
    result_file = result_dir / eval_file.stem / filename
    result_file.parent.mkdir(parents=True, exist_ok=True)
    eval_frame = pd.read_csv(eval_file)
    # select target words
    results_df_sel = results_df[results_df["target_word"].isin(eval_frame["word"])]
    results_df_sel.to_csv(result_file, index=False)
    logger.info(f"Eval set saved to: {result_file}")


#######################################################
# load freq-related file


def load_unigram(model_name, device) -> torch.Tensor:
    """Load unigram distribution based on model type."""
    # Load unigram distribution
    if "pythia" in model_name:
        file_path = settings.PATH.unigram_dir / "pythia-unigrams.npy"
        unigram_distrib, unigram_count = get_pile_unigram_distribution(
            device=device, pad_to_match_W_U=True, file_path=file_path
        )
        logger.info(f"Loaded unigram freq from {file_path}")
    elif "gpt" in model_name:
        file_path = settings.PATH.unigram_dir / "gpt2-small-unigrams_openwebtext-2M_rows_500000.npy"
        unigram_distrib, unigram_count = get_pile_unigram_distribution(
            device=device, pad_to_match_W_U=False, file_path=file_path
        )
        logger.info(f"Loading unigram freq from {file_path}")
    else:
        raise Exception(f"No unigram distribution for {model_name}")

    return unigram_distrib, unigram_count


def load_tail_threshold_stat(longtail_path: Path) -> tuple[float | None, dict | None]:
    """Load longtail threshold from the jason file."""
    data = JsonProcessor.load_json(longtail_path)
    return data["threshold_info"]["probability"]


#######################################################
# Memory management


def cleanup() -> None:
    """Release memory after results are no longer needed."""
    # Force cleanup of any tensor references
    gc.collect()
    # Clear CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
