#!/usr/bin/env python
import ast
import logging
import random
from pathlib import Path
from typing import Any

import neel.utils as nutils
import pandas as pd
from datasets import load_dataset
from transformer_lens import utils

from neuron_analyzer import settings
from neuron_analyzer.ablation.abl_util import load_model_from_tl_name

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
random.seed(42)


#######################################################
# Util func to set up model steps


class StepConfig:
    """Configuration for step-wise analysis of model checkpoints."""

    def __init__(
        self,
        resume: bool = False,
        debug: bool = False,
        file_path: Path | None = None,
        interval: int = 1,
        start_idx: int = 0,
        end_idx: int | None = None,
    ) -> None:
        """Initialize step configuration."""
        # Generate the complete list of steps
        self.steps = self.generate_pythia_checkpoints()

        # Apply interval sampling while preserving start and end steps
        if interval > 1 or start_idx > 0 or end_idx is not None:
            self.steps = self._apply_interval_sampling(self.steps, interval, start_idx, end_idx)
            range_info = f"from index {start_idx}"
            if end_idx is not None:
                range_info += f" to {end_idx}"
            logger.info(
                f"Applied interval sampling with n={interval} {range_info}, resulting in {len(self.steps)} steps"
            )
        if debug:
            self.steps = self.steps[:5]
            logger.info("Entering debugging mode, select first 5 steps.")
        # If resuming, filter out already processed steps
        if resume and file_path is not None:
            self.steps = self.recover_steps(file_path)
        logger.info(f"Generated {len(self.steps)} checkpoint steps")

    def _apply_interval_sampling(
        self, steps: list[int], interval: int = 1, start_idx: int = 0, end_idx: int | None = None
    ) -> list[int]:
        """Sample steps at every nth interval within specified range while preserving start and end steps."""
        if len(steps) <= 2:
            return steps

        # Validate indices
        start_idx = max(0, min(start_idx, len(steps) - 2))  # Ensure it's between 0 and len(steps)-2

        if end_idx is None:
            end_idx = len(steps) - 1
        else:
            end_idx = max(
                start_idx + 1, min(end_idx, len(steps) - 1)
            )  # Ensure it's between start_idx+1 and len(steps)-1

        # Always keep first and last steps of the original list
        first_step = steps[0]
        last_step = steps[-1]

        # Determine the range for sampling
        if start_idx == 0 and end_idx == len(steps) - 1:
            # Full range case - sample all middle elements
            middle_steps = steps[1:-1][::interval]
        else:
            # Create a slice for the range to sample
            range_to_sample = steps[start_idx : end_idx + 1]

            # If start_idx is 0, we need to exclude the first element to avoid duplication
            if start_idx == 0:
                range_to_sample = range_to_sample[1:]

            # If end_idx is the last index, we need to exclude the last element to avoid duplication
            if end_idx == len(steps) - 1:
                range_to_sample = range_to_sample[:-1]

            # Apply interval sampling to the specified range
            middle_steps = range_to_sample[::interval]

        # Combine all parts
        result = []
        # Always include the first step
        result.append(first_step)
        # Include sampled middle steps
        result.extend(middle_steps)
        # Always include the last step
        result.append(last_step)
        # Ensure no duplicates
        return sorted(list(set(result)))

    def generate_pythia_checkpoints(self) -> list[int]:
        """Generate complete list of Pythia checkpoint steps."""
        # Initial checkpoint
        checkpoints = [0]

        # Log-spaced checkpoints (2^0 to 2^9)
        log_spaced = [2**i for i in range(10)]  # 1, 2, 4, ..., 512

        # Evenly-spaced checkpoints from 1000 to 143000
        step_size = (143000 - 1000) // 142  # Calculate step size for even spacing
        linear_spaced = list(range(1000, 143001, step_size))

        # Combine all checkpoints
        checkpoints.extend(log_spaced)
        checkpoints.extend(linear_spaced)

        # Remove duplicates and sort
        return sorted(list(set(checkpoints)))

    def recover_steps(self, file_path: Path) -> list[int]:
        """Filter out steps that have already been processed based on column names."""
        if not file_path.is_file():
            return self.steps

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Extract completed steps from column headers (only consider fully numeric columns)
        completed_steps = set()
        for col in df.columns:
            if col.isdigit():
                completed_steps.add(int(col))

        # Return only steps that haven't been completed yet
        return [step for step in self.steps if step not in completed_steps]


#######################################################
# Util func to load model neurons


class NeuronLoader:
    """Class for loading and processing neuron data from CSV files."""

    def __init__(self, file_path=None, top_n: int = 0, min_val: int = 1, max_val: int = 2047):
        """Initialize the NeuronLoader with range parameters for random neuron generation."""
        self.min_val = min_val
        self.max_val = max_val
        self.top_n = top_n
        self.df = pd.read_csv(file_path) if isinstance(file_path, Path) else file_path

    def load_neuron_dict(
        self,
        key_col: str = "step",
        value_col: str = "top_neurons",
        random_base: bool = False,
    ) -> tuple[dict[int, list[int]], int]:
        """Load a DataFrame and convert neuron values to integers."""
        result = {}
        layer_num = 0
        for _, row in self.df.iterrows():
            neuron_value = row[value_col]
            neurons, layer_num = self.extract_neurons(neuron_value)
            # Generate random neurons if requested
            if random_base:
                neurons = self._generate_neurons(neurons)
            result[row[key_col]] = neurons
        logger.info(f"Successfully parsed {self.top_n} neurons from layer {layer_num}")
        return result, layer_num

    def load_neuron_stat(
        self,
        key_col: str = "step",
        value_col: str = "top_neurons",
        col_dict={"prob": "delta_loss_post"},
    ) -> tuple[dict[int, list[int]], int]:
        """Load a DataFrame and convert neuron stats to floats."""
        result = {}
        for _, row in self.df.iterrows():
            neuron_stat = row[col_dict[value_col]]
            stats = self.extract_stats(neuron_stat)
            result[row[key_col]] = stats
        logger.info(f"Successfully processed stat info from {self.top_n} neurons.")
        return result

    def extract_stats(self, neuron_stat) -> list[int]:
        """Extract neuron indices from the inputs strings.."""
        float_stats = self._convert_to_list(neuron_stat)
        # Extract the decimal part as integer; Converts '5.2021' format to 2021.
        stats = [float(stat) for stat in float_stats]
        # Limit to top_n if requested
        if self.top_n > 0 and len(stats) > self.top_n:
            stats = stats[: self.top_n]
        return stats

    def extract_neurons(self, neuron_value) -> list[int]:
        """Extract neuron indices from the inputs strings.."""
        float_neurons = self._convert_to_list(neuron_value)
        # Extract the decimal part as integer; Converts '5.2021' format to 2021.
        neurons = [self.parse_neurons(str(neuron)) for neuron in float_neurons]
        # Limit to top_n if requested
        # if self.top_n > 0 and len(neurons) > self.top_n:
        if self.top_n < 0:
            raise ValueError("The neuron number must be non-negative")
        neurons = neurons[: self.top_n]
        # Get layer number from the first neuron (assuming all neurons are from the same layer)
        layer_num = int(str(float(float_neurons[0])).split(".")[0])
        return neurons, layer_num

    def _convert_to_list(self, input_val: str | list) -> list[int | float]:
        """Convert the input into the a list."""
        return ast.literal_eval(input_val) if not isinstance(input_val, list) else input_val

    def parse_neurons(self, neuron: str) -> int:
        """Parse neuron index."""
        return int(str(float(neuron)).split(".")[1])

    def generate_neurons(self, exclude_list: list[int]) -> list[int]:
        """Generate a list of non-repeating random neuron indices with the same size as the input list."""
        # Convert to set for faster lookups
        excluded_ints = set(exclude_list)
        # Calculate how many numbers we need
        count_needed = len(exclude_list)
        # Ensure the range is large enough to generate required unique numbers
        available_range = self.max_val - self.min_val + 1 - len(excluded_ints)
        max_val = self.max_val

        if available_range < count_needed:
            max_val = self.min_val + count_needed + len(excluded_ints) - 1
        # Generate the list of non-repeating random integers
        result = []
        while len(result) < count_needed:
            rand_int = random.randint(self.min_val, max_val)
            if rand_int not in excluded_ints and rand_int not in result:
                result.append(rand_int)

        return result


#######################################################
# Util func to load tokenizer and model


class ModelHandler:
    def load_model_and_tokenizer(
        self, step: int, model_name: str, hf_token_path: str | Path, device: str
    ) -> tuple[Any, Any]:
        """Load model and tokenizer for processing."""
        # Load HF token
        with open(settings.PATH.unigram_dir / hf_token_path) as f:
            hf_token = f.read()

        model, self.tokenizer = load_model_from_tl_name(
            model_name, device, step=step, cache_dir=settings.PATH.model_dir, hf_token=hf_token
        )
        self.model = model.to(device)
        self.model.eval()
        return self.model, self.tokenizer

    def tokenize_data(self, dataset: str, data_range_start: int, data_range_end: int, seed: int, get_df=False) -> Any:
        """Load and tokenize data."""
        data = load_dataset(dataset, split="train")
        data_slice = data.select(list(range(data_range_start, data_range_end)))
        tokenized_data = utils.tokenize_and_concatenate(data_slice, self.tokenizer, max_length=256, column_name="text")
        tokenized_data = tokenized_data.shuffle(seed)
        if get_df:
            return tokenized_data, nutils.make_token_df(tokenized_data["tokens"], model=self.model)
        return tokenized_data
