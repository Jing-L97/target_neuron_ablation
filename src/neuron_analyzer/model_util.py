#!/usr/bin/env python
import ast
import logging
import random
import typing as t
from pathlib import Path

import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
random.seed(42)
T = t.TypeVar("T")


#######################################################
# Util func to set up model steps


class StepConfig:
    """Configuration for step-wise analysis of model checkpoints."""

    def __init__(
        self, resume: bool = False, debug: bool = False, file_path: Path | None = None, interval: int = 1
    ) -> None:
        """Initialize step configuration."""
        # Generate the complete list of steps
        self.steps = self.generate_pythia_checkpoints()

        # Apply interval sampling while preserving start and end steps
        if interval > 1:
            self.steps = self._apply_interval_sampling(self.steps, interval)
            logger.info(f"Applied interval sampling with n={interval}, resulting in {len(self.steps)} steps")

        if debug:
            self.steps = self.steps[:5]
            logger.info("Entering debugging mode, select first 5 steps.")

        # If resuming, filter out already processed steps
        if resume and file_path is not None:
            self.steps = self.recover_steps(file_path)

        logger.info(f"Generated {len(self.steps)} checkpoint steps")

    def _apply_interval_sampling(self, steps: list[int], interval: int) -> list[int]:
        """Sample steps at every nth interval while preserving start and end steps."""
        if len(steps) <= 2:
            return steps

        # Always keep first and last steps
        first_step = steps[0]
        last_step = steps[-1]

        # Sample the middle steps at the specified interval
        middle_steps = steps[1:-1][::interval]

        # Combine and return
        result = [first_step] + middle_steps + [last_step]

        # Ensure no duplicates if interval sampling results in last step being included twice
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

        return [step for step in self.steps if step not in completed_steps]


#######################################################
# Util func to load model neurons


class NeuronLoader:
    """Class for loading and processing neuron data from CSV files."""

    def __init__(self, min_val: int = 1, max_val: int = 2047):
        """Initialize the NeuronLoader with range parameters for random neuron generation."""
        self.min_val = min_val
        self.max_val = max_val

    def load_neuron_dict(
        self,
        file_path: Path,
        key_col: str = "step",
        value_col: str = "top_neurons",
        random_base: bool = False,
        top_n: int = 0,
    ) -> tuple[dict[int, list[int]], int]:
        """Load a DataFrame and convert neuron values to integers."""
        df = pd.read_csv(file_path)
        result = {}
        layer_num = 0

        for _, row in df.iterrows():
            try:
                # Parse the string to list of floats
                float_neurons = ast.literal_eval(row[value_col])

                # Extract the decimal part as integer; Converts '5.2021' format to 2021.
                neurons = [int(str(float(x)).split(".")[1]) for x in float_neurons]

                # Generate random neurons if requested
                if random_base:
                    neurons = self._generate_neurons(neurons)

                # Limit to top_n if requested
                if top_n != 0:
                    neurons = neurons[:top_n]

                result[row[key_col]] = neurons

                # Get layer number from the first neuron (assuming all neurons are from the same layer)
                layer_num = int(str(float(float_neurons[0])).split(".")[0])

            except (ValueError, SyntaxError) as e:
                print(f"Error parsing neuron list for step {row[key_col]}: {e}")
                result[row[key_col]] = []

        logger.info(f"Computing on {top_n} neurons")
        return result, layer_num

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
