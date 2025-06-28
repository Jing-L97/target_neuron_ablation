import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


@dataclass
class NeuronGroups:
    """Container for neuron group definitions."""

    boost: list[int]
    suppress: list[int]
    excluded: list[int]
    random_groups: dict[str, list[int]]


class DataManager:
    """Handles neural activation data preprocessing, validation, and matrix creation."""

    def __init__(
        self,
        activation_data: pd.DataFrame,
        activation_column: str = "activation",
        token_column: str = "str_tokens",
        context_column: str = "context",
        component_column: str = "component_name",
        device: str = "auto",
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize DataManager with activation data."""
        self.activation_col = activation_column
        self.token_col = token_column
        self.context_col = context_column
        self.component_col = component_column

        # Setup device
        self.device = self._setup_device(device)
        self.dtype = dtype

        # Validate and prepare data
        self.data = self._validate_and_prepare_data(activation_data)

        # Extract basic info
        self.token_contexts = self.data["token_context_id"].unique()
        self.all_neuron_indices = sorted(self.data[self.component_col].unique())

        logger.info(
            f"DataManager initialized: {len(self.token_contexts)} contexts, "
            f"{len(self.all_neuron_indices)} neurons, device: {self.device}"
        )

    def _setup_device(self, device: str) -> str:
        """Setup computation device with fallback."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, using CPU")
            return "cpu"
        return device

    def _validate_and_prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate input data and create derived columns."""
        if data.empty:
            raise ValueError("Activation data cannot be empty")

        # Check required columns
        required_cols = [self.activation_col, self.token_col, self.context_col, self.component_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Create working copy
        df = data.copy()

        # Validate and clean component indices
        try:
            df[self.component_col] = df[self.component_col].astype(int)
        except (ValueError, TypeError):
            raise ValueError(f"Component column '{self.component_col}' must contain integer indices")

        # Validate activation values
        if not pd.api.types.is_numeric_dtype(df[self.activation_col]):
            raise ValueError(f"Activation column '{self.activation_col}' must contain numeric values")

        # Remove any rows with NaN values in critical columns
        initial_rows = len(df)
        df = df.dropna(subset=required_cols)
        if len(df) < initial_rows:
            logger.warning(f"Dropped {initial_rows - len(df)} rows with missing values")

        if df.empty:
            raise ValueError("No valid data remaining after cleaning")

        # Create token-context identifier
        df["token_context_id"] = df[self.token_col].astype(str) + "_" + df[self.context_col].astype(str)

        logger.info(f"Data validated: {len(df)} rows, {df[self.component_col].nunique()} unique neurons")
        return df

    def create_neuron_groups(
        self,
        boost_indices: list[int],
        suppress_indices: list[int],
        excluded_indices: list[int],
        num_random_groups: int = 2,
        random_seed: int = 42,
    ) -> NeuronGroups:
        """Create and validate neuron groups for analysis."""
        # Validate indices exist in data
        available_indices = set(self.all_neuron_indices)

        for name, indices in [("boost", boost_indices), ("suppress", suppress_indices), ("excluded", excluded_indices)]:
            invalid = set(indices) - available_indices
            if invalid:
                raise ValueError(f"Invalid {name} indices not found in data: {invalid}")

        # Check for overlaps
        boost_set = set(boost_indices)
        suppress_set = set(suppress_indices)
        excluded_set = set(excluded_indices)

        if boost_set & suppress_set:
            raise ValueError("Boost and suppress indices cannot overlap")

        # Generate random groups
        np.random.seed(random_seed)
        random_groups = self._generate_random_groups(
            boost_indices, suppress_indices, excluded_indices, num_random_groups
        )

        return NeuronGroups(
            boost=boost_indices, suppress=suppress_indices, excluded=excluded_indices, random_groups=random_groups
        )

    def _generate_random_groups(
        self, boost_indices: list[int], suppress_indices: list[int], excluded_indices: list[int], num_groups: int
    ) -> dict[str, list[int]]:
        """Generate random neuron groups for control comparisons."""
        # Calculate group size (use larger of boost/suppress groups)
        group_size = max(len(boost_indices), len(suppress_indices))
        if group_size == 0:
            raise ValueError("Cannot generate random groups: no boost or suppress indices provided")

        # Get available indices (excluding special groups)
        special_indices = set(boost_indices + suppress_indices + excluded_indices)
        available_indices = [idx for idx in self.all_neuron_indices if idx not in special_indices]

        if len(available_indices) < group_size * num_groups:
            logger.warning(f"Insufficient neurons for {num_groups} random groups of size {group_size}")
            # Adjust number of groups or group size
            if len(available_indices) >= num_groups:
                group_size = len(available_indices) // num_groups
                logger.info(f"Reduced random group size to {group_size}")
            else:
                num_groups = max(1, len(available_indices) // group_size)
                logger.info(f"Reduced number of random groups to {num_groups}")

        # Generate non-overlapping random groups
        random_groups = {}
        remaining_indices = available_indices.copy()

        for i in range(num_groups):
            if len(remaining_indices) < group_size:
                break

            group_indices = list(np.random.choice(remaining_indices, size=group_size, replace=False))
            random_groups[f"random_{i + 1}"] = group_indices

            # Remove selected indices to ensure no overlap
            remaining_indices = [idx for idx in remaining_indices if idx not in group_indices]

        logger.info(f"Generated {len(random_groups)} random groups")
        return random_groups

    def create_activation_matrix(self, neuron_indices: list[int]) -> np.ndarray:
        """Create activation matrix for specified neurons."""
        if not neuron_indices:
            return np.empty((len(self.token_contexts), 0))

        # Filter data for specified neurons
        neuron_data = self.data[self.data[self.component_col].isin(neuron_indices)]

        if neuron_data.empty:
            logger.warning(f"No data found for neuron indices: {neuron_indices}")
            return np.zeros((len(self.token_contexts), len(neuron_indices)))

        # Create pivot table
        pivot_table = neuron_data.pivot_table(
            index="token_context_id",
            columns=self.component_col,
            values=self.activation_col,
            aggfunc="mean",  # Handle duplicate entries by averaging
        )

        # Ensure all contexts and neurons are represented
        pivot_table = pivot_table.reindex(index=self.token_contexts, columns=neuron_indices, fill_value=0.0)

        # Convert to numpy array
        matrix = pivot_table.values

        # Handle any remaining NaN values
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

        logger.debug(f"Created activation matrix: {matrix.shape} for {len(neuron_indices)} neurons")
        return matrix

    def create_activation_tensors(self, neuron_groups: NeuronGroups) -> dict[str, torch.Tensor]:
        """Create PyTorch tensors for all neuron groups."""
        tensors = {}

        # Create tensors for main groups
        for group_name, indices in [("boost", neuron_groups.boost), ("suppress", neuron_groups.suppress)]:
            if indices:  # Only create if group is not empty
                matrix = self.create_activation_matrix(indices)
                tensor = torch.tensor(matrix, dtype=self.dtype, device=self.device)
                tensors[group_name] = tensor
                logger.debug(f"Created tensor for {group_name}: {tensor.shape}")

        # Create tensors for random groups
        for group_name, indices in neuron_groups.random_groups.items():
            matrix = self.create_activation_matrix(indices)
            tensor = torch.tensor(matrix, dtype=self.dtype, device=self.device)
            tensors[group_name] = tensor
            logger.debug(f"Created tensor for {group_name}: {tensor.shape}")

        return tensors

    def create_rare_token_mask(
        self, rare_token_mask: np.ndarray | None = None, frequency_threshold: float | None = None
    ) -> np.ndarray:
        """Create or validate rare token mask."""
        if rare_token_mask is not None:
            # Validate provided mask
            if len(rare_token_mask) != len(self.token_contexts):
                raise ValueError(
                    f"Rare token mask length ({len(rare_token_mask)}) != "
                    f"number of contexts ({len(self.token_contexts)})"
                )

            mask = np.asarray(rare_token_mask, dtype=bool)
            logger.info(f"Using provided rare token mask: {np.sum(mask)} rare contexts")
            return mask

        # Create frequency-based mask
        if frequency_threshold is None:
            frequency_threshold = 25.0  # Bottom 25th percentile

        # Count token frequencies
        token_counts = self.data.groupby(self.token_col).size()

        # Determine threshold
        threshold = np.percentile(token_counts.values, frequency_threshold)
        rare_tokens = set(token_counts[token_counts <= threshold].index)

        # Extract tokens from context IDs and create mask
        context_tokens = [ctx.split("_")[0] for ctx in self.token_contexts]
        mask = np.array([token in rare_tokens for token in context_tokens])

        logger.info(
            f"Created frequency-based rare token mask: {np.sum(mask)} rare contexts "
            f"({frequency_threshold}th percentile, threshold: {threshold:.1f})"
        )

        return mask

    def get_context_subset(self, context_mask: np.ndarray) -> list[str]:
        """Get subset of contexts based on boolean mask."""
        if len(context_mask) != len(self.token_contexts):
            raise ValueError("Context mask length mismatch")
        return self.token_contexts[context_mask].tolist()

    def get_data_summary(self) -> dict[str, Any]:
        """Get summary statistics of the managed data."""
        return {
            "n_contexts": len(self.token_contexts),
            "n_neurons": len(self.all_neuron_indices),
            "n_data_points": len(self.data),
            "neuron_range": (min(self.all_neuron_indices), max(self.all_neuron_indices)),
            "activation_stats": {
                "mean": float(self.data[self.activation_col].mean()),
                "std": float(self.data[self.activation_col].std()),
                "min": float(self.data[self.activation_col].min()),
                "max": float(self.data[self.activation_col].max()),
            },
            "device": self.device,
            "dtype": str(self.dtype),
        }

    def cleanup_tensors(self, tensors: dict[str, torch.Tensor]) -> None:
        """Clean up GPU memory from tensors."""
        if self.device == "cuda":
            for tensor in tensors.values():
                del tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.debug("Cleaned up GPU memory")
