import numpy as np


def compute_gini(array: np.ndarray) -> float:
    """Compute the Gini coefficient for a 1D numpy array."""
    array = np.array(array).flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)  # Make all values non-negative
    array = array + 1e-8  # Avoid division by zero
    array = np.sort(array)
    n = len(array)
    index = np.arange(1, n + 1)
    return ((2 * np.sum(index * array)) / (n * np.sum(array))) - (n + 1) / n
