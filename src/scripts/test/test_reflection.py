#!/usr/bin/env python
import pickle
from pathlib import Path

import numpy as np
import torch

from neuron_analyzer.classify.reflection import OptimizedSVMHyperplaneReflection

#######################################################################################################
# Fucntions applied in the main scripts
#######################################################################################################


def test_hyperplane_reflection():
    """Test the hyperplane reflection implementation."""
    # Create a simple hyperplane (x + y = 0)
    normal_vector = np.array([1.0, 1.0])
    intercept = 0.0

    # Create mock SVM model
    mock_svm = {"w": normal_vector, "b": intercept}

    # Save mock model
    test_path = Path("test_svm.blob")
    with open(test_path, "wb") as f:
        pickle.dump(mock_svm, f)

    # Create mock torch model and weights
    class MockModel(torch.nn.Module):
        def forward(self, x):
            return torch.randn(1, 1, 100)  # Mock output

    model = MockModel()
    output_weights = torch.randn(2, 100)  # 2 neurons, 100 vocab size

    # Initialize reflector
    reflector = OptimizedSVMHyperplaneReflection(
        model=model, svm_checkpoint_path=test_path, output_projection_weights=output_weights
    )

    # Test reflection
    test_point = np.array([2.0, 1.0])
    reflected = reflector.reflect_across_hyperplane(test_point)

    print(f"Original point: {test_point}")
    print(f"Reflected point: {reflected}")

    # Verify reflection properties
    orig_side = np.dot(test_point, normal_vector) + intercept
    refl_side = np.dot(reflected, normal_vector) + intercept

    print(f"Original side value: {orig_side}")
    print(f"Reflected side value: {refl_side}")
    print(f"Are on opposite sides? {np.sign(orig_side) != np.sign(refl_side)}")

    # Clean up
    test_path.unlink()


#######################################################################################################
# Entry point of the script
#######################################################################################################


def main() -> None:
    """Main function demonstrating usage."""
    test_hyperplane_reflection()


if __name__ == "__main__":
    main()
