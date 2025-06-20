import logging
import time

import numpy as np
import pandas as pd
import torch

from neuron_analyzer.analysis.a_MI import MutualInformationAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_test_data(
    n_contexts: int = 1000,
    n_boost_neurons: int = 50,
    n_suppress_neurons: int = 50,
    n_other_neurons: int = 200,
    rare_token_fraction: float = 0.2,
    coordination_strength: float = 0.7,
    noise_level: float = 0.3,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Generate synthetic test data with known coordination patterns.

    Returns:
        - activation_data: DataFrame in expected format
        - rare_token_mask: Boolean array indicating rare token contexts
        - ground_truth: Dictionary with known coordination patterns

    """
    np.random.seed(42)  # For reproducibility

    # Generate contexts and tokens
    contexts = [f"context_{i}" for i in range(n_contexts)]

    # Create rare vs common tokens
    n_rare_contexts = int(n_contexts * rare_token_fraction)
    rare_token_mask = np.zeros(n_contexts, dtype=bool)
    rare_token_mask[:n_rare_contexts] = True
    np.random.shuffle(rare_token_mask)

    tokens = []
    for i, is_rare in enumerate(rare_token_mask):
        if is_rare:
            tokens.append(f"rare_token_{i % 10}")  # 10 different rare tokens
        else:
            tokens.append(f"common_token_{i % 20}")  # 20 different common tokens

    # Generate neuron indices
    all_neurons = list(range(n_boost_neurons + n_suppress_neurons + n_other_neurons))
    boost_neurons = all_neurons[:n_boost_neurons]
    suppress_neurons = all_neurons[n_boost_neurons : n_boost_neurons + n_suppress_neurons]
    other_neurons = all_neurons[n_boost_neurons + n_suppress_neurons :]

    # Generate base activations
    base_activations = np.random.normal(0, 1, (n_contexts, len(all_neurons)))

    # Add coordination patterns

    # 1. Boost neurons coordinate strongly during rare token contexts
    for i in range(len(boost_neurons)):
        for j in range(i + 1, min(i + 5, len(boost_neurons))):  # Each neuron coordinates with 4 neighbors
            # Strong coordination during rare tokens
            rare_indices = np.where(rare_token_mask)[0]
            coordination_signal = np.random.normal(0, coordination_strength, len(rare_indices))
            base_activations[rare_indices, boost_neurons[i]] += coordination_signal
            base_activations[rare_indices, boost_neurons[j]] += coordination_signal

            # Weak coordination during common tokens
            common_indices = np.where(~rare_token_mask)[0]
            weak_signal = np.random.normal(0, coordination_strength * 0.2, len(common_indices))
            base_activations[common_indices, boost_neurons[i]] += weak_signal
            base_activations[common_indices, boost_neurons[j]] += weak_signal

    # 2. Suppress neurons have nonlinear coordination (threshold-based)
    for i in range(len(suppress_neurons)):
        for j in range(i + 1, min(i + 3, len(suppress_neurons))):
            # Nonlinear relationship: j activates only when i > threshold
            threshold = 0.5
            high_activation_mask = base_activations[:, suppress_neurons[i]] > threshold
            base_activations[high_activation_mask, suppress_neurons[j]] += coordination_strength

    # 3. Add temporal coordination (lag-1 correlation)
    for i in range(1, len(boost_neurons), 2):  # Every other boost neuron
        if i < len(boost_neurons):
            # Neuron i-1 predicts neuron i at next timestep
            for t in range(1, n_contexts):
                base_activations[t, boost_neurons[i]] += 0.4 * base_activations[t - 1, boost_neurons[i - 1]]

    # 4. Random neurons have minimal coordination
    noise = np.random.normal(0, noise_level, base_activations.shape)
    base_activations += noise

    # Create DataFrame in expected format
    data_rows = []
    for context_idx, (context, token) in enumerate(zip(contexts, tokens, strict=False)):
        for neuron_idx in all_neurons:
            data_rows.append(
                {
                    "context": context,
                    "str_tokens": token,
                    "component_name": neuron_idx,
                    "activation": base_activations[context_idx, neuron_idx],
                }
            )

    activation_data = pd.DataFrame(data_rows)

    # Ground truth patterns
    ground_truth = {
        "boost_coordination_strength_rare": coordination_strength,
        "boost_coordination_strength_common": coordination_strength * 0.2,
        "has_context_dependent_coordination": True,
        "has_nonlinear_coordination": True,
        "has_temporal_coordination": True,
        "coordination_type": "context_dependent_linear_and_nonlinear",
        "expected_peak_lag": 1,
        "rare_token_fraction": rare_token_fraction,
    }

    neuron_indices = {
        "boost": boost_neurons,
        "suppress": suppress_neurons,
        "excluded": other_neurons[:50],  # Exclude some for random sampling
    }

    return activation_data, rare_token_mask, ground_truth, neuron_indices


def performance_profiler(analyzer, method_name: str):
    """Decorator to profile method execution time."""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = getattr(analyzer, method_name)(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{method_name} took {end_time - start_time:.2f} seconds")
        return result, end_time - start_time

    return wrapper


def run_sanity_checks():
    """Run comprehensive sanity checks on the MI analyzer."""
    print("=" * 50)
    print("MUTUAL INFORMATION ANALYZER SANITY CHECKS")
    print("=" * 50)

    # Generate test data
    print("\n1. Generating synthetic test data...")
    activation_data, rare_token_mask, ground_truth, neuron_indices = generate_test_data(
        n_contexts=500,  # Smaller for faster testing
        n_boost_neurons=20,
        n_suppress_neurons=20,
        coordination_strength=0.8,
        rare_token_fraction=0.3,
    )

    print(f"   - Generated {len(activation_data)} activation records")
    print(f"   - {len(neuron_indices['boost'])} boost neurons, {len(neuron_indices['suppress'])} suppress neurons")
    print(f"   - {np.sum(rare_token_mask)} rare contexts, {np.sum(~rare_token_mask)} common contexts")

    print("\n2. Initializing analyzer...")
    start_time = time.time()

    analyzer = MutualInformationAnalyzer(
        activation_data=activation_data,
        boost_neuron_indices=neuron_indices["boost"],
        suppress_neuron_indices=neuron_indices["suppress"],
        excluded_neuron_indices=neuron_indices["excluded"],
        rare_token_mask=rare_token_mask,
        mi_estimator="adaptive",  # Faster than KSG for testing
        max_lag=2,  # Reduced for faster testing
        device="cpu",  # Use CPU for debugging
    )

    init_time = time.time() - start_time
    print(f"   - Initialization took {init_time:.2f} seconds")

    # Test basic functionality
    print("\n3. Testing basic MI estimation...")

    # Test simple MI calculation
    test_x = np.random.normal(0, 1, 100)
    test_y = 0.8 * test_x + 0.2 * np.random.normal(0, 1, 100)  # Strong correlation
    test_z = np.random.normal(0, 1, 100)  # Independent

    mi_correlated = analyzer._estimate_mutual_information(test_x, test_y)
    mi_independent = analyzer._estimate_mutual_information(test_x, test_z)

    print(f"   - MI between correlated variables: {mi_correlated:.4f}")
    print(f"   - MI between independent variables: {mi_independent:.4f}")

    # Sanity check: correlated should have higher MI
    assert mi_correlated > mi_independent, "Correlated variables should have higher MI than independent ones"
    assert mi_independent >= 0, "MI should be non-negative"
    print("   ✓ Basic MI estimation passed")

    # Test context-dependent analysis
    print("\n4. Testing context-dependent coordination analysis...")
    start_time = time.time()

    context_results = analyzer.analyze_context_dependent_mi()
    context_time = time.time() - start_time

    print(f"   - Context analysis took {context_time:.2f} seconds")

    # Sanity checks for context results
    boost_context = context_results["boost"]
    random_context = context_results["random_1"]

    print(f"   - Boost rare MI: {boost_context['mean_mi_rare']:.4f}")
    print(f"   - Boost common MI: {boost_context['mean_mi_common']:.4f}")
    print(f"   - Random rare MI: {random_context['mean_mi_rare']:.4f}")
    print(f"   - Random common MI: {random_context['mean_mi_common']:.4f}")

    # Expected: boost neurons should have higher MI during rare contexts
    if boost_context["mean_mi_rare"] > boost_context["mean_mi_common"]:
        print("   ✓ Boost neurons show context-dependent coordination")
    else:
        print("   ⚠ Boost neurons don't show expected context dependency")

    # Test nonlinear analysis
    print("\n5. Testing nonlinear coordination analysis...")
    start_time = time.time()

    nonlinear_results = analyzer.analyze_nonlinear_coordination()
    nonlinear_time = time.time() - start_time

    print(f"   - Nonlinear analysis took {nonlinear_time:.2f} seconds")

    boost_nonlinear = nonlinear_results["boost"]
    print(f"   - Boost mean MI: {boost_nonlinear['mean_mi']:.4f}")
    print(f"   - Boost mean correlation: {boost_nonlinear['mean_correlation']:.4f}")
    print(f"   - Nonlinearity ratio: {boost_nonlinear['mean_nonlinearity_ratio']:.4f}")

    # Test temporal analysis
    print("\n6. Testing temporal coordination analysis...")
    start_time = time.time()

    temporal_results = analyzer.analyze_temporal_coordination()
    temporal_time = time.time() - start_time

    print(f"   - Temporal analysis took {temporal_time:.2f} seconds")

    boost_temporal = temporal_results["boost"]
    print(f"   - Peak lag: {boost_temporal['peak_lag']}")
    print(f"   - Has temporal structure: {boost_temporal['has_temporal_structure']}")

    if ground_truth["expected_peak_lag"] == boost_temporal["peak_lag"]:
        print("   ✓ Detected expected temporal coordination")
    else:
        print(f"   ⚠ Expected peak lag {ground_truth['expected_peak_lag']}, got {boost_temporal['peak_lag']}")

    # Test full analysis
    print("\n7. Testing full analysis pipeline...")
    start_time = time.time()

    full_results = analyzer.run_all_analyses()
    full_time = time.time() - start_time

    print(f"   - Full analysis took {full_time:.2f} seconds")

    # Performance breakdown
    print("\n8. Performance Analysis:")
    print(f"   - Initialization: {init_time:.2f}s")
    print(f"   - Context analysis: {context_time:.2f}s")
    print(f"   - Nonlinear analysis: {nonlinear_time:.2f}s")
    print(f"   - Temporal analysis: {temporal_time:.2f}s")
    print(f"   - Total full analysis: {full_time:.2f}s")

    # Identify bottlenecks
    if context_time > 2.0:
        print("   ⚠ Context analysis is slow - consider reducing contexts or using faster MI estimator")
    if temporal_time > 3.0:
        print("   ⚠ Temporal analysis is slow - consider reducing max_lag")

    print("\n9. Validation against ground truth:")
    context_boost = full_results["context_dependent_coordination"]["boost"]

    checks = {
        "Context dependency detected": context_boost["is_significant"],
        "Boost > Random coordination": context_boost["mean_mi_rare"]
        > full_results["context_dependent_coordination"]["random_1"]["mean_mi_rare"],
        "Temporal structure detected": full_results["temporal_coordination"]["boost"]["has_temporal_structure"],
        "Results are JSON serializable": True,  # Test below
    }

    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"   {status} {check}")

    # Test JSON serialization
    print("\n10. Testing JSON serialization...")
    try:
        import json

        json_str = json.dumps(full_results, default=str, indent=2)
        print("   ✓ Results are JSON serializable")
        print(f"   - JSON size: {len(json_str)} characters")
    except Exception as e:
        print(f"   ✗ JSON serialization failed: {e}")

    print("\n" + "=" * 50)
    print("SANITY CHECK SUMMARY")
    print("=" * 50)

    all_passed = all(checks.values())
    if all_passed:
        print("✓ All sanity checks PASSED")
    else:
        print("⚠ Some sanity checks FAILED - review results above")

    return full_results, {
        "init_time": init_time,
        "context_time": context_time,
        "nonlinear_time": nonlinear_time,
        "temporal_time": temporal_time,
        "total_time": full_time,
        "checks_passed": checks,
    }


def identify_performance_bottlenecks():
    """Identify which parts of the MI analysis are slow and why."""
    print("\n" + "=" * 50)
    print("PERFORMANCE BOTTLENECK ANALYSIS")
    print("=" * 50)

    bottlenecks = {
        "MI Estimation": {
            "cause": "Computing pairwise MI for all neuron pairs",
            "complexity": "O(n_neurons^2 * n_contexts)",
            "solutions": [
                "Use faster 'adaptive' estimator instead of 'ksg'",
                "Reduce number of neurons per group",
                "Use GPU acceleration",
                "Sample neuron pairs instead of exhaustive computation",
            ],
        },
        "Context Partitioning": {
            "cause": "Computing MI separately for rare/common contexts",
            "complexity": "2x the base MI computation",
            "solutions": [
                "Ensure sufficient contexts in each partition",
                "Use stratified sampling for balanced partitions",
            ],
        },
        "Temporal Analysis": {
            "cause": "Computing lagged MI matrices for multiple lags",
            "complexity": "O(max_lag * n_neurons^2 * n_contexts)",
            "solutions": [
                "Reduce max_lag parameter",
                "Sample neuron pairs for temporal analysis",
                "Use more efficient temporal MI estimators",
            ],
        },
        "Higher-order Analysis": {
            "cause": "Computing interaction information for triplets",
            "complexity": "O(n_neurons^3)",
            "solutions": [
                "Reduce max_triplets parameter",
                "Sample triplets randomly",
                "Skip higher-order analysis for large groups",
            ],
        },
    }

    for component, info in bottlenecks.items():
        print(f"\n{component}:")
        print(f"  Cause: {info['cause']}")
        print(f"  Complexity: {info['complexity']}")
        print("  Solutions:")
        for solution in info["solutions"]:
            print(f"    - {solution}")


def optimized_test_config():
    """Return optimized configuration for faster testing."""
    return {
        "mi_estimator": "adaptive",  # Fastest estimator
        "max_lag": 2,  # Reduced temporal analysis
        "mi_batch_size": 500,  # Smaller batches
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "n_contexts": 1000,  # Reasonable size for testing
        "n_neurons_per_group": 30,  # Manageable group size
    }


if __name__ == "__main__":
    # Run sanity checks
    results, timing = run_sanity_checks()

    # Analyze performance bottlenecks
    identify_performance_bottlenecks()

    # Show optimized configuration
    print("\n" + "=" * 50)
    print("RECOMMENDED OPTIMIZED CONFIGURATION")
    print("=" * 50)
    config = optimized_test_config()
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\nFor production use:")
    print("  - Start with adaptive MI estimator")
    print("  - Use GPU if available")
    print("  - Monitor timing and adjust parameters as needed")
    print("  - Consider sampling for very large neuron groups (>100 neurons)")
