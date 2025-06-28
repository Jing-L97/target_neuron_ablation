import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)
import logging

import numpy as np
import pandas as pd

# Import all the restructured classes
from neuron_analyzer.analysis.modularity_util.DataLoader import DataManager  # noqa: E402
from neuron_analyzer.analysis.modularity_util.GraphConstructer import GraphBuilder, GraphConfig  # noqa: E402
from neuron_analyzer.analysis.modularity_util.HypothesisTester import HypothesisTestSuite  # noqa: E402
from neuron_analyzer.analysis.modularity_util.NetworkAnalyzer import CommunityAnalyzer  # noqa: E402
from neuron_analyzer.analysis.modularity_util.ResultManager import ResultsManager  # noqa: E402
from neuron_analyzer.analysis.modularity_util.StatValidator import StatisticalValidator  # noqa: E402


@dataclass
class AnalysisConfig:
    """Simplified configuration for step-by-step analysis."""

    # Data parameters
    activation_column: str = "activation"
    token_column: str = "str_tokens"
    context_column: str = "context"
    component_column: str = "component_name"

    # Graph construction
    edge_construction_method: str = "correlation"  # "correlation", "mi", "hybrid"
    correlation_threshold: float = 0.3
    mi_threshold: float = 0.1
    preserve_edge_signs: bool = True
    apply_abs: bool = True

    # Community detection
    algorithm: str = "louvain"
    resolution: float = 1.0
    random_state: int = 42

    # Analysis parameters
    num_random_groups: int = 3
    significance_level: float = 0.05
    n_permutations: int = 1000


def run_all_analyses(
    activation_data: pd.DataFrame,
    boost_neuron_indices: list[int],
    suppress_neuron_indices: list[int],
    excluded_neuron_indices: list[int] | None = None,
    rare_token_mask: np.ndarray | None = None,
    config: AnalysisConfig | None = None,
    data_description: str = "Neural activation modularity analysis",
) -> dict[str]:
    """Run step-by-step modularity analysis using simplified classes.

    Args:
        activation_data: Neural activation DataFrame
        boost_neuron_indices: List of boost neuron indices
        suppress_neuron_indices: List of suppress neuron indices
        excluded_neuron_indices: List of excluded neuron indices
        rare_token_mask: Boolean mask for rare tokens
        config: Analysis configuration
        data_description: Description of the analysis

    Returns:
        Dictionary containing results from all steps

    """
    if config is None:
        config = AnalysisConfig()

    if excluded_neuron_indices is None:
        excluded_neuron_indices = []

    logger.info("=" * 80)
    logger.info("STARTING STEP-BY-STEP MODULARITY ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Data: {data_description}")
    logger.info(f"Edge method: {config.edge_construction_method}")
    logger.info(f"Algorithm: {config.algorithm}")

    results = {}

    # =========================================================================
    # STEP 1: DATA MANAGEMENT AND PREPROCESSING
    # =========================================================================
    logger.info("\nSTEP 1: Data Management and Preprocessing")
    logger.info("-" * 50)

    # Initialize data manager
    data_manager = DataManager(
        activation_data=activation_data,
        activation_column=config.activation_column,
        token_column=config.token_column,
        context_column=config.context_column,
        component_column=config.component_column,
        device="auto",
    )

    # Create neuron groups
    neuron_groups = data_manager.create_neuron_groups(
        boost_indices=boost_neuron_indices,
        suppress_indices=suppress_neuron_indices,
        excluded_indices=excluded_neuron_indices,
        num_random_groups=config.num_random_groups,
        random_seed=config.random_state,
    )

    # Create activation tensors
    activation_tensors = data_manager.create_activation_tensors(neuron_groups)

    # Create rare token mask
    rare_mask = data_manager.create_rare_token_mask(rare_token_mask)
    common_mask = ~rare_mask

    # Log data summary
    data_summary = data_manager.get_data_summary()
    logger.info(f"Data summary: {data_summary['n_contexts']} contexts, {data_summary['n_neurons']} neurons")

    # Store step 1 results
    results["step1_data_preprocessing"] = {
        "data_summary": data_summary,
        "neuron_groups": {
            "boost": neuron_groups.boost,
            "suppress": neuron_groups.suppress,
            "random_groups": neuron_groups.random_groups,
            "excluded": neuron_groups.excluded,
        },
        "activation_tensors": activation_tensors,
        "rare_mask": rare_mask,
        "common_mask": common_mask,
    }

    # =========================================================================
    # STEP 2: GRAPH CONSTRUCTION
    # =========================================================================
    logger.info("\nSTEP 2: Graph Construction")
    logger.info("-" * 50)

    # Configure graph builder
    graph_config = GraphConfig(
        correlation_threshold=config.correlation_threshold,
        mi_threshold=config.mi_threshold,
        preserve_edge_signs=config.preserve_edge_signs,
        apply_abs=config.apply_abs,
    )

    # Initialize graph builder
    graph_builder = GraphBuilder(method=config.edge_construction_method, config=graph_config)

    # Build graphs for all neuron groups
    graphs = graph_builder.build_graphs_for_groups(activation_tensors)

    # Log graph statistics
    graph_stats = {}
    for group_name, graph in graphs.items():
        stats = graph_builder.get_edge_statistics(graph)
        graph_stats[group_name] = stats
        logger.info(
            f"{group_name}: {stats['n_nodes']} nodes, {stats['n_edges']} edges, density: {stats['density']:.3f}"
        )

    # Store step 2 results
    results["step2_graph_construction"] = {
        "graphs": graphs,
        "graph_statistics": graph_stats,
        "construction_method": config.edge_construction_method,
        "graph_config": graph_config.__dict__,
    }

    # =========================================================================
    # STEP 3: COMMUNITY DETECTION (Using CommunityAnalyzer instead of NetworkAnalyzer)
    # =========================================================================
    logger.info("\nSTEP 3: Community Detection")
    logger.info("-" * 50)

    # Initialize community analyzer
    community_analyzer = CommunityAnalyzer(
        algorithm=config.algorithm, resolution=config.resolution, random_state=config.random_state
    )

    # Perform community detection on all graphs
    community_results = {}
    for group_name, graph in graphs.items():
        logger.info(f"Detecting communities in {group_name}...")

        # Detect communities
        communities = community_analyzer.detect_communities(graph)

        # Analyze graph balance if signed
        if graph.graph.get("signed", False):
            balance = community_analyzer.analyze_graph_balance(graph)
            communities["graph_balance"] = balance

        community_results[group_name] = communities

        # Log results
        modularity = communities.get("modularity", 0)
        signed_modularity = communities.get("signed_modularity", 0)
        n_communities = communities.get("n_communities", 0)

        logger.info(
            f"  {group_name}: Q={modularity:.3f}, Q_signed={signed_modularity:.3f}, {n_communities} communities"
        )

    # Store step 3 results
    results["step3_community_detection"] = {
        "community_results": community_results,
        "algorithm_used": config.algorithm,
        "analyzer_config": {
            "algorithm": config.algorithm,
            "resolution": config.resolution,
            "random_state": config.random_state,
        },
    }

    # =========================================================================
    # STEP 4: HYPOTHESIS TESTING (Simplified to Modularity Only)
    # =========================================================================
    logger.info("\nSTEP 4: Hypothesis Testing (Modularity)")
    logger.info("-" * 50)

    # Initialize simplified hypothesis tester (modularity only)
    hypothesis_tester = HypothesisTestSuite(
        modularity_config={
            "modularity_threshold": 0.3,
            "significance_level": config.significance_level,
            "min_communities": 2,
        }
    )

    # Test modularity hypothesis for each group
    hypothesis_results = {}
    for group_name, communities in community_results.items():
        logger.info(f"Testing modularity hypothesis for {group_name}...")

        # Test modularity hypothesis
        graph = graphs[group_name]
        result = hypothesis_tester.test_modularity_hypothesis(communities, graph)
        hypothesis_results[group_name] = result

        # Log result
        status = "SUPPORTED" if result.supported else "NOT SUPPORTED"
        logger.info(f"  {group_name}: {status} (evidence: {result.evidence_strength:.3f})")

    # Store step 4 results
    results["step4_hypothesis_testing"] = {
        "hypothesis_results": hypothesis_results,
        "supported_groups": [name for name, result in hypothesis_results.items() if result.supported],
        "overall_support": {
            "total_groups": len(hypothesis_results),
            "supported_groups": sum(1 for result in hypothesis_results.values() if result.supported),
        },
    }

    # =========================================================================
    # STEP 5: STATISTICAL VALIDATION
    # =========================================================================
    logger.info("\nSTEP 5: Statistical Validation")
    logger.info("-" * 50)

    # Initialize validator
    validator = StatisticalValidator(
        significance_level=config.significance_level,
        n_permutations=config.n_permutations,
        random_seed=config.random_state,
    )

    # Separate special and random groups
    special_groups = {}
    random_modularities = []

    for group_name, communities in community_results.items():
        signed_modularity = communities.get("signed_modularity", 0.0)

        if group_name.startswith("random"):
            random_modularities.append(signed_modularity)
        else:
            special_groups[group_name] = signed_modularity

    logger.info(f"Comparing {len(special_groups)} special groups against {len(random_modularities)} random baselines")

    # Perform statistical validation
    if random_modularities and special_groups:
        statistical_results = validator.compare_multiple_groups(special_groups, random_modularities)

        # Log results
        significant_groups = [name for name, result in statistical_results.items() if result.is_significant]
        logger.info(f"Statistical validation: {len(significant_groups)} significant groups")

        for group_name, result in statistical_results.items():
            status = "SIGNIFICANT" if result.is_significant else "NOT SIGNIFICANT"
            logger.info(f"  {group_name}: {status} (p={result.p_value:.4f})")
    else:
        logger.warning("Insufficient groups for statistical validation")
        statistical_results = {}
        significant_groups = []

    # Store step 5 results
    results["step5_statistical_validation"] = {
        "statistical_results": statistical_results,
        "random_baseline_stats": {
            "mean": np.mean(random_modularities) if random_modularities else 0.0,
            "std": np.std(random_modularities) if random_modularities else 0.0,
            "values": random_modularities,
        },
        "significant_groups": significant_groups,
        "validation_summary": {
            "total_comparisons": len(statistical_results),
            "significant_comparisons": len(significant_groups),
        },
    }

    # =========================================================================
    # STEP 6: RESULTS MANAGEMENT AND AGGREGATION
    # =========================================================================
    logger.info("\nSTEP 6: Results Management and Aggregation")
    logger.info("-" * 50)

    # Initialize results manager
    results_manager = ResultsManager()

    # Store metadata
    analysis_id = results_manager.store_metadata(
        data_description=data_description,
        neuron_groups={
            "boost": neuron_groups.boost,
            "suppress": neuron_groups.suppress,
            **{name: indices for name, indices in neuron_groups.random_groups.items()},
        },
        algorithm_config=config.__dict__,
        random_seed=config.random_state,
    )

    # Store modularity results for each group
    for group_name, communities in community_results.items():
        results_manager.store_group_modularity(
            group_name=group_name,
            modularity=communities.get("modularity", 0.0),
            signed_modularity=communities.get("signed_modularity", 0.0),
            community_details=communities,
        )

    # Store statistical comparisons
    for comparison_name, result in statistical_results.items():
        results_manager.store_statistical_comparison(f"{comparison_name}_vs_random", result)

    # Generate summary and report
    summary = results_manager.generate_summary()
    full_report = results_manager.generate_report(include_details=True)

    # Store step 6 results
    results["step6_results_management"] = {
        "analysis_id": analysis_id,
        "summary": summary,
        "full_report": full_report,
        "export_data": results_manager.export_results(),
    }

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Analysis ID: {analysis_id}")
    logger.info(f"Total groups analyzed: {len(community_results)}")
    logger.info(f"Significant groups: {len(significant_groups)}")
    logger.info(f"Best group: {summary.best_group} (Q_signed = {summary.best_modularity:.3f})")
    logger.info(f"Overall finding: {summary.overall_finding}")

    # Add final aggregated results
    results["final_summary"] = {
        "analysis_id": analysis_id,
        "config": config.__dict__,
        "total_groups": len(community_results),
        "significant_groups": len(significant_groups),
        "best_group": summary.best_group,
        "best_modularity": summary.best_modularity,
        "analysis_completed": True,
        "steps_completed": list(results.keys()),
    }

    logger.info("All steps completed successfully!")
    return results
