import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Import all the restructured classes
from neuron_analyzer.analysis.a_graph_util.DataLoader import DataManager
from neuron_analyzer.analysis.a_graph_util.GraphConstructer import GraphBuilder, GraphConfig
from neuron_analyzer.analysis.a_graph_util.HypothesisTester import HypothesisTestSuite
from neuron_analyzer.analysis.a_graph_util.NetworkAnalyzer import NetworkAnalyzer
from neuron_analyzer.analysis.a_graph_util.ResultManager import ResultsManager
from neuron_analyzer.analysis.a_graph_util.StatValidator import StatisticalValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("neural_coordination_analysis.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

#######################################################
# Default Analysis config


@dataclass
class AnalysisConfig:
    """Configuration for the complete analysis pipeline."""

    # Data parameters
    activation_column: str = "activation"
    token_column: str = "str_tokens"
    context_column: str = "context"
    component_column: str = "component_name"

    # Graph construction
    correlation_threshold: float = 0.3
    mi_threshold: float = 0.1
    edge_construction_method: str = "correlation"  # "correlation", "mi", "hybrid"
    preserve_edge_signs: bool = True
    apply_abs: bool = True

    # Analysis parameters
    num_random_groups: int = 2
    min_graph_size: int = 3
    random_seed: int = 42

    # Statistical validation
    null_model_type: str = "permutation"  # "permutation", "configuration", "signed"
    n_null_samples: int = 1000
    n_bootstrap: int = 1000
    significance_level: float = 0.05
    multiple_testing_correction: str = "bonferroni"

    # Hypothesis testing
    hierarchy_threshold: float = 0.1
    modularity_threshold: float = 0.3
    adaptivity_threshold: float = 0.1
    optimization_threshold: float = 0.5


#######################################################
# Entry point to run the func


def run_all_analyses(
    activation_data: pd.DataFrame,
    boost_neuron_indices: list[int],
    suppress_neuron_indices: list[int],
    excluded_neuron_indices: list[int] | None = None,
    rare_token_mask: np.ndarray | None = None,
    config: AnalysisConfig | None = None,
    data_description: str = "Neural activation analysis",
) -> dict[str, Any]:
    """Run complete neural coordination analysis using the restructured framework."""
    # Use default config if none provided
    if config is None:
        config = AnalysisConfig()

    if excluded_neuron_indices is None:
        excluded_neuron_indices = []

    logger.info("=" * 80)
    logger.info("STARTING NEURAL COORDINATION ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Data description: {data_description}")
    logger.info(f"Boost neurons: {len(boost_neuron_indices)}")
    logger.info(f"Suppress neurons: {len(suppress_neuron_indices)}")
    logger.info(f"Excluded neurons: {len(excluded_neuron_indices)}")
    logger.info(f"Random seed: {config.random_seed}")

    try:
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
            random_seed=config.random_seed,
        )

        # Create activation tensors
        activation_tensors = data_manager.create_activation_tensors(neuron_groups)

        # Create rare token mask
        rare_mask = data_manager.create_rare_token_mask(rare_token_mask)
        common_mask = ~rare_mask

        # Log data summary
        data_summary = data_manager.get_data_summary()
        logger.info(f"Data summary: {data_summary['n_contexts']} contexts, {data_summary['n_neurons']} neurons")

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

        graph_builder = GraphBuilder(method=config.edge_construction_method, config=graph_config)

        # Build graphs for all neuron groups
        graphs = graph_builder.build_graphs_for_groups(activation_tensors)

        # Build context-specific graphs (rare vs common)
        context_graphs = {}
        for group_name, tensor in activation_tensors.items():
            data = tensor.detach().cpu().numpy()

            # Check if we have sufficient data for both contexts
            if np.sum(rare_mask) >= config.min_graph_size and np.sum(common_mask) >= config.min_graph_size:
                rare_graph, common_graph = graph_builder.build_context_specific_graphs(data, rare_mask, common_mask)
                context_graphs[group_name] = {"rare_context": rare_graph, "common_context": common_graph}
            else:
                logger.warning(f"Insufficient data for context analysis in {group_name}")
                context_graphs[group_name] = {"rare_context": None, "common_context": None}

        # Log graph statistics
        for group_name, graph in graphs.items():
            stats = graph_builder.get_edge_statistics(graph)
            logger.info(
                f"{group_name}: {stats['n_nodes']} nodes, {stats['n_edges']} edges, density: {stats['density']:.3f}"
            )

        # =========================================================================
        # STEP 3: NETWORK ANALYSIS
        # =========================================================================
        logger.info("\nSTEP 3: Network Analysis")
        logger.info("-" * 50)

        # Initialize network analyzer
        network_analyzer = NetworkAnalyzer()

        # Analyze all networks
        network_analyses = network_analyzer.analyze_multiple_networks(graphs)

        # Analyze context-specific networks
        context_analyses = {}
        for group_name, context_graphs_group in context_graphs.items():
            if context_graphs_group["rare_context"] and context_graphs_group["common_context"]:
                rare_analysis = network_analyzer.analyze_network(context_graphs_group["rare_context"])
                common_analysis = network_analyzer.analyze_network(context_graphs_group["common_context"])

                context_analyses[group_name] = {"rare_context": rare_analysis, "common_context": common_analysis}

        # Log analysis results
        for group_name, analysis in network_analyses.items():
            summary = network_analyzer.get_analysis_summary(analysis)
            logger.info(f"\n{group_name} Network Analysis:")
            for line in summary.split("\n")[:5]:  # First 5 lines
                logger.info(f"  {line}")

        # =========================================================================
        # STEP 4: HYPOTHESIS TESTING
        # =========================================================================
        logger.info("\nSTEP 4: Hypothesis Testing")
        logger.info("-" * 50)

        # Configure hypothesis testers
        hierarchy_config = {"hub_threshold": config.hierarchy_threshold}
        modularity_config = {"modularity_threshold": config.modularity_threshold}
        adaptivity_config = {"topology_change_threshold": config.adaptivity_threshold}
        optimization_config = {"efficiency_threshold": config.optimization_threshold}

        # Initialize hypothesis test suite
        hypothesis_suite = HypothesisTestSuite(
            hierarchy_config=hierarchy_config,
            modularity_config=modularity_config,
            adaptivity_config=adaptivity_config,
            optimization_config=optimization_config,
        )

        # Run hypothesis tests for all networks
        hypothesis_results = {}
        for group_name in network_analyses.keys():
            logger.info(f"Testing hypotheses for {group_name}...")

            # Prepare context data if available
            context_data = context_analyses.get(group_name)

            # Run tests
            group_results = hypothesis_suite.run_all_tests(
                network_analyses[group_name], graphs[group_name], context_data=context_data
            )

            hypothesis_results[group_name] = group_results

            # Log results
            supported_count = sum(1 for result in group_results.values() if result.supported)
            logger.info(f"  {group_name}: {supported_count}/{len(group_results)} hypotheses supported")

        # Generate comparative hypothesis report
        comparative_report = hypothesis_suite.generate_comparative_report(hypothesis_results)
        logger.info("\nHypothesis Testing Summary:")
        logger.info(f"  Networks analyzed: {comparative_report['summary']['n_networks']}")
        logger.info(
            f"  Strongest evidence: {comparative_report.get('strongest_evidence', {}).get('hypothesis', 'None')}"
        )

        # =========================================================================
        # STEP 5: STATISTICAL VALIDATION
        # =========================================================================
        logger.info("\nSTEP 5: Statistical Validation")
        logger.info("-" * 50)

        # Initialize statistical validator
        statistical_validator = StatisticalValidator(
            null_model_type=config.null_model_type,
            n_null_samples=config.n_null_samples,
            n_bootstrap=config.n_bootstrap,
            significance_level=config.significance_level,
            multiple_testing_correction=config.multiple_testing_correction,
            random_seed=config.random_seed,
        )

        # Validate network metrics for each group
        validation_results = {}
        for group_name in network_analyses.keys():
            logger.info(f"Validating metrics for {group_name}...")

            # Extract observed metrics
            analysis = network_analyses[group_name]
            observed_metrics = {}

            # Topology metrics
            if "topology" in analysis:
                for metric, value in analysis["topology"].items():
                    if isinstance(value, (int, float)):
                        observed_metrics[f"topology_{metric}"] = value

            # Community metrics
            if "communities" in analysis:
                for metric, value in analysis["communities"].items():
                    if isinstance(value, (int, float)):
                        observed_metrics[f"community_{metric}"] = value

            # Get original data for this group
            if group_name == "boost":
                group_data = data_manager.create_activation_matrix(neuron_groups.boost)
            elif group_name == "suppress":
                group_data = data_manager.create_activation_matrix(neuron_groups.suppress)
            else:
                # Random group
                group_idx = int(group_name.split("_")[1]) - 1
                if group_idx < len(neuron_groups.random_groups):
                    random_group_name = f"random_{group_idx + 1}"
                    group_indices = neuron_groups.random_groups[random_group_name]
                    group_data = data_manager.create_activation_matrix(group_indices)
                else:
                    continue

            # Validate metrics
            group_validation = statistical_validator.validate_network_metrics(
                observed_metrics, group_data, graphs[group_name]
            )

            validation_results[group_name] = group_validation

            # Log validation summary
            significant_count = sum(1 for result in group_validation.values() if result.is_significant)
            logger.info(f"  {group_name}: {significant_count}/{len(group_validation)} metrics significant")

        # =========================================================================
        # STEP 6: RESULTS MANAGEMENT AND AGGREGATION
        # =========================================================================
        logger.info("\nSTEP 6: Results Management and Aggregation")
        logger.info("-" * 50)

        # Initialize results manager (in-memory only)
        results_manager = ResultsManager()

        # Store analysis metadata
        analysis_id = results_manager.store_analysis_metadata(
            data_description=data_description,
            neuron_groups={
                "boost": neuron_groups.boost,
                "suppress": neuron_groups.suppress,
                **{f"random_{i + 1}": indices for i, indices in enumerate(neuron_groups.random_groups.values())},
            },
            configuration=config.__dict__,
            random_seed=config.random_seed,
        )

        # Store all results
        for group_name in network_analyses.keys():
            results_manager.store_network_analysis_results(group_name, network_analyses[group_name])
            results_manager.store_hypothesis_test_results(group_name, hypothesis_results[group_name])
            results_manager.store_statistical_validation_results(group_name, validation_results[group_name])

        # Generate comprehensive aggregated results
        aggregated_results = results_manager.aggregate_results()

        # Generate summary report
        summary_text = results_manager.generate_summary_report()
        logger.info("\n" + "=" * 80)
        logger.info("ANALYSIS SUMMARY")
        logger.info("=" * 80)
        logger.info(summary_text)

        # =========================================================================
        # STEP 7: CLEANUP AND FINAL RESULTS
        # =========================================================================
        logger.info("\nSTEP 7: Cleanup and Final Results")
        logger.info("-" * 50)

        # Clean up GPU memory if used
        data_manager.cleanup_tensors(activation_tensors)

        # Prepare final results dictionary
        final_results = {
            "analysis_id": analysis_id,
            "config": config.__dict__,  # Convert to dict for serialization
            "data_summary": data_summary,
            "neuron_groups": {
                "boost": neuron_groups.boost,
                "suppress": neuron_groups.suppress,
                "random_groups": neuron_groups.random_groups,
            },
            "graphs": graphs,  # Note: NetworkX graphs are not JSON serializable
            "context_graphs": context_graphs,
            "network_analyses": network_analyses,
            "context_analyses": context_analyses,
            "hypothesis_results": hypothesis_results,
            "validation_results": validation_results,
            "comparative_report": comparative_report,
            "aggregated_results": aggregated_results,  # This contains all serialized results
            "summary_report": summary_text,  # Text summary for easy access
            "analysis_metadata": {
                "total_networks": len(network_analyses),
                "analysis_completed": True,
                "processing_time": None,  # Could add timing if needed
                "results_manager_summary": results_manager.get_storage_summary(),
            },
        }

        # Log final summary
        logger.info("Analysis completed successfully!")
        logger.info(f"Analysis ID: {analysis_id}")
        logger.info(f"Networks analyzed: {len(network_analyses)}")
        logger.info(f"Hypothesis tests completed: {len(hypothesis_results)}")
        logger.info(f"Statistical validations: {len(validation_results)}")

        # Log key findings
        if aggregated_results.get("summary"):
            summary_data = aggregated_results["summary"]
            total_supported = sum(summary_data.get("supported_hypotheses", {}).values())
            total_hypotheses = len(summary_data.get("supported_hypotheses", {})) * len(network_analyses)
            logger.info(f"Overall hypothesis support: {total_supported}/{total_hypotheses}")
            logger.info(f"Overall analysis score: {summary_data.get('overall_score', 0):.3f}")

        return final_results

    except Exception as e:
        logger.error(f"Analysis failed with error: {e}")
        logger.error("Stack trace:", exc_info=True)
        raise
