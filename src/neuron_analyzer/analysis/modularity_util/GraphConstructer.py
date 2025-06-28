import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import torch
from sklearn.feature_selection import mutual_info_regression

warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


#######################################################
# Construct graph


@dataclass
class GraphConfig:
    """Configuration for graph construction parameters."""

    correlation_threshold: float = 0.3
    mi_threshold: float = 0.1
    preserve_edge_signs: bool = True
    apply_abs: bool = True
    min_edge_weight: float = 0.0
    max_edges_per_node: int | None = None


class BaseGraphBuilder(ABC):
    """Abstract base class for graph construction methods."""

    def __init__(self, config: GraphConfig):
        self.config = config
        self.preserve_signs = config.preserve_edge_signs
        self.apply_abs = config.apply_abs

    @abstractmethod
    def compute_edge_weights(self, data: np.ndarray) -> np.ndarray:
        """Compute edge weight matrix from activation data."""

    def build_graph(self, data: np.ndarray, context_mask: np.ndarray | None = None) -> nx.Graph:
        """Build NetworkX graph from activation data."""
        # Apply context mask if provided
        if context_mask is not None:
            if len(context_mask) != data.shape[0]:
                raise ValueError("Context mask length must match number of contexts")
            data = data[context_mask]

        if data.shape[0] < 2:
            logger.warning("Insufficient data for graph construction")
            return self._create_empty_graph(data.shape[1])

        # Compute edge weights
        edge_matrix = self.compute_edge_weights(data)

        # Apply thresholding and filtering
        edge_matrix = self._apply_filtering(edge_matrix)

        # Create NetworkX graph
        return self._create_networkx_graph(edge_matrix)

    def _apply_filtering(self, edge_matrix: np.ndarray) -> np.ndarray:
        """Apply thresholding and filtering to edge matrix."""
        # Apply minimum weight threshold
        if self.config.min_edge_weight > 0:
            if self.config.apply_abs:
                edge_matrix[np.abs(edge_matrix) < self.config.min_edge_weight] = 0.0
            else:
                edge_matrix[edge_matrix < self.config.min_edge_weight] = 0.0

        # Limit edges per node if specified
        if self.config.max_edges_per_node is not None:
            edge_matrix = self._limit_edges_per_node(edge_matrix)

        # Zero out diagonal
        np.fill_diagonal(edge_matrix, 0.0)

        return edge_matrix

    def _limit_edges_per_node(self, edge_matrix: np.ndarray) -> np.ndarray:
        """Limit maximum number of edges per node by keeping strongest connections."""
        n_nodes = edge_matrix.shape[0]
        filtered_matrix = np.zeros_like(edge_matrix)

        for i in range(n_nodes):
            # Get edge strengths for this node
            edge_strengths = np.abs(edge_matrix[i]) if self.config.apply_abs else edge_matrix[i]
            # Find indices of strongest edges
            top_indices = np.argpartition(edge_strengths, -self.config.max_edges_per_node)[
                -self.config.max_edges_per_node :
            ]

            # Keep only top edges (preserve original signs)
            filtered_matrix[i, top_indices] = edge_matrix[i, top_indices]

        # Make symmetric
        filtered_matrix = (filtered_matrix + filtered_matrix.T) / 2

        return filtered_matrix

    def _create_networkx_graph(self, edge_matrix: np.ndarray) -> nx.Graph:
        """Convert edge matrix to NetworkX graph."""
        G = nx.Graph()
        n_nodes = edge_matrix.shape[0]

        # Add nodes
        G.add_nodes_from(range(n_nodes))

        # Add edges with weights
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                weight = edge_matrix[i, j]
                if weight != 0:
                    G.add_edge(i, j, weight=weight)
                    if self.preserve_signs:
                        G[i][j]["sign"] = np.sign(weight)

        # Add graph metadata
        G.graph["signed"] = self.preserve_signs
        G.graph["builder_type"] = self.__class__.__name__
        G.graph["n_edges"] = G.number_of_edges()

        if self.preserve_signs:
            pos_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("weight", 0) > 0)
            neg_edges = G.number_of_edges() - pos_edges
            G.graph["positive_edges"] = pos_edges
            G.graph["negative_edges"] = neg_edges

        logger.debug(f"Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        return G

    def _create_empty_graph(self, n_nodes: int) -> nx.Graph:
        """Create empty graph with specified number of nodes."""
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        G.graph["signed"] = self.preserve_signs
        G.graph["builder_type"] = self.__class__.__name__
        return G


class CorrelationGraphBuilder(BaseGraphBuilder):
    """Graph builder using Pearson correlation coefficients."""

    def compute_edge_weights(self, data: np.ndarray) -> np.ndarray:
        """Compute correlation-based edge weights."""
        try:
            # Compute correlation matrix
            corr_matrix = np.corrcoef(data.T)

            # Handle NaN values (from constant variables)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)

            # Apply threshold
            if self.preserve_signs:
                # Keep signs but threshold by absolute value
                if self.apply_abs:
                    threshold_mask = np.abs(corr_matrix) >= self.config.correlation_threshold
                else:
                    threshold_mask = corr_matrix >= self.config.correlation_threshold
                edge_matrix = corr_matrix * threshold_mask
            else:
                # Use absolute values
                edge_matrix = np.abs(corr_matrix) if self.apply_abs else corr_matrix
                edge_matrix[edge_matrix < self.config.correlation_threshold] = 0.0

            return edge_matrix

        except Exception as e:
            logger.warning(f"Error computing correlations: {e}")
            return np.zeros((data.shape[1], data.shape[1]))


#######################################################
# MI builder


class MutualInfoGraphBuilder(BaseGraphBuilder):
    """Graph builder using mutual information."""

    def __init__(self, config: GraphConfig, n_neighbors: int = 3, random_state: int = 42):
        super().__init__(config)
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def compute_edge_weights(self, data: np.ndarray) -> np.ndarray:
        """Compute mutual information-based edge weights."""
        n_neurons = data.shape[1]
        mi_matrix = np.zeros((n_neurons, n_neurons))

        try:
            for i in range(n_neurons):
                for j in range(i + 1, n_neurons):
                    mi_val = self._compute_mutual_info(data[:, i], data[:, j])
                    mi_matrix[i, j] = mi_val
                    mi_matrix[j, i] = mi_val

            # Apply threshold
            mi_matrix[mi_matrix < self.config.mi_threshold] = 0.0

            return mi_matrix

        except Exception as e:
            logger.warning(f"Error computing mutual information: {e}")
            return np.zeros((n_neurons, n_neurons))

    def _compute_mutual_info(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute mutual information between two variables."""
        try:
            # Remove invalid values
            valid_mask = np.isfinite(x) & np.isfinite(y)
            if np.sum(valid_mask) < 3:
                return 0.0

            x_clean = x[valid_mask]
            y_clean = y[valid_mask]

            # Check for constant variables
            if np.std(x_clean) == 0 or np.std(y_clean) == 0:
                return 0.0

            # Compute mutual information
            x_reshaped = x_clean.reshape(-1, 1)
            mi = mutual_info_regression(
                x_reshaped,
                y_clean,
                discrete_features=False,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state,
            )

            return float(mi[0])

        except Exception as e:
            logger.debug(f"MI computation failed: {e}")
            return 0.0


#######################################################
# Hybrid graph builder


class HybridGraphBuilder(BaseGraphBuilder):
    """Graph builder combining correlation and mutual information."""

    def __init__(
        self, config: GraphConfig, correlation_weight: float = 0.6, mi_weight: float = 0.4, require_both: bool = True
    ):
        super().__init__(config)
        self.correlation_weight = correlation_weight
        self.mi_weight = mi_weight
        self.require_both = require_both

        # Create sub-builders
        self.corr_builder = CorrelationGraphBuilder(config)
        self.mi_builder = MutualInfoGraphBuilder(config)

    def compute_edge_weights(self, data: np.ndarray) -> np.ndarray:
        """Compute hybrid edge weights from correlation and MI."""
        try:
            # Get correlation matrix
            corr_matrix = self.corr_builder.compute_edge_weights(data)

            # Get MI matrix
            mi_matrix = self.mi_builder.compute_edge_weights(data)

            if self.preserve_signs:
                # Use correlation signs with MI magnitudes
                corr_signs = np.sign(corr_matrix)
                mi_magnitudes = mi_matrix

                if self.require_both:
                    # Require both correlation and MI thresholds
                    if self.apply_abs:
                        corr_mask = np.abs(corr_matrix) >= self.config.correlation_threshold
                    else:
                        corr_mask = corr_matrix >= self.config.correlation_threshold
                    mi_mask = mi_matrix >= self.config.mi_threshold
                    combined_mask = corr_mask & mi_mask

                    edge_matrix = corr_signs * mi_magnitudes * combined_mask
                else:
                    # Weighted combination
                    edge_matrix = self.correlation_weight * corr_matrix + self.mi_weight * mi_matrix * np.sign(
                        corr_matrix
                    )
            else:
                # Standard weighted combination for unsigned graphs
                if self.apply_abs:
                    edge_matrix = self.correlation_weight * np.abs(corr_matrix) + self.mi_weight * mi_matrix
                else:
                    edge_matrix = self.correlation_weight * corr_matrix + self.mi_weight * mi_matrix
                if self.require_both:
                    if self.apply_abs:
                        corr_mask = np.abs(corr_matrix) >= self.config.correlation_threshold
                    else:
                        corr_mask = corr_matrix >= self.config.correlation_threshold
                    mi_mask = mi_matrix >= self.config.mi_threshold
                    edge_matrix *= corr_mask & mi_mask

            return edge_matrix

        except Exception as e:
            logger.warning(f"Error computing hybrid weights: {e}")
            return np.zeros((data.shape[1], data.shape[1]))


#######################################################
# Graph builder


class GraphBuilder:
    """Main graph builder class that orchestrates different building methods."""

    def __init__(self, method: str = "correlation", config: GraphConfig | None = None):
        """Initialize GraphBuilder with specified method."""
        self.method = method
        self.config = config or GraphConfig()
        self.builder = self._create_builder()

    def _create_builder(self) -> BaseGraphBuilder:
        """Create appropriate graph builder based on method."""
        if self.method == "correlation":
            return CorrelationGraphBuilder(self.config)
        if self.method == "mi":
            return MutualInfoGraphBuilder(self.config)
        if self.method == "hybrid":
            return HybridGraphBuilder(self.config)
        raise ValueError(f"Unknown graph building method: {self.method}")

    def build_graph(self, data: np.ndarray, context_mask: np.ndarray | None = None) -> nx.Graph:
        """Build graph from activation data."""
        return self.builder.build_graph(data, context_mask)

    def build_graphs_for_groups(
        self, activation_tensors: dict[str, torch.Tensor], context_masks: dict[str, np.ndarray] | None = None
    ) -> dict[str, nx.Graph]:
        """Build graphs for multiple neuron groups."""
        graphs = {}

        for group_name, tensor in activation_tensors.items():
            # Convert tensor to numpy
            data = tensor.detach().cpu().numpy()

            # Get context mask for this group if provided
            context_mask = None
            if context_masks and group_name in context_masks:
                context_mask = context_masks[group_name]

            # Build graph
            try:
                graph = self.build_graph(data, context_mask)
                graphs[group_name] = graph
                logger.debug(f"Built graph for {group_name}: {graph.number_of_edges()} edges")
            except Exception as e:
                logger.warning(f"Failed to build graph for {group_name}: {e}")
                graphs[group_name] = self.builder._create_empty_graph(data.shape[1])

        return graphs

    def build_context_specific_graphs(
        self, data: np.ndarray, rare_mask: np.ndarray, common_mask: np.ndarray | None = None
    ) -> tuple[nx.Graph, nx.Graph]:
        """Build separate graphs for rare and common contexts."""
        if common_mask is None:
            common_mask = ~rare_mask

        # Check that we have sufficient data for both contexts
        if np.sum(rare_mask) < 2:
            logger.warning("Insufficient rare contexts for graph construction")
            rare_graph = self.builder._create_empty_graph(data.shape[1])
        else:
            rare_graph = self.build_graph(data, rare_mask)

        if np.sum(common_mask) < 2:
            logger.warning("Insufficient common contexts for graph construction")
            common_graph = self.builder._create_empty_graph(data.shape[1])
        else:
            common_graph = self.build_graph(data, common_mask)

        return rare_graph, common_graph

    def get_edge_statistics(self, graph: nx.Graph) -> dict[str, Any]:
        """Get statistics about the constructed graph edges."""
        stats = {
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "method": self.method,
            "config": self.config.__dict__,
        }

        if graph.number_of_edges() > 0:
            weights = [d["weight"] for _, _, d in graph.edges(data=True)]
            stats.update(
                {
                    "weight_mean": float(np.mean(weights)),
                    "weight_std": float(np.std(weights)),
                    "weight_min": float(np.min(weights)),
                    "weight_max": float(np.max(weights)),
                }
            )

            if self.config.preserve_edge_signs:
                positive_weights = [w for w in weights if w > 0]
                negative_weights = [w for w in weights if w < 0]

                stats.update(
                    {
                        "n_positive_edges": len(positive_weights),
                        "n_negative_edges": len(negative_weights),
                        "edge_balance_ratio": len(positive_weights) / len(weights) if weights else 0.5,
                    }
                )

        return stats

    @staticmethod
    def create_binary_graph_builder(threshold: float = 0.3) -> "GraphBuilder":
        """Create builder for binary graphs."""
        config = GraphConfig(
            correlation_threshold=threshold, preserve_edge_signs=False, apply_abs=False, min_edge_weight=threshold
        )
        return GraphBuilder("correlation", config)

    @staticmethod
    def create_signed_graph_builder(threshold: float = 0.3) -> "GraphBuilder":
        """Create builder for signed weighted graphs."""
        config = GraphConfig(
            correlation_threshold=threshold, preserve_edge_signs=True, apply_abs=False, min_edge_weight=0.0
        )
        return GraphBuilder("correlation", config)

    @staticmethod
    def create_sparse_graph_builder(max_edges_per_node: int = 10) -> "GraphBuilder":
        """Create builder for sparse graphs with limited connectivity."""
        config = GraphConfig(
            correlation_threshold=0.1, preserve_edge_signs=True, apply_abs=False, max_edges_per_node=max_edges_per_node
        )
        return GraphBuilder("hybrid", config)
