import logging
import typing as t
import warnings
from collections import Counter, defaultdict

import networkx as nx
import numpy as np

from neuron_analyzer.analysis.modularity_util.CommunityDetector import CommunityDetectionFactory, SpectralDetection

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class CommunityAnalyzer:
    """Handles community detection and modularity analysis with multiple algorithms."""

    def __init__(self, algorithm: str = "louvain", random_state: int = 42, **algorithm_params):
        self.algorithm_name = algorithm
        self.random_state = random_state
        self.algorithm_params = algorithm_params

        # Create the detection algorithm
        self.detector = CommunityDetectionFactory.create_algorithm(
            algorithm, random_state=random_state, **algorithm_params
        )

    def detect_communities(self, G: nx.Graph) -> dict[str, t.Any]:
        """Detect communities in the graph using specified algorithm.

        Args:
            G: NetworkX graph

        Returns:
            Dictionary with community assignments and metrics

        """
        if G.number_of_nodes() < 3:
            return self._trivial_communities(G)

        try:
            # Use the detection algorithm
            communities = self.detector.detect(G)

            # Compute modularity metrics
            if "labels" in communities:
                communities.update(self._compute_modularity_metrics(G, communities["labels"]))

            return communities

        except Exception as e:
            logger.warning(f"Algorithm {self.algorithm_name} failed: {e}, using fallback")
            return self._fallback_detection(G)

    def _trivial_communities(self, G: nx.Graph) -> dict[str, t.Any]:
        """Handle trivial cases with few nodes."""
        n_nodes = G.number_of_nodes()
        return {
            "labels": [0] * n_nodes,
            "n_communities": 1 if n_nodes > 0 else 0,
            "modularity": 0.0,
            "signed_modularity": 0.0,
            "community_sizes": [n_nodes] if n_nodes > 0 else [],
            "algorithm": "trivial",
        }

    def _fallback_detection(self, G: nx.Graph) -> dict[str, t.Any]:
        """Fallback to simple spectral clustering."""
        try:
            fallback_detector = SpectralDetection(random_state=self.random_state)
            communities = fallback_detector.detect(G)
            communities.update(self._compute_modularity_metrics(G, communities["labels"]))
            communities["algorithm"] = "fallback_spectral"
            return communities
        except Exception:
            return self._trivial_communities(G)

    def _compute_modularity_metrics(self, G: nx.Graph, labels: list[int]) -> dict[str, t.Any]:
        """Compute various modularity metrics."""
        metrics = {}

        # Basic modularity
        try:
            communities_sets = self._labels_to_sets(labels)
            metrics["modularity"] = nx.community.modularity(G, communities_sets)
        except Exception:
            metrics["modularity"] = 0.0

        # Signed modularity (if applicable)
        if G.graph.get("signed", False):
            metrics["signed_modularity"] = self._compute_signed_modularity(G, labels)
        else:
            metrics["signed_modularity"] = metrics["modularity"]

        # Community statistics
        community_sizes = list(Counter(labels).values())
        metrics.update(
            {
                "n_communities": len(set(labels)),
                "community_sizes": community_sizes,
                "avg_community_size": float(np.mean(community_sizes)) if community_sizes else 0.0,
                "modularity_density_ratio": self._compute_modularity_density_ratio(G, labels),
            }
        )

        return metrics

    def _compute_signed_modularity(self, G: nx.Graph, labels: list[int]) -> float:
        """Compute signed modularity: Q_signed = Q⁺ - Q⁻

        Formula:
        Q⁺ = (1/2m⁺) Σᵢⱼ [A⁺ᵢⱼ - (k⁺ᵢk⁺ⱼ)/(2m⁺)] δ(cᵢ,cⱼ)
        Q⁻ = (1/2m⁻) Σᵢⱼ [A⁻ᵢⱼ - (k⁻ᵢk⁻ⱼ)/(2m⁻)] δ(cᵢ,cⱼ)
        """
        try:
            nodes = list(G.nodes())
            n_nodes = len(nodes)

            if n_nodes == 0:
                return 0.0

            # Separate positive and negative edge weights
            pos_weights = {}  # (i,j) -> positive weight
            neg_weights = {}  # (i,j) -> negative weight (stored as positive)

            for i, j, data in G.edges(data=True):
                weight = data.get("weight", 1.0)
                if weight > 0:
                    pos_weights[(i, j)] = weight
                    pos_weights[(j, i)] = weight  # Undirected
                elif weight < 0:
                    neg_weights[(i, j)] = abs(weight)
                    neg_weights[(j, i)] = abs(weight)  # Undirected

            # Calculate total positive and negative weights
            m_pos = sum(pos_weights.values()) / 2  # Divide by 2 for undirected
            m_neg = sum(neg_weights.values()) / 2

            if m_pos == 0 and m_neg == 0:
                return 0.0

            # Calculate weighted degrees for positive and negative networks
            k_pos = self._calculate_weighted_degrees(nodes, pos_weights)
            k_neg = self._calculate_weighted_degrees(nodes, neg_weights)

            # Compute Q⁺ and Q⁻
            q_pos = self._compute_modularity_component(nodes, labels, pos_weights, k_pos, m_pos) if m_pos > 0 else 0.0

            q_neg = self._compute_modularity_component(nodes, labels, neg_weights, k_neg, m_neg) if m_neg > 0 else 0.0

            return q_pos - q_neg

        except Exception as e:
            logger.warning(f"Error computing signed modularity: {e}")
            return 0.0

    def _calculate_weighted_degrees(self, nodes: list, weights: dict[tuple[int, int], float]) -> dict[int, float]:
        """Calculate weighted degrees for nodes."""
        degrees = defaultdict(float)
        for (i, j), weight in weights.items():
            degrees[i] += weight
        return dict(degrees)

    def _compute_modularity_component(
        self,
        nodes: list,
        labels: list[int],
        weights: dict[tuple[int, int], float],
        degrees: dict[int, float],
        total_weight: float,
    ) -> float:
        """Compute modularity component (Q⁺ or Q⁻)."""
        if total_weight == 0:
            return 0.0

        modularity = 0.0

        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                # Only consider pairs in same community
                if labels[i] == labels[j]:
                    # Actual edge weight
                    A_ij = weights.get((node_i, node_j), 0.0)

                    # Expected weight
                    ki = degrees.get(node_i, 0.0)
                    kj = degrees.get(node_j, 0.0)
                    expected = (ki * kj) / (2 * total_weight)

                    modularity += A_ij - expected

        return modularity / (2 * total_weight)

    def _labels_to_sets(self, labels: list[int]) -> list[set]:
        """Convert community labels to sets of nodes."""
        communities = defaultdict(set)
        for node, label in enumerate(labels):
            communities[label].add(node)
        return list(communities.values())

    def _compute_modularity_density_ratio(self, G: nx.Graph, labels: list[int]) -> float:
        """Compute ratio of intra-community to inter-community density."""
        try:
            intra_edges = 0
            inter_edges = 0
            intra_possible = 0
            inter_possible = 0

            nodes = list(G.nodes())
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    same_community = labels[i] == labels[j]

                    if same_community:
                        intra_possible += 1
                        if G.has_edge(nodes[i], nodes[j]):
                            intra_edges += 1
                    else:
                        inter_possible += 1
                        if G.has_edge(nodes[i], nodes[j]):
                            inter_edges += 1

            intra_density = intra_edges / intra_possible if intra_possible > 0 else 0.0
            inter_density = inter_edges / inter_possible if inter_possible > 0 else 0.0

            return intra_density / inter_density if inter_density > 0 else float("inf")

        except Exception as e:
            logger.warning(f"Error computing density ratio: {e}")
            return 1.0

    def analyze_graph_balance(self, G: nx.Graph) -> dict[str, float]:
        """Analyze how balanced the signed graph is.

        Returns:
            Dictionary with balance metrics

        """
        pos_edges = 0
        neg_edges = 0
        pos_weight = 0.0
        neg_weight = 0.0

        for _, _, data in G.edges(data=True):
            weight = data.get("weight", 1.0)
            if weight > 0:
                pos_edges += 1
                pos_weight += weight
            elif weight < 0:
                neg_edges += 1
                neg_weight += abs(weight)

        total_edges = pos_edges + neg_edges
        total_weight = pos_weight + neg_weight

        return {
            "pos_edge_ratio": pos_edges / total_edges if total_edges > 0 else 0.0,
            "neg_edge_ratio": neg_edges / total_edges if total_edges > 0 else 0.0,
            "pos_weight_ratio": pos_weight / total_weight if total_weight > 0 else 0.0,
            "neg_weight_ratio": neg_weight / total_weight if total_weight > 0 else 0.0,
            "balance_score": min(pos_weight, neg_weight) / max(pos_weight, neg_weight)
            if max(pos_weight, neg_weight) > 0
            else 0.0,
        }
