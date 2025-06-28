import logging
import warnings
from collections import Counter, defaultdict
from typing import Any

import networkx as nx
import numpy as np
from sklearn.cluster import SpectralClustering

warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


#######################################################
# Centrality analyzer


class CentralityAnalyzer:
    """Handles computation of various centrality measures."""

    def __init__(self, handle_disconnected: bool = True, max_iterations: int = 1000):
        self.handle_disconnected = handle_disconnected
        self.max_iterations = max_iterations

    def compute_all_centralities(self, G: nx.Graph) -> dict[str, list[float]]:
        """Compute all centrality measures for a graph."""
        if not self._validate_graph(G):
            return self._empty_centralities(G.number_of_nodes())

        centralities = {}

        # Basic centralities (always computable)
        centralities.update(self._compute_basic_centralities(G))

        # Advanced centralities (may require connected graph)
        centralities.update(self._compute_advanced_centralities(G))

        # Signed network centralities (if applicable)
        if G.graph.get("signed", False):
            centralities.update(self._compute_signed_centralities(G))

        return centralities

    def _validate_graph(self, G: nx.Graph) -> bool:
        """Validate graph for centrality computation."""
        if G.number_of_nodes() == 0:
            logger.warning("Cannot compute centralities for empty graph")
            return False
        if G.number_of_edges() == 0:
            logger.warning("Graph has no edges, centralities will be zero")
        return True

    def _empty_centralities(self, n_nodes: int) -> dict[str, list[float]]:
        """Return empty centrality dictionary for invalid graphs."""
        return {
            "degree": [0.0] * n_nodes,
            "betweenness": [0.0] * n_nodes,
            "closeness": [0.0] * n_nodes,
            "eigenvector": [0.0] * n_nodes,
            "pagerank": [1.0 / n_nodes] * n_nodes if n_nodes > 0 else [],
            "positive_degree": [0.0] * n_nodes,
            "negative_degree": [0.0] * n_nodes,
            "signed_balance": [0.5] * n_nodes,
        }

    def _compute_basic_centralities(self, G: nx.Graph) -> dict[str, list[float]]:
        """Compute basic centrality measures that always work."""
        centralities = {}

        try:
            centralities["degree"] = list(nx.degree_centrality(G).values())
        except:
            centralities["degree"] = [0.0] * G.number_of_nodes()

        try:
            centralities["betweenness"] = list(nx.betweenness_centrality(G).values())
        except:
            centralities["betweenness"] = [0.0] * G.number_of_nodes()

        try:
            centralities["closeness"] = list(nx.closeness_centrality(G).values())
        except:
            centralities["closeness"] = [0.0] * G.number_of_nodes()

        return centralities

    def _compute_advanced_centralities(self, G: nx.Graph) -> dict[str, list[float]]:
        """Compute advanced centralities that may fail on disconnected graphs."""
        centralities = {}
        n_nodes = G.number_of_nodes()

        # Eigenvector centrality
        try:
            if self._has_negative_weights(G):
                # Use absolute weights for eigenvector centrality
                G_abs = self._create_absolute_weight_graph(G)
                if nx.is_connected(G_abs):
                    centralities["eigenvector"] = list(
                        nx.eigenvector_centrality(G_abs, max_iter=self.max_iterations).values()
                    )
                else:
                    centralities["eigenvector"] = [0.0] * n_nodes
            elif nx.is_connected(G):
                centralities["eigenvector"] = list(nx.eigenvector_centrality(G, max_iter=self.max_iterations).values())
            else:
                centralities["eigenvector"] = [0.0] * n_nodes
        except (nx.PowerIterationFailedConvergence, np.linalg.LinAlgError):
            centralities["eigenvector"] = [0.0] * n_nodes

        # PageRank
        try:
            if self._has_negative_weights(G):
                G_abs = self._create_absolute_weight_graph(G)
                centralities["pagerank"] = list(nx.pagerank(G_abs, max_iter=self.max_iterations).values())
            else:
                centralities["pagerank"] = list(nx.pagerank(G, max_iter=self.max_iterations).values())
        except (nx.PowerIterationFailedConvergence, ZeroDivisionError):
            centralities["pagerank"] = [1.0 / n_nodes] * n_nodes

        return centralities

    def _compute_signed_centralities(self, G: nx.Graph) -> dict[str, list[float]]:
        """Compute signed network specific centralities."""
        n_nodes = G.number_of_nodes()
        positive_degrees = [0.0] * n_nodes
        negative_degrees = [0.0] * n_nodes

        try:
            for node in G.nodes():
                pos_sum = 0.0
                neg_sum = 0.0

                for neighbor in G.neighbors(node):
                    weight = G[node][neighbor].get("weight", 1.0)
                    if weight > 0:
                        pos_sum += weight
                    elif weight < 0:
                        neg_sum += abs(weight)

                positive_degrees[node] = pos_sum
                negative_degrees[node] = neg_sum

            # Normalize
            max_pos = max(positive_degrees) if positive_degrees else 1.0
            max_neg = max(negative_degrees) if negative_degrees else 1.0

            if max_pos > 0:
                positive_degrees = [d / max_pos for d in positive_degrees]
            if max_neg > 0:
                negative_degrees = [d / max_neg for d in negative_degrees]

            # Compute balance scores
            balance_scores = []
            for i in range(n_nodes):
                total = positive_degrees[i] + negative_degrees[i]
                if total > 0:
                    balance_scores.append(positive_degrees[i] / total)
                else:
                    balance_scores.append(0.5)

            return {
                "positive_degree": positive_degrees,
                "negative_degree": negative_degrees,
                "signed_balance": balance_scores,
            }

        except Exception as e:
            logger.warning(f"Error computing signed centralities: {e}")
            return {
                "positive_degree": [0.0] * n_nodes,
                "negative_degree": [0.0] * n_nodes,
                "signed_balance": [0.5] * n_nodes,
            }

    def _has_negative_weights(self, G: nx.Graph) -> bool:
        """Check if graph has negative edge weights."""
        return any(data.get("weight", 1.0) < 0 for _, _, data in G.edges(data=True))

    def _create_absolute_weight_graph(self, G: nx.Graph) -> nx.Graph:
        """Create copy of graph with absolute edge weights."""
        G_abs = G.copy()
        for i, j, data in G_abs.edges(data=True):
            data["weight"] = abs(data.get("weight", 1.0))
        return G_abs


#######################################################
# Modularity analyzer


class CommunityAnalyzer:
    """Handles community detection and modularity analysis."""

    def __init__(self, algorithm: str = "louvain", resolution: float = 1.0, random_state: int = 42):
        self.algorithm = algorithm
        self.resolution = resolution
        self.random_state = random_state

    def detect_communities(self, G: nx.Graph) -> dict[str, Any]:
        """Detect communities in the graph.

        Args:
            G: NetworkX graph

        Returns:
            Dictionary with community assignments and metrics

        """
        if G.number_of_nodes() < 3:
            return self._trivial_communities(G)

        communities = {}

        # Detect communities using specified algorithm
        if self.algorithm == "louvain":
            communities.update(self._louvain_communities(G))
        elif self.algorithm == "spectral":
            communities.update(self._spectral_communities(G))
        else:
            logger.warning(f"Unknown algorithm {self.algorithm}, using Louvain")
            communities.update(self._louvain_communities(G))

        # Compute modularity metrics
        if "labels" in communities:
            communities.update(self._compute_modularity_metrics(G, communities["labels"]))

        return communities

    def _trivial_communities(self, G: nx.Graph) -> dict[str, Any]:
        """Handle trivial cases with few nodes."""
        n_nodes = G.number_of_nodes()
        return {
            "labels": [0] * n_nodes,
            "n_communities": 1 if n_nodes > 0 else 0,
            "modularity": 0.0,
            "signed_modularity": 0.0,
            "community_sizes": [n_nodes] if n_nodes > 0 else [],
        }

    def _louvain_communities(self, G: nx.Graph) -> dict[str, Any]:
        """Detect communities using Louvain algorithm."""
        try:
            import community as community_louvain

            # Use absolute weights for Louvain if graph is signed
            working_graph = G
            if G.graph.get("signed", False):
                working_graph = self._create_absolute_weight_graph(G)

            partition = community_louvain.best_partition(
                working_graph, resolution=self.resolution, random_state=self.random_state
            )

            labels = [partition.get(i, 0) for i in range(G.number_of_nodes())]

            return {"labels": labels, "algorithm": "louvain"}

        except ImportError:
            logger.warning("python-louvain not available, falling back to spectral clustering")
            return self._spectral_communities(G)
        except Exception as e:
            logger.warning(f"Louvain algorithm failed: {e}")
            return self._spectral_communities(G)

    def _spectral_communities(self, G: nx.Graph) -> dict[str, Any]:
        """Detect communities using spectral clustering."""
        try:
            # Create adjacency matrix
            adj_matrix = nx.adjacency_matrix(G, weight="weight").toarray()

            if G.graph.get("signed", False):
                adj_matrix = np.abs(adj_matrix)  # Use absolute values

            # Determine number of clusters using eigengap heuristic
            n_clusters = self._estimate_n_clusters(adj_matrix)

            # Perform spectral clustering
            clustering = SpectralClustering(
                n_clusters=n_clusters, affinity="precomputed", random_state=self.random_state, eigen_solver="arpack"
            )

            labels = clustering.fit_predict(adj_matrix)

            return {"labels": labels.tolist(), "algorithm": "spectral", "n_clusters_used": n_clusters}

        except Exception as e:
            logger.warning(f"Spectral clustering failed: {e}")
            # Return single community as fallback
            return {"labels": [0] * G.number_of_nodes(), "algorithm": "fallback"}

    def _estimate_n_clusters(self, adj_matrix: np.ndarray) -> int:
        """Estimate number of clusters using eigengap heuristic."""
        try:
            eigenvals = np.linalg.eigvals(adj_matrix)
            eigenvals = np.sort(eigenvals)[::-1]

            if len(eigenvals) > 3:
                gaps = np.diff(eigenvals[: min(10, len(eigenvals))])
                n_clusters = np.argmax(gaps) + 2
            else:
                n_clusters = 2

            return max(2, min(n_clusters, adj_matrix.shape[0] - 1))

        except:
            return 2

    def _compute_modularity_metrics(self, G: nx.Graph, labels: list[int]) -> dict[str, Any]:
        """Compute various modularity metrics."""
        metrics = {}

        # Basic modularity
        try:
            communities_sets = self._labels_to_sets(labels)
            metrics["modularity"] = nx.community.modularity(G, communities_sets)
        except:
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

    def _labels_to_sets(self, labels: list[int]) -> list[set]:
        """Convert community labels to sets of nodes."""
        communities = defaultdict(set)
        for node, label in enumerate(labels):
            communities[label].add(node)
        return list(communities.values())

    def _compute_signed_modularity(self, G: nx.Graph, labels: list[int]) -> float:
        """Compute modularity for signed networks."""
        try:
            total_weight = sum(abs(d.get("weight", 1.0)) for _, _, d in G.edges(data=True))
            if total_weight == 0:
                return 0.0

            modularity = 0.0
            nodes = list(G.nodes())

            for i, node_i in enumerate(nodes):
                for j, node_j in enumerate(nodes):
                    if labels[i] == labels[j]:  # Same community
                        # Actual edge weight
                        A_ij = G[node_i][node_j].get("weight", 1.0) if G.has_edge(node_i, node_j) else 0.0

                        # Expected weight
                        ki = sum(abs(G[node_i][neighbor].get("weight", 1.0)) for neighbor in G.neighbors(node_i))
                        kj = sum(abs(G[node_j][neighbor].get("weight", 1.0)) for neighbor in G.neighbors(node_j))

                        expected = (ki * kj) / (2 * total_weight) if total_weight > 0 else 0.0
                        modularity += A_ij - expected

            return modularity / (2 * total_weight)

        except Exception as e:
            logger.warning(f"Error computing signed modularity: {e}")
            return 0.0

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

    def _create_absolute_weight_graph(self, G: nx.Graph) -> nx.Graph:
        """Create copy of graph with absolute edge weights."""
        G_abs = G.copy()
        for i, j, data in G_abs.edges(data=True):
            data["weight"] = abs(data.get("weight", 1.0))
        return G_abs


#######################################################
# Topology analyzer


class TopologyAnalyzer:
    """Handles graph topology and structural analysis."""

    def compute_topology_metrics(self, G: nx.Graph) -> dict[str, Any]:
        """Compute comprehensive topology metrics."""
        metrics = {}

        # Basic metrics
        metrics.update(self._compute_basic_metrics(G))

        # Path-based metrics
        metrics.update(self._compute_path_metrics(G))

        # Clustering metrics
        metrics.update(self._compute_clustering_metrics(G))

        # Efficiency metrics
        metrics.update(self._compute_efficiency_metrics(G))

        # Signed network metrics (if applicable)
        if G.graph.get("signed", False):
            metrics.update(self._compute_signed_metrics(G))

        return metrics

    def _compute_basic_metrics(self, G: nx.Graph) -> dict[str, Any]:
        """Compute basic graph metrics."""
        return {
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "density": nx.density(G),
            "is_connected": nx.is_connected(G),
            "n_components": nx.number_connected_components(G),
        }

    def _compute_path_metrics(self, G: nx.Graph) -> dict[str, Any]:
        """Compute path-based metrics."""
        metrics = {}

        # Use absolute weights for signed graphs
        working_graph = G
        if G.graph.get("signed", False) and self._has_negative_weights(G):
            working_graph = self._create_absolute_weight_graph(G)

        try:
            if nx.is_connected(working_graph):
                metrics["avg_path_length"] = nx.average_shortest_path_length(working_graph)
                metrics["diameter"] = nx.diameter(working_graph)
                metrics["radius"] = nx.radius(working_graph)
            else:
                # Use largest connected component
                largest_cc = max(nx.connected_components(working_graph), key=len)
                subgraph = working_graph.subgraph(largest_cc)

                if len(subgraph) > 1:
                    metrics["avg_path_length"] = nx.average_shortest_path_length(subgraph)
                    metrics["diameter"] = nx.diameter(subgraph)
                    metrics["radius"] = nx.radius(subgraph)
                else:
                    metrics["avg_path_length"] = 0.0
                    metrics["diameter"] = 0.0
                    metrics["radius"] = 0.0

        except Exception as e:
            logger.warning(f"Error computing path metrics: {e}")
            metrics.update({"avg_path_length": 0.0, "diameter": 0.0, "radius": 0.0})

        return metrics

    def _compute_clustering_metrics(self, G: nx.Graph) -> dict[str, Any]:
        """Compute clustering-related metrics."""
        try:
            # Use absolute weights for signed graphs
            working_graph = G
            if G.graph.get("signed", False) and self._has_negative_weights(G):
                working_graph = self._create_absolute_weight_graph(G)

            avg_clustering = nx.average_clustering(working_graph)
            transitivity = nx.transitivity(working_graph)

            return {"avg_clustering": float(avg_clustering), "transitivity": float(transitivity)}

        except Exception as e:
            logger.warning(f"Error computing clustering metrics: {e}")
            return {"avg_clustering": 0.0, "transitivity": 0.0}

    def _compute_efficiency_metrics(self, G: nx.Graph) -> dict[str, Any]:
        """Compute efficiency-related metrics."""
        try:
            working_graph = G
            if G.graph.get("signed", False) and self._has_negative_weights(G):
                working_graph = self._create_absolute_weight_graph(G)

            global_eff = nx.global_efficiency(working_graph)
            local_eff = nx.local_efficiency(working_graph)

            metrics = {"global_efficiency": float(global_eff), "local_efficiency": float(local_eff)}

            # Degree assortativity
            if working_graph.number_of_edges() > 0:
                try:
                    assortativity = nx.degree_assortativity_coefficient(working_graph)
                    metrics["degree_assortativity"] = float(assortativity)
                except:
                    metrics["degree_assortativity"] = 0.0
            else:
                metrics["degree_assortativity"] = 0.0

            return metrics

        except Exception as e:
            logger.warning(f"Error computing efficiency metrics: {e}")
            return {"global_efficiency": 0.0, "local_efficiency": 0.0, "degree_assortativity": 0.0}

    def _compute_signed_metrics(self, G: nx.Graph) -> dict[str, Any]:
        """Compute signed network specific metrics."""
        metrics = {}

        try:
            # Edge sign analysis
            pos_edges, neg_edges, balance_ratio = self._analyze_edge_signs(G)
            metrics.update(
                {"positive_edges": pos_edges, "negative_edges": neg_edges, "edge_balance_ratio": balance_ratio}
            )

            # Structural balance
            metrics["structural_balance"] = self._compute_structural_balance(G)

            # Frustration
            metrics["edge_frustration"] = self._compute_edge_frustration(G)

        except Exception as e:
            logger.warning(f"Error computing signed metrics: {e}")
            metrics.update(
                {
                    "positive_edges": 0,
                    "negative_edges": 0,
                    "edge_balance_ratio": 0.5,
                    "structural_balance": 1.0,
                    "edge_frustration": 0.0,
                }
            )

        return metrics

    def _analyze_edge_signs(self, G: nx.Graph) -> tuple[int, int, float]:
        """Analyze distribution of positive and negative edges."""
        pos_edges = 0
        neg_edges = 0

        for _, _, data in G.edges(data=True):
            weight = data.get("weight", 1.0)
            if weight > 0:
                pos_edges += 1
            elif weight < 0:
                neg_edges += 1

        total_edges = pos_edges + neg_edges
        balance_ratio = pos_edges / total_edges if total_edges > 0 else 0.5

        return pos_edges, neg_edges, balance_ratio

    def _compute_structural_balance(self, G: nx.Graph) -> float:
        """Compute structural balance based on triangle signs."""
        try:
            balanced_triangles = 0
            total_triangles = 0

            # Check all triangles
            for triangle in nx.enumerate_all_cliques(G):
                if len(triangle) == 3:
                    total_triangles += 1
                    i, j, k = triangle

                    # Get edge signs
                    sign_ij = np.sign(G[i][j].get("weight", 1.0)) if G.has_edge(i, j) else 0
                    sign_jk = np.sign(G[j][k].get("weight", 1.0)) if G.has_edge(j, k) else 0
                    sign_ik = np.sign(G[i][k].get("weight", 1.0)) if G.has_edge(i, k) else 0

                    # Triangle is balanced if it has even number of negative edges
                    negative_edges = sum(1 for sign in [sign_ij, sign_jk, sign_ik] if sign < 0)
                    if negative_edges % 2 == 0:
                        balanced_triangles += 1

            return balanced_triangles / total_triangles if total_triangles > 0 else 1.0

        except Exception as e:
            logger.warning(f"Error computing structural balance: {e}")
            return 1.0

    def _compute_edge_frustration(self, G: nx.Graph) -> float:
        """Compute edge frustration metric."""
        try:
            frustrated_edges = 0
            total_triangles = 0

            for triangle in nx.enumerate_all_cliques(G):
                if len(triangle) == 3:
                    total_triangles += 1
                    i, j, k = triangle

                    # Get edge weights
                    w_ij = G[i][j].get("weight", 1.0) if G.has_edge(i, j) else 0
                    w_jk = G[j][k].get("weight", 1.0) if G.has_edge(j, k) else 0
                    w_ik = G[i][k].get("weight", 1.0) if G.has_edge(i, k) else 0

                    # Check if triangle is frustrated
                    sign_product = np.sign(w_ij) * np.sign(w_jk) * np.sign(w_ik)
                    if sign_product < 0:
                        frustrated_edges += 1

            return frustrated_edges / total_triangles if total_triangles > 0 else 0.0

        except Exception as e:
            logger.warning(f"Error computing edge frustration: {e}")
            return 0.0

    def _has_negative_weights(self, G: nx.Graph) -> bool:
        """Check if graph has negative weights."""
        return any(data.get("weight", 1.0) < 0 for _, _, data in G.edges(data=True))

    def _create_absolute_weight_graph(self, G: nx.Graph) -> nx.Graph:
        """Create copy with absolute weights."""
        G_abs = G.copy()
        for i, j, data in G_abs.edges(data=True):
            data["weight"] = abs(data.get("weight", 1.0))
        return G_abs


#######################################################
# Network analyzer


class NetworkAnalyzer:
    """Main network analysis class that orchestrates different analyzers.

    Coordinates centrality, community, and topology analysis for comprehensive
    network characterization.
    """

    def __init__(self, centrality_config: dict[str, Any] | None = None, community_config: dict[str, Any] | None = None):
        """Initialize NetworkAnalyzer with component analyzers.

        Args:
            centrality_config: Configuration for centrality analysis
            community_config: Configuration for community detection

        """
        # Initialize component analyzers
        self.centrality_analyzer = CentralityAnalyzer(**(centrality_config or {}))
        self.community_analyzer = CommunityAnalyzer(**(community_config or {}))
        self.topology_analyzer = TopologyAnalyzer()

    def analyze_network(self, G: nx.Graph) -> dict[str, Any]:
        """Perform comprehensive network analysis.

        Args:
            G: NetworkX graph to analyze

        Returns:
            Dictionary containing all analysis results

        """
        if G.number_of_nodes() == 0:
            logger.warning("Cannot analyze empty graph")
            return self._empty_analysis_result()

        results = {}

        # Centrality analysis
        logger.debug("Computing centralities...")
        results["centralities"] = self.centrality_analyzer.compute_all_centralities(G)

        # Community detection
        logger.debug("Detecting communities...")
        results["communities"] = self.community_analyzer.detect_communities(G)

        # Topology analysis
        logger.debug("Analyzing topology...")
        results["topology"] = self.topology_analyzer.compute_topology_metrics(G)

        # Hub analysis
        results["hubs"] = self._analyze_hubs(results["centralities"], G)

        # Summary statistics
        results["summary"] = self._compute_summary_stats(results, G)

        return results

    def analyze_multiple_networks(self, graphs: dict[str, nx.Graph]) -> dict[str, dict[str, Any]]:
        """Analyze multiple networks and return results for each."""
        results = {}

        for name, graph in graphs.items():
            logger.debug(f"Analyzing network: {name}")
            try:
                results[name] = self.analyze_network(graph)
            except Exception as e:
                logger.warning(f"Analysis failed for {name}: {e}")
                results[name] = self._empty_analysis_result()

        return results

    def compare_networks(
        self, graph1: nx.Graph, graph2: nx.Graph, name1: str = "graph1", name2: str = "graph2"
    ) -> dict[str, Any]:
        """Compare two networks across multiple dimensions."""
        # Analyze both networks
        results1 = self.analyze_network(graph1)
        results2 = self.analyze_network(graph2)

        comparison = {
            "networks": {name1: results1, name2: results2},
            "differences": self._compute_network_differences(results1, results2),
            "similarities": self._compute_network_similarities(results1, results2),
            "summary": {
                "more_hierarchical": self._compare_hierarchy(results1, results2, name1, name2),
                "more_modular": self._compare_modularity(results1, results2, name1, name2),
                "more_efficient": self._compare_efficiency(results1, results2, name1, name2),
                "more_balanced": self._compare_balance(results1, results2, name1, name2),
            },
        }

        return comparison

    def _analyze_hubs(self, centralities: dict[str, list[float]], G: nx.Graph) -> dict[str, Any]:
        """Analyze hub nodes based on centrality measures."""
        if not centralities.get("degree"):
            return {"hub_nodes": [], "n_hubs": 0, "hub_strength": 0.0}

        degree_cents = centralities["degree"]
        betweenness_cents = centralities.get("betweenness", [0.0] * len(degree_cents))

        # Define hubs as top 10% by degree centrality
        hub_threshold = np.percentile(degree_cents, 90) if degree_cents else 0
        hub_nodes = [i for i, dc in enumerate(degree_cents) if dc >= hub_threshold]

        # Hub analysis for signed networks
        if G.graph.get("signed", False):
            return self._analyze_signed_hubs(centralities, hub_nodes)
        return self._analyze_unsigned_hubs(centralities, hub_nodes)

    def _analyze_signed_hubs(self, centralities: dict[str, list[float]], hub_nodes: list[int]) -> dict[str, Any]:
        """Analyze hubs in signed networks."""
        pos_degrees = centralities.get("positive_degree", [])
        neg_degrees = centralities.get("negative_degree", [])
        balance_scores = centralities.get("signed_balance", [])

        if not pos_degrees or not neg_degrees:
            return self._analyze_unsigned_hubs(centralities, hub_nodes)

        # Classify hubs by their positive/negative degree balance
        excitatory_hubs = []
        inhibitory_hubs = []
        balanced_hubs = []

        for hub in hub_nodes:
            if hub < len(balance_scores):
                balance = balance_scores[hub]
                if balance > 0.7:
                    excitatory_hubs.append(hub)
                elif balance < 0.3:
                    inhibitory_hubs.append(hub)
                else:
                    balanced_hubs.append(hub)

        return {
            "hub_nodes": hub_nodes,
            "n_hubs": len(hub_nodes),
            "excitatory_hubs": excitatory_hubs,
            "inhibitory_hubs": inhibitory_hubs,
            "balanced_hubs": balanced_hubs,
            "hub_balance_ratio": len(excitatory_hubs) / len(hub_nodes) if hub_nodes else 0.5,
            "hub_diversity": len(
                set(
                    [
                        "excitatory" if h in excitatory_hubs else "inhibitory" if h in inhibitory_hubs else "balanced"
                        for h in hub_nodes
                    ]
                )
            ),
        }

    def _analyze_unsigned_hubs(self, centralities: dict[str, list[float]], hub_nodes: list[int]) -> dict[str, Any]:
        """Analyze hubs in unsigned networks."""
        betweenness_cents = centralities.get("betweenness", [])

        hub_betweenness = np.mean([betweenness_cents[i] for i in hub_nodes]) if hub_nodes and betweenness_cents else 0.0

        return {
            "hub_nodes": hub_nodes,
            "n_hubs": len(hub_nodes),
            "hub_betweenness_centrality": float(hub_betweenness),
            "hub_strength": float(np.mean([centralities["degree"][i] for i in hub_nodes])) if hub_nodes else 0.0,
        }

    def _compute_summary_stats(self, results: dict[str, Any], G: nx.Graph) -> dict[str, Any]:
        """Compute summary statistics across all analyses."""
        summary = {
            "graph_type": "signed" if G.graph.get("signed", False) else "unsigned",
            "analysis_quality": self._assess_analysis_quality(results, G),
            "key_properties": self._identify_key_properties(results),
            "complexity_score": self._compute_complexity_score(results),
        }

        return summary

    def _assess_analysis_quality(self, results: dict[str, Any], G: nx.Graph) -> dict[str, Any]:
        """Assess the quality and reliability of the analysis."""
        quality = {
            "sufficient_data": G.number_of_nodes() >= 10 and G.number_of_edges() >= 10,
            "connected": results.get("topology", {}).get("is_connected", False),
            "non_trivial_communities": results.get("communities", {}).get("n_communities", 1) > 1,
            "meaningful_centralities": max(results.get("centralities", {}).get("degree", [0])) > 0.1,
        }

        quality["overall_score"] = sum(quality.values()) / len(quality)
        return quality

    def _identify_key_properties(self, results: dict[str, Any]) -> list[str]:
        """Identify key structural properties of the network."""
        properties = []

        # Check for hierarchy
        centralities = results.get("centralities", {})
        if centralities.get("betweenness") and max(centralities["betweenness"]) > 0.1:
            properties.append("hierarchical")

        # Check for modularity
        communities = results.get("communities", {})
        if communities.get("modularity", 0) > 0.3:
            properties.append("modular")

        # Check for small-world properties
        topology = results.get("topology", {})
        avg_clustering = topology.get("avg_clustering", 0)
        if avg_clustering > 0.3:
            properties.append("clustered")

        # Check for efficiency
        if topology.get("global_efficiency", 0) > 0.5:
            properties.append("efficient")

        # Check for balance (signed networks)
        if topology.get("structural_balance", 1.0) > 0.8:
            properties.append("balanced")
        elif topology.get("edge_frustration", 0) > 0.3:
            properties.append("frustrated")

        return properties

    def _compute_complexity_score(self, results: dict[str, Any]) -> float:
        """Compute overall network complexity score."""
        try:
            # Combine multiple complexity indicators
            topology = results.get("topology", {})
            communities = results.get("communities", {})

            # Modularity contributes to complexity
            modularity_score = min(communities.get("modularity", 0), 1.0)

            # Number of communities (normalized)
            n_communities = communities.get("n_communities", 1)
            n_nodes = topology.get("n_nodes", 1)
            community_score = min(n_communities / n_nodes, 0.5) * 2  # Cap at 0.5, then scale to 1

            # Clustering contributes to complexity
            clustering_score = min(topology.get("avg_clustering", 0), 1.0)

            # Hub diversity (for signed networks)
            hubs = results.get("hubs", {})
            hub_score = min(hubs.get("hub_diversity", 1) / 3, 1.0)  # Max 3 types of hubs

            # Weighted combination
            complexity = 0.3 * modularity_score + 0.3 * clustering_score + 0.2 * community_score + 0.2 * hub_score

            return float(complexity)

        except Exception as e:
            logger.warning(f"Error computing complexity score: {e}")
            return 0.0

    def _compute_network_differences(self, results1: dict[str, Any], results2: dict[str, Any]) -> dict[str, Any]:
        """Compute differences between two network analyses."""
        differences = {}

        # Topology differences
        topo1 = results1.get("topology", {})
        topo2 = results2.get("topology", {})

        topology_diffs = {}
        for metric in ["density", "avg_clustering", "global_efficiency", "avg_path_length"]:
            val1 = topo1.get(metric, 0)
            val2 = topo2.get(metric, 0)
            topology_diffs[f"{metric}_difference"] = val2 - val1
            if val1 != 0:
                topology_diffs[f"{metric}_relative_change"] = (val2 - val1) / abs(val1)

        differences["topology"] = topology_diffs

        # Community differences
        comm1 = results1.get("communities", {})
        comm2 = results2.get("communities", {})

        differences["communities"] = {
            "modularity_difference": comm2.get("modularity", 0) - comm1.get("modularity", 0),
            "n_communities_difference": comm2.get("n_communities", 1) - comm1.get("n_communities", 1),
        }

        # Hub differences
        hubs1 = results1.get("hubs", {})
        hubs2 = results2.get("hubs", {})

        differences["hubs"] = {
            "n_hubs_difference": hubs2.get("n_hubs", 0) - hubs1.get("n_hubs", 0),
            "hub_strength_difference": hubs2.get("hub_strength", 0) - hubs1.get("hub_strength", 0),
        }

        return differences

    def _compute_network_similarities(self, results1: dict[str, Any], results2: dict[str, Any]) -> dict[str, Any]:
        """Compute similarities between two network analyses."""
        similarities = {}

        # Structural similarity
        topo1 = results1.get("topology", {})
        topo2 = results2.get("topology", {})

        # Compute correlation between centrality distributions
        cent1 = results1.get("centralities", {})
        cent2 = results2.get("centralities", {})

        centrality_correlations = {}
        for measure in ["degree", "betweenness", "closeness"]:
            if measure in cent1 and measure in cent2:
                try:
                    corr = np.corrcoef(cent1[measure], cent2[measure])[0, 1]
                    centrality_correlations[f"{measure}_correlation"] = float(corr) if not np.isnan(corr) else 0.0
                except:
                    centrality_correlations[f"{measure}_correlation"] = 0.0

        similarities["centrality_correlations"] = centrality_correlations

        # Property overlap
        props1 = set(results1.get("summary", {}).get("key_properties", []))
        props2 = set(results2.get("summary", {}).get("key_properties", []))

        jaccard_similarity = len(props1 & props2) / len(props1 | props2) if props1 or props2 else 1.0

        similarities["property_similarity"] = jaccard_similarity

        return similarities

    def _compare_hierarchy(self, results1: dict[str, Any], results2: dict[str, Any], name1: str, name2: str) -> str:
        """Compare hierarchy between two networks."""
        hubs1 = results1.get("hubs", {}).get("n_hubs", 0)
        hubs2 = results2.get("hubs", {}).get("n_hubs", 0)

        betweenness1 = max(results1.get("centralities", {}).get("betweenness", [0]))
        betweenness2 = max(results2.get("centralities", {}).get("betweenness", [0]))

        if hubs2 > hubs1 and betweenness2 > betweenness1:
            return name2
        if hubs1 > hubs2 and betweenness1 > betweenness2:
            return name1
        return "similar"

    def _compare_modularity(self, results1: dict[str, Any], results2: dict[str, Any], name1: str, name2: str) -> str:
        """Compare modularity between two networks."""
        mod1 = results1.get("communities", {}).get("modularity", 0)
        mod2 = results2.get("communities", {}).get("modularity", 0)

        if mod2 > mod1 + 0.1:
            return name2
        if mod1 > mod2 + 0.1:
            return name1
        return "similar"

    def _compare_efficiency(self, results1: dict[str, Any], results2: dict[str, Any], name1: str, name2: str) -> str:
        """Compare efficiency between two networks."""
        eff1 = results1.get("topology", {}).get("global_efficiency", 0)
        eff2 = results2.get("topology", {}).get("global_efficiency", 0)

        if eff2 > eff1 + 0.1:
            return name2
        if eff1 > eff2 + 0.1:
            return name1
        return "similar"

    def _compare_balance(self, results1: dict[str, Any], results2: dict[str, Any], name1: str, name2: str) -> str:
        """Compare balance between two signed networks."""
        balance1 = results1.get("topology", {}).get("structural_balance", 1.0)
        balance2 = results2.get("topology", {}).get("structural_balance", 1.0)

        if balance2 > balance1 + 0.1:
            return name2
        if balance1 > balance2 + 0.1:
            return name1
        return "similar"

    def _empty_analysis_result(self) -> dict[str, Any]:
        """Return empty analysis result for invalid graphs."""
        return {
            "centralities": {},
            "communities": {"n_communities": 0, "modularity": 0.0},
            "topology": {"n_nodes": 0, "n_edges": 0, "density": 0.0},
            "hubs": {"n_hubs": 0},
            "summary": {"key_properties": [], "complexity_score": 0.0},
        }

    def get_analysis_summary(self, results: dict[str, Any]) -> str:
        """Generate human-readable summary of network analysis."""
        if not results or results.get("topology", {}).get("n_nodes", 0) == 0:
            return "Empty or invalid network - no analysis possible."

        topology = results.get("topology", {})
        communities = results.get("communities", {})
        hubs = results.get("hubs", {})
        summary = results.get("summary", {})

        n_nodes = topology.get("n_nodes", 0)
        n_edges = topology.get("n_edges", 0)
        density = topology.get("density", 0)
        modularity = communities.get("modularity", 0)
        n_communities = communities.get("n_communities", 1)
        n_hubs = hubs.get("n_hubs", 0)
        key_props = summary.get("key_properties", [])

        summary_text = f"""
        Network Analysis Summary:
        - Structure: {n_nodes} nodes, {n_edges} edges (density: {density:.3f})
        - Communities: {n_communities} communities (modularity: {modularity:.3f})
        - Hubs: {n_hubs} hub nodes identified
        - Key properties: {", ".join(key_props) if key_props else "basic connectivity"}
        - Complexity score: {summary.get("complexity_score", 0):.3f}
        """

        if topology.get("structural_balance"):
            balance = topology.get("structural_balance", 1.0)
            frustration = topology.get("edge_frustration", 0.0)
            summary_text += f"- Signed network: {balance:.3f} balance, {frustration:.3f} frustration\n"

        return summary_text.strip()
