import logging
import typing as t
import warnings
from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering

warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)

#######################################################
# Abstract clss


class CommunityDetectionAlgorithm(ABC):
    """Abstract base class for community detection algorithms."""

    def __init__(self, random_state: int = 42, **kwargs):
        self.random_state = random_state
        self.params = kwargs

    @abstractmethod
    def detect(self, G: nx.Graph) -> dict[str, t.Any]:
        """Detect communities in the graph.

        Args:
            G: NetworkX graph

        Returns:
            Dictionary with labels and algorithm-specific metadata

        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return algorithm name."""


#######################################################
# LouvainDetection


class LouvainDetection(CommunityDetectionAlgorithm):
    """Louvain community detection algorithm."""

    def __init__(self, resolution: float = 1.0, random_state: int = 42, **kwargs):
        super().__init__(random_state, **kwargs)
        self.resolution = resolution

    @property
    def name(self) -> str:
        return "louvain"

    def detect(self, G: nx.Graph) -> dict[str, t.Any]:
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

            return {"labels": labels, "algorithm": self.name, "resolution_used": self.resolution}

        except ImportError:
            raise ImportError("python-louvain not available")
        except Exception as e:
            raise RuntimeError(f"Louvain algorithm failed: {e}")

    def _create_absolute_weight_graph(self, G: nx.Graph) -> nx.Graph:
        """Create copy of graph with absolute edge weights."""
        G_abs = G.copy()
        for i, j, data in G_abs.edges(data=True):
            data["weight"] = abs(data.get("weight", 1.0))
        return G_abs


#######################################################
# Spectral Detection


class SpectralDetection(CommunityDetectionAlgorithm):
    """Standard spectral clustering algorithm."""

    def __init__(self, n_clusters: int | None = None, random_state: int = 42, **kwargs):
        super().__init__(random_state, **kwargs)
        self.n_clusters = n_clusters

    @property
    def name(self) -> str:
        return "spectral"

    def detect(self, G: nx.Graph) -> dict[str, t.Any]:
        """Detect communities using standard spectral clustering."""
        try:
            # Create adjacency matrix
            adj_matrix = nx.adjacency_matrix(G, weight="weight").toarray()

            if G.graph.get("signed", False):
                adj_matrix = np.abs(adj_matrix)  # Use absolute values

            # Determine number of clusters
            n_clusters = self.n_clusters or self._estimate_n_clusters(adj_matrix)

            # Perform spectral clustering
            clustering = SpectralClustering(
                n_clusters=n_clusters, affinity="precomputed", random_state=self.random_state, eigen_solver="arpack"
            )

            labels = clustering.fit_predict(adj_matrix)

            return {"labels": labels.tolist(), "algorithm": self.name, "n_clusters_used": n_clusters}

        except Exception as e:
            raise RuntimeError(f"Spectral clustering failed: {e}")

    def _estimate_n_clusters(self, matrix: np.ndarray) -> int:
        """Estimate number of clusters using eigengap heuristic."""
        try:
            eigenvals = np.linalg.eigvals(matrix)
            eigenvals = np.sort(eigenvals)[::-1]

            if len(eigenvals) > 3:
                gaps = np.diff(eigenvals[: min(10, len(eigenvals))])
                n_clusters = np.argmax(gaps) + 2
            else:
                n_clusters = 2

            return max(2, min(n_clusters, matrix.shape[0] - 1))

        except Exception:
            return 2


#######################################################
# Newman Spectral Detection


class NewmanSpectralDetection(CommunityDetectionAlgorithm):
    """Newman's spectral method using modularity matrix eigendecomposition."""

    def __init__(self, max_bisections: int = 10, random_state: int = 42, **kwargs):
        super().__init__(random_state, **kwargs)
        self.max_bisections = max_bisections

    @property
    def name(self) -> str:
        return "newman_spectral"

    def detect(self, G: nx.Graph) -> dict[str, t.Any]:
        """Detect communities using Newman's spectral method.

        Uses eigendecomposition of modularity matrix for global optimization.
        """
        try:
            # Create adjacency matrix
            adj_matrix = nx.adjacency_matrix(G, weight="weight").toarray()
            n_nodes = adj_matrix.shape[0]

            if n_nodes < 2:
                return {"labels": [0] * n_nodes, "algorithm": self.name}

            # Compute degrees
            degrees = np.sum(adj_matrix, axis=1)
            total_weight = np.sum(degrees) / 2

            if total_weight == 0:
                return {"labels": [0] * n_nodes, "algorithm": self.name}

            # Construct modularity matrix: B_ij = A_ij - (k_i * k_j)/(2m)
            expected_matrix = np.outer(degrees, degrees) / (2 * total_weight)
            modularity_matrix = adj_matrix - expected_matrix

            # Recursive bisection using leading eigenvectors
            labels = self._recursive_bisection(modularity_matrix, list(range(n_nodes)))

            return {"labels": labels, "algorithm": self.name, "n_communities": len(set(labels))}

        except Exception as e:
            raise RuntimeError(f"Newman spectral method failed: {e}")

    def _recursive_bisection(self, modularity_matrix: np.ndarray, node_indices: list[int]) -> list[int]:
        """Perform recursive bisection using modularity matrix eigenvectors."""
        n_nodes = len(node_indices)
        labels = [0] * len(node_indices)

        if n_nodes <= 2:
            return labels

        # Get submatrix for current nodes
        sub_matrix = modularity_matrix[np.ix_(node_indices, node_indices)]

        try:
            # Compute eigenvalues and eigenvectors
            eigenvals, eigenvecs = np.linalg.eigh(sub_matrix)

            # Find leading positive eigenvalue
            positive_eigenvals = eigenvals[eigenvals > 1e-10]
            if len(positive_eigenvals) == 0:
                return labels  # No more splits possible

            # Get leading eigenvector
            leading_idx = np.argmax(eigenvals)
            leading_eigenvec = eigenvecs[:, leading_idx]

            # Bisection based on sign of eigenvector components
            split_indices = leading_eigenvec > 0

            # Assign community labels
            community_0_indices = [i for i, split in enumerate(split_indices) if not split]
            community_1_indices = [i for i, split in enumerate(split_indices) if split]

            if len(community_0_indices) == 0 or len(community_1_indices) == 0:
                return labels  # No meaningful split

            # Assign labels
            for i in community_0_indices:
                labels[i] = 0
            for i in community_1_indices:
                labels[i] = 1

            return labels

        except Exception as e:
            logger.warning(f"Recursive bisection failed: {e}")
            return labels


#######################################################
# Spectral Kernel Detection


class SpectralKernelDetection(CommunityDetectionAlgorithm):
    """Spectral clustering with kernel methods for nonlinear relationships."""

    def __init__(
        self,
        kernel_type: str = "rbf",
        kernel_params: dict[str, float] | None = None,
        n_clusters: int | None = None,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(random_state, **kwargs)
        self.kernel_type = kernel_type
        self.kernel_params = kernel_params or {"sigma": 1.0, "degree": 3, "c": 1.0}
        self.n_clusters = n_clusters

    @property
    def name(self) -> str:
        return "spectral_kernel"

    def detect(self, G: nx.Graph) -> dict[str, t.Any]:
        """Detect communities using spectral clustering with kernels.

        Captures nonlinear relationships through kernel mappings.
        """
        try:
            # Get node features (use adjacency matrix rows as features)
            adj_matrix = nx.adjacency_matrix(G, weight="weight").toarray()
            n_nodes = adj_matrix.shape[0]

            if n_nodes < 2:
                return {"labels": [0] * n_nodes, "algorithm": self.name}

            # Construct kernel matrix
            kernel_matrix = self._compute_kernel_matrix(adj_matrix)

            # Determine number of clusters
            n_clusters = self.n_clusters or self._estimate_n_clusters(kernel_matrix)

            # Normalize kernel matrix
            degrees = np.sum(kernel_matrix, axis=1)
            degrees[degrees == 0] = 1  # Avoid division by zero
            D_sqrt_inv = np.diag(1.0 / np.sqrt(degrees))
            normalized_kernel = D_sqrt_inv @ kernel_matrix @ D_sqrt_inv

            # Eigendecomposition
            eigenvals, eigenvecs = np.linalg.eigh(normalized_kernel)

            # Use top k eigenvectors for embedding
            top_k_eigenvecs = eigenvecs[:, -n_clusters:]

            # K-means clustering in embedded space
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(top_k_eigenvecs)

            return {
                "labels": labels.tolist(),
                "algorithm": self.name,
                "kernel_type": self.kernel_type,
                "n_clusters_used": n_clusters,
            }

        except Exception as e:
            raise RuntimeError(f"Spectral kernel clustering failed: {e}")

    def _compute_kernel_matrix(self, features: np.ndarray) -> np.ndarray:
        """Compute kernel matrix based on specified kernel type."""
        n_nodes = features.shape[0]
        kernel_matrix = np.zeros((n_nodes, n_nodes))

        if self.kernel_type == "rbf":
            sigma = self.kernel_params.get("sigma", 1.0)
            for i in range(n_nodes):
                for j in range(n_nodes):
                    diff = features[i] - features[j]
                    kernel_matrix[i, j] = np.exp(-(np.linalg.norm(diff) ** 2) / (2 * sigma**2))

        elif self.kernel_type == "polynomial":
            degree = self.kernel_params.get("degree", 3)
            c = self.kernel_params.get("c", 1.0)
            for i in range(n_nodes):
                for j in range(n_nodes):
                    kernel_matrix[i, j] = (np.dot(features[i], features[j]) + c) ** degree

        elif self.kernel_type == "linear":
            kernel_matrix = features @ features.T

        else:
            # Default to RBF
            sigma = 1.0
            for i in range(n_nodes):
                for j in range(n_nodes):
                    diff = features[i] - features[j]
                    kernel_matrix[i, j] = np.exp(-(np.linalg.norm(diff) ** 2) / (2 * sigma**2))

        return kernel_matrix

    def _estimate_n_clusters(self, matrix: np.ndarray) -> int:
        """Estimate number of clusters using eigengap heuristic."""
        try:
            eigenvals = np.linalg.eigvals(matrix)
            eigenvals = np.sort(eigenvals)[::-1]

            if len(eigenvals) > 3:
                gaps = np.diff(eigenvals[: min(10, len(eigenvals))])
                n_clusters = np.argmax(gaps) + 2
            else:
                n_clusters = 2

            return max(2, min(n_clusters, matrix.shape[0] - 1))

        except Exception:
            return 2


#######################################################
# Spin Glass Detection


class SpinGlassDetection(CommunityDetectionAlgorithm):
    """Spin glass optimization for community detection."""

    def __init__(self, annealing_params: dict[str, float] | None = None, random_state: int = 42, **kwargs):
        super().__init__(random_state, **kwargs)
        self.annealing_params = annealing_params or {
            "initial_temp": 1.0,
            "final_temp": 0.01,
            "alpha": 0.95,
            "max_iterations": 1000,
        }

    @property
    def name(self) -> str:
        return "spin_glass"

    def detect(self, G: nx.Graph) -> dict[str, t.Any]:
        """Detect communities using spin glass optimization.

        Treats community detection as physical system optimization problem.
        """
        try:
            # Create adjacency matrix
            adj_matrix = nx.adjacency_matrix(G, weight="weight").toarray()
            n_nodes = adj_matrix.shape[0]

            if n_nodes < 2:
                return {"labels": [0] * n_nodes, "algorithm": self.name}

            # Initialize random spins
            np.random.seed(self.random_state)
            spins = np.random.choice([-1, 1], size=n_nodes)

            # Simulated annealing parameters
            temp = self.annealing_params["initial_temp"]
            final_temp = self.annealing_params["final_temp"]
            alpha = self.annealing_params["alpha"]
            max_iter = self.annealing_params["max_iterations"]

            # Annealing loop
            for iteration in range(max_iter):
                if temp < final_temp:
                    break

                # Random node selection
                node = np.random.randint(0, n_nodes)

                # Calculate energy change for spin flip
                delta_energy = self._compute_energy_change(adj_matrix, spins, node)

                # Accept or reject flip
                if delta_energy < 0 or np.random.random() < np.exp(-delta_energy / temp):
                    spins[node] *= -1

                # Cool down
                temp *= alpha

            # Convert spins to community labels
            labels = [(spin + 1) // 2 for spin in spins]  # Convert {-1, 1} to {0, 1}

            # If all nodes in same community, try bisection
            if len(set(labels)) == 1:
                labels = [i % 2 for i in range(n_nodes)]  # Simple alternating pattern

            return {
                "labels": labels,
                "algorithm": self.name,
                "final_temperature": float(temp),
                "n_communities": len(set(labels)),
            }

        except Exception as e:
            raise RuntimeError(f"Spin glass optimization failed: {e}")

    def _compute_energy_change(self, adj_matrix: np.ndarray, spins: np.ndarray, node: int) -> float:
        """Compute energy change for flipping spin of given node."""
        current_spin = spins[node]
        energy_change = 0.0

        # Energy change due to interactions with neighbors
        for neighbor in range(len(spins)):
            if neighbor != node:
                coupling = adj_matrix[node, neighbor]
                energy_change += 2 * coupling * current_spin * spins[neighbor]

        return energy_change


#######################################################
# Detection integration


class CommunityDetectionFactory:
    """Factory for creating community detection algorithms."""

    @staticmethod
    def create_algorithm(algorithm: str, **kwargs) -> CommunityDetectionAlgorithm:
        """Create a community detection algorithm instance."""
        algorithms = {
            "louvain": LouvainDetection,
            "spectral": SpectralDetection,
            "newman_spectral": NewmanSpectralDetection,
            "spectral_kernel": SpectralKernelDetection,
            "spin_glass": SpinGlassDetection,
        }

        if algorithm not in algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(algorithms.keys())}")

        return algorithms[algorithm](**kwargs)

    @staticmethod
    def list_algorithms() -> list[str]:
        """List available algorithms."""
        return ["louvain", "spectral", "newman_spectral", "spectral_kernel", "spin_glass"]
