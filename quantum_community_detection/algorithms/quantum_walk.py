"""
Algorithm 2: Quantum Walk Community Detection
=============================================
Uses continuous-time quantum walks (CTQW) on the graph Laplacian.

Key insight: Quantum walks exhibit interference that localises within
communities. Nodes in the same community have high quantum walk overlap
probability (|⟨i| e^{-iLt} |j⟩|²), while inter-community pairs have low overlap.

The algorithm:
  1. Encode graph Laplacian L as walk Hamiltonian
  2. Evolve |v⟩ → e^{-iLt} |v⟩ for each starting node v
  3. Compute pairwise quantum walk similarity matrix S_{ij}
  4. Cluster S with spectral or threshold method

References:
  - Childs et al. (2002). Exponential algorithmic speedup by quantum walk.
  - Mukai & Fujii (2020). Classifying time series using quantum walks.
  - Dernbach et al. (2019). Quantum Walk Neural Networks.
"""

import numpy as np
from scipy.linalg import expm
from typing import List, Optional, Dict, Any

from ..utils.quantum_sim import QuantumState, kron_n, I2


class QuantumWalkCommunityDetection:
    """
    Continuous-time quantum walk (CTQW) community detection.

    The walk Hamiltonian is the normalized graph Laplacian L.
    Quantum walk interference creates a natural similarity metric
    that respects community structure.

    Parameters
    ----------
    t_values : list of float
        Evolution times to average over (reduces time-dependency).
    n_clusters : int or None
        Number of communities. If None, estimated from eigengap.
    """

    def __init__(self, t_values: Optional[List[float]] = None,
                 n_clusters: Optional[int] = None):
        self.t_values = t_values if t_values is not None else [
            0.5, 1.0, 2.0, 3.0, 5.0
        ]
        # Accept positional int as n_clusters (handle misuse gracefully)
        if isinstance(t_values, int):
            self.n_clusters = t_values
            self.t_values = [0.5, 1.0, 2.0, 3.0, 5.0]
        else:
            self.n_clusters = n_clusters
        self.similarity_matrix_: Optional[np.ndarray] = None

    # ── Quantum walk core ───────────────────────────────────────────────────

    def _graph_laplacian(self, adj: np.ndarray) -> np.ndarray:
        """Normalized graph Laplacian L = D - A."""
        degree = np.sum(adj, axis=1)
        L = np.diag(degree) - adj
        return L.astype(complex)

    def _walk_operator(self, L: np.ndarray, t: float) -> np.ndarray:
        """Continuous-time quantum walk operator U(t) = e^{-iLt}."""
        return expm(-1j * t * L)

    def _node_state(self, node: int, n: int) -> np.ndarray:
        """Computational basis state |node⟩ as column vector."""
        state = np.zeros(n, dtype=complex)
        state[node] = 1.0
        return state

    def _quantum_walk_similarity(self, adj: np.ndarray) -> np.ndarray:
        """
        Compute QW similarity matrix averaged over multiple evolution times.

        S_{ij} = (1/|T|) * sum_t |⟨j| U(t) |i⟩|²

        High S_{ij} → i and j are in the same community.
        """
        n = adj.shape[0]
        L = self._graph_laplacian(adj)
        S = np.zeros((n, n), dtype=float)

        for t in self.t_values:
            U_t = self._walk_operator(L, t)
            # |U_t[j,i]|^2 = probability of walk from i reaching j
            S += np.abs(U_t) ** 2

        S /= len(self.t_values)
        return S

    def _quantum_interference_matrix(self, adj: np.ndarray) -> np.ndarray:
        """
        Alternative: quantum interference similarity using
        coherences between walk amplitudes.

        S^{int}_{ij} = |sum_t ⟨j|U(t)|i⟩|^2 / |T|^2

        Captures phase information, sharper community boundaries.
        """
        n = adj.shape[0]
        L = self._graph_laplacian(adj)
        amp_sum = np.zeros((n, n), dtype=complex)

        for t in self.t_values:
            U_t = self._walk_operator(L, t)
            amp_sum += U_t

        S = np.abs(amp_sum) ** 2 / len(self.t_values) ** 2
        return S

    # ── Spectral clustering on similarity matrix ────────────────────────────

    def _estimate_n_clusters(self, S: np.ndarray) -> int:
        """Estimate number of clusters from eigengap of similarity Laplacian."""
        # Compute Laplacian of similarity matrix
        row_sums = np.sum(S, axis=1)
        D = np.diag(row_sums)
        L_S = D - S
        eigenvalues = np.sort(np.linalg.eigvalsh(L_S))
        # Eigengap heuristic
        gaps = np.diff(eigenvalues[1:])  # Skip zero eigenvalue
        k = np.argmax(gaps) + 2  # +2 for offset and 1-indexing
        return max(2, min(k, len(S) // 2))

    def _spectral_cluster(self, S: np.ndarray, k: int) -> np.ndarray:
        """
        Spectral clustering on quantum walk similarity matrix.
        Uses top-k eigenvectors of normalized similarity Laplacian.
        """
        n = S.shape[0]
        # Row-normalize similarity → affinity
        D_inv_sqrt = np.diag(1.0 / (np.sqrt(np.sum(S, axis=1)) + 1e-10))
        L_norm = D_inv_sqrt @ S @ D_inv_sqrt

        # Top k eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
        idx = np.argsort(eigenvalues)[::-1]
        V = eigenvectors[:, idx[:k]]  # n × k matrix

        # Normalize rows
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        V_norm = V / (norms + 1e-10)

        # k-means on eigenvectors
        labels = self._kmeans(V_norm, k)
        return labels

    def _kmeans(self, X: np.ndarray, k: int, max_iter: int = 100) -> np.ndarray:
        """Simple k-means with k-means++ initialization."""
        n = X.shape[0]
        # k-means++ init
        centers = [X[np.random.randint(n)]]
        for _ in range(k - 1):
            dists = np.min([np.sum((X - c)**2, axis=1) for c in centers], axis=0)
            probs = dists / (dists.sum() + 1e-10)
            centers.append(X[np.random.choice(n, p=probs)])
        centers = np.array(centers)

        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            # Assign
            dists = np.array([np.sum((X - c)**2, axis=1) for c in centers])
            new_labels = np.argmin(dists, axis=0)
            if np.all(new_labels == labels):
                break
            labels = new_labels
            # Update centers
            for j in range(k):
                mask = labels == j
                if mask.any():
                    centers[j] = X[mask].mean(axis=0)

        return labels

    # ── Quantum walk evolution tracking ────────────────────────────────────

    def evolve_walk(self, adj: np.ndarray, start_node: int,
                    t: float) -> np.ndarray:
        """
        Evolve quantum walk from start_node for time t.
        Returns probability distribution over all nodes.
        """
        L = self._graph_laplacian(adj)
        U = self._walk_operator(L, t)
        psi0 = self._node_state(start_node, adj.shape[0])
        psi_t = U @ psi0
        return np.abs(psi_t) ** 2

    def get_walk_animation_data(self, adj: np.ndarray, start_node: int,
                                 n_frames: int = 50) -> List[np.ndarray]:
        """
        Generate walk probability snapshots for visualization.
        Returns list of probability vectors over time.
        """
        times = np.linspace(0, 6.0, n_frames)
        L = self._graph_laplacian(adj)
        psi0 = self._node_state(start_node, adj.shape[0])
        frames = []
        for t in times:
            U = self._walk_operator(L, t)
            psi_t = U @ psi0
            frames.append(np.abs(psi_t) ** 2)
        return frames

    # ── Main API ────────────────────────────────────────────────────────────

    def fit(self, adj: np.ndarray, verbose: bool = False) -> "QuantumWalkCommunityDetection":
        """Detect communities using quantum walk similarity."""
        n = adj.shape[0]
        self.n_nodes_ = n

        if verbose:
            print(f"Quantum Walk CD: {n} nodes, "
                  f"{len(self.t_values)} time steps")

        # Compute both similarity measures and average
        S_prob = self._quantum_walk_similarity(adj)
        S_int  = self._quantum_interference_matrix(adj)
        self.similarity_matrix_ = 0.6 * S_prob + 0.4 * S_int

        # Determine number of clusters
        k = self.n_clusters or self._estimate_n_clusters(self.similarity_matrix_)
        self.k_ = k

        if verbose:
            print(f"Estimated {k} communities")

        # Spectral cluster on similarity
        self.labels_ = self._spectral_cluster(self.similarity_matrix_, k)
        self.modularity_ = self._compute_modularity(adj, self.labels_)

        if verbose:
            print(f"Modularity Q = {self.modularity_:.4f}")

        return self

    def _compute_modularity(self, adj: np.ndarray, labels: np.ndarray) -> float:
        degree = np.sum(adj, axis=1)
        m = np.sum(adj) / 2.0
        n = len(labels)
        Q = 0.0
        for i in range(n):
            for j in range(n):
                if labels[i] == labels[j]:
                    Q += adj[i, j] - degree[i] * degree[j] / (2 * m + 1e-10)
        return Q / (2 * m + 1e-10)

    def get_circuit_stats(self) -> Dict[str, Any]:
        return {
            "algorithm": "Quantum Walk",
            "n_time_steps": len(self.t_values),
            "time_range": f"[{min(self.t_values)}, {max(self.t_values)}]",
            "n_nodes": getattr(self, "n_nodes_", "N/A"),
            "similarity": "QW probability + interference",
        }
