"""
Benchmark Suite and Graph Generators
=====================================
Standard test graphs for community detection benchmarking.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
import time


# ─── Graph Generators ─────────────────────────────────────────────────────────

def stochastic_block_model(community_sizes: List[int],
                            p_in: float = 0.7,
                            p_out: float = 0.1,
                            weighted: bool = False,
                            seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stochastic Block Model (SBM) — ground-truth community graph.

    Parameters
    ----------
    community_sizes : list of ints
        Number of nodes per community.
    p_in : float
        Edge probability within communities.
    p_out : float
        Edge probability between communities.
    weighted : bool
        If True, weights are w ~ U[0.5, 1.5] intra, U[0.1, 0.5] inter.

    Returns
    -------
    adj : ndarray, adjacency matrix
    labels : ndarray, ground-truth community labels
    """
    if seed is not None:
        np.random.seed(seed)

    n = sum(community_sizes)
    adj = np.zeros((n, n))
    labels = np.zeros(n, dtype=int)

    offset = 0
    for comm_idx, size in enumerate(community_sizes):
        labels[offset:offset+size] = comm_idx
        offset += size

    for i in range(n):
        for j in range(i+1, n):
            same_comm = labels[i] == labels[j]
            p = p_in if same_comm else p_out
            if np.random.rand() < p:
                if weighted:
                    w = np.random.uniform(0.5, 1.5) if same_comm else np.random.uniform(0.1, 0.5)
                else:
                    w = 1.0
                adj[i, j] = w
                adj[j, i] = w

    return adj, labels


def karate_club_graph() -> Tuple[np.ndarray, np.ndarray]:
    """
    Zachary's Karate Club (34 nodes, 78 edges).
    Classic community detection benchmark.
    Ground truth: two factions from the 1977 social network study.
    """
    # Adjacency list (Zachary 1977)
    edges = [
        (1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(1,9),(1,11),(1,12),(1,13),(1,14),
        (1,18),(1,20),(1,22),(1,32),(2,3),(2,4),(2,8),(2,14),(2,18),(2,20),(2,22),
        (2,31),(3,4),(3,8),(3,9),(3,10),(3,14),(3,28),(3,29),(3,33),(4,8),(4,13),(4,14),
        (5,7),(5,11),(6,7),(6,11),(6,17),(7,17),(9,31),(9,33),(9,34),(10,34),(14,34),
        (15,33),(15,34),(16,33),(16,34),(19,33),(19,34),(20,34),(21,33),(21,34),
        (23,33),(23,34),(24,26),(24,28),(24,30),(24,33),(24,34),(25,26),(25,28),
        (25,32),(26,32),(27,30),(27,34),(28,34),(29,32),(29,34),(30,33),(30,34),
        (31,33),(31,34),(32,33),(32,34),(33,34)
    ]
    n = 34
    adj = np.zeros((n, n))
    for i, j in edges:
        adj[i-1, j-1] = 1.0
        adj[j-1, i-1] = 1.0

    # Ground truth: node 1 (idx 0) faction vs node 34 (idx 33) faction
    labels = np.array([
        0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1
    ])
    return adj, labels


def planted_partition(n: int = 12, k: int = 3,
                       p_in: float = 0.8, p_out: float = 0.05,
                       seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Balanced planted partition model."""
    sizes = [n // k] * k
    sizes[0] += n - sum(sizes)  # Handle remainder
    return stochastic_block_model(sizes, p_in, p_out, seed=seed)


def ring_of_cliques(clique_size: int = 4, n_cliques: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ring of cliques: k cliques connected in a ring.
    Very clear community structure (good for testing).
    """
    n = clique_size * n_cliques
    adj = np.zeros((n, n))
    labels = np.zeros(n, dtype=int)

    for c in range(n_cliques):
        # Fully connect within clique
        start = c * clique_size
        for i in range(start, start + clique_size):
            for j in range(start, start + clique_size):
                if i != j:
                    adj[i, j] = 1.0
            labels[i] = c

        # Connect to next clique (ring) via single bridge edge
        next_c = (c + 1) % n_cliques
        next_start = next_c * clique_size
        adj[start, next_start] = 0.3
        adj[next_start, start] = 0.3

    return adj, labels


def small_test_graph(n: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Small graph suitable for QAOA and exact quantum simulation."""
    if n == 8:
        # Two groups of 4 with weak inter-group connections
        adj = np.array([
            [0,1,1,1, 0,0,0,0],
            [1,0,1,1, 0,0,0,1],
            [1,1,0,1, 0,0,1,0],
            [1,1,1,0, 0,1,0,0],
            [0,0,0,0, 0,1,1,1],
            [0,0,0,1, 1,0,1,1],
            [0,0,1,0, 1,1,0,1],
            [0,1,0,0, 1,1,1,0],
        ], dtype=float)
        # Weak bridge
        adj[1,4] = adj[4,1] = 0.2
        adj[2,5] = adj[5,2] = 0.2
        labels = np.array([0,0,0,0,1,1,1,1])
        return adj, labels
    else:
        return planted_partition(n, k=2, p_in=0.75, p_out=0.1, seed=42)


# ─── Evaluation Metrics ────────────────────────────────────────────────────────

def normalized_mutual_information(labels_true: np.ndarray,
                                   labels_pred: np.ndarray) -> float:
    """
    Normalized Mutual Information (NMI) between two label assignments.
    NMI ∈ [0, 1]; 1 = perfect match.
    """
    n = len(labels_true)
    classes_true = np.unique(labels_true)
    classes_pred = np.unique(labels_pred)

    # Contingency matrix
    cont = np.zeros((len(classes_true), len(classes_pred)))
    for i, c1 in enumerate(classes_true):
        for j, c2 in enumerate(classes_pred):
            cont[i, j] = np.sum((labels_true == c1) & (labels_pred == c2))

    # Mutual information
    p_true = np.sum(cont, axis=1) / n
    p_pred = np.sum(cont, axis=0) / n
    p_joint = cont / n

    MI = 0.0
    for i in range(len(classes_true)):
        for j in range(len(classes_pred)):
            if p_joint[i, j] > 1e-10:
                MI += p_joint[i, j] * np.log(
                    p_joint[i, j] / (p_true[i] * p_pred[j] + 1e-10)
                )

    # Entropies
    H_true = -np.sum(p_true[p_true > 1e-10] * np.log(p_true[p_true > 1e-10]))
    H_pred = -np.sum(p_pred[p_pred > 1e-10] * np.log(p_pred[p_pred > 1e-10]))

    if H_true + H_pred < 1e-10:
        return 1.0
    return 2 * MI / (H_true + H_pred)


def modularity_score(adj: np.ndarray, labels: np.ndarray) -> float:
    """Newman-Girvan modularity Q."""
    degree = np.sum(adj, axis=1)
    m = np.sum(adj) / 2.0
    if m < 1e-10:
        return 0.0
    Q = 0.0
    n = len(labels)
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                Q += adj[i, j] - degree[i] * degree[j] / (2 * m)
    return Q / (2 * m)


def conductance(adj: np.ndarray, labels: np.ndarray) -> float:
    """
    Average conductance across communities.
    φ(S) = cut(S, V\S) / min(vol(S), vol(V\S))
    Lower is better (≈0 for perfect communities).
    """
    unique = np.unique(labels)
    conductances = []
    for lab in unique:
        S = np.where(labels == lab)[0]
        S_bar = np.where(labels != lab)[0]
        if len(S) == 0 or len(S_bar) == 0:
            continue
        cut = np.sum(adj[np.ix_(S, S_bar)])
        vol_S = np.sum(adj[S, :])
        vol_bar = np.sum(adj[S_bar, :])
        phi = cut / (min(vol_S, vol_bar) + 1e-10)
        conductances.append(phi)
    return float(np.mean(conductances)) if conductances else 0.0


def accuracy_best_permutation(labels_true: np.ndarray,
                               labels_pred: np.ndarray) -> float:
    """
    Best-permutation accuracy: try all label permutations and return max accuracy.
    Handles arbitrary label indexing.
    """
    from itertools import permutations
    classes = np.unique(labels_true)
    best_acc = 0.0
    for perm in permutations(range(len(classes))):
        mapping = {classes[i]: perm[i] for i in range(len(classes))}
        remapped = np.array([mapping.get(l, l) for l in labels_pred])
        acc = float(np.mean(remapped == labels_true))
        if acc > best_acc:
            best_acc = acc
    return best_acc


# ─── Benchmark Runner ──────────────────────────────────────────────────────────

def run_benchmark(algorithms: dict, adj: np.ndarray,
                  labels_true: np.ndarray,
                  verbose: bool = True) -> Dict[str, Dict]:
    """
    Run all algorithms and compute evaluation metrics.

    Parameters
    ----------
    algorithms : dict mapping name → algorithm instance (with .fit() method)
    adj : adjacency matrix
    labels_true : ground-truth labels
    verbose : print results

    Returns
    -------
    results : dict mapping name → metrics dict
    """
    results = {}

    for name, algo in algorithms.items():
        if verbose:
            print(f"\n{'─'*50}")
            print(f"Running {name}...")

        t0 = time.time()
        try:
            algo.fit(adj, verbose=verbose)
            elapsed = time.time() - t0

            labels_pred = algo.labels_

            metrics = {
                "labels": labels_pred,
                "modularity": modularity_score(adj, labels_pred),
                "nmi": normalized_mutual_information(labels_true, labels_pred),
                "conductance": conductance(adj, labels_pred),
                "accuracy": accuracy_best_permutation(labels_true, labels_pred),
                "time_sec": elapsed,
                "circuit_stats": algo.get_circuit_stats() if hasattr(algo, "get_circuit_stats") else {},
                "success": True,
            }

        except Exception as e:
            elapsed = time.time() - t0
            metrics = {
                "labels": np.zeros(len(labels_true), dtype=int),
                "modularity": 0.0,
                "nmi": 0.0,
                "conductance": 1.0,
                "accuracy": 0.0,
                "time_sec": elapsed,
                "error": str(e),
                "success": False,
            }
            if verbose:
                print(f"  ERROR: {e}")

        results[name] = metrics

        if verbose and metrics["success"]:
            print(f"  Modularity Q   = {metrics['modularity']:.4f}")
            print(f"  NMI            = {metrics['nmi']:.4f}")
            print(f"  Accuracy       = {metrics['accuracy']:.4f}")
            print(f"  Conductance    = {metrics['conductance']:.4f}")
            print(f"  Runtime        = {elapsed:.2f}s")

    return results
