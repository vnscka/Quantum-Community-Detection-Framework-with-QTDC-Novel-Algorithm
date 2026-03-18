"""
Algorithm 1: QAOA-based Community Detection
============================================
Uses the Quantum Approximate Optimization Algorithm (Farhi et al. 2014)
to maximize Newman-Girvan modularity.

The QAOA circuit alternates between:
  - Cost unitary:   U_C(γ) = e^{-iγ H_C}  (problem Hamiltonian)
  - Mixer unitary:  U_B(β)  = e^{-iβ H_B}  (transverse field mixer)

After p layers, measure bitstrings and assign communities by spin direction.

References:
  - Farhi, Goldstone, Gutmann (2014). A Quantum Approximate Optimization Algorithm.
  - Shaydulin et al. (2019). A Hybrid Approach for Solving Optimization Problems
    on Small Quantum Computers.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, List, Dict, Any

from ..utils.quantum_sim import (
    QuantumState, modularity_hamiltonian, pauli_string,
    kron_n, I2, X, Z
)


class QAOACommunityDetection:
    """
    QAOA community detection via modularity maximization.

    Parameters
    ----------
    n_layers : int
        Number of QAOA layers p (depth). More layers → better approximation.
    n_shots : int
        Number of measurement shots for expectation estimation.
    optimizer : str
        Classical optimizer for variational parameters.
    """

    def __init__(self, n_layers: int = 2, n_shots: int = 2048,
                 optimizer: str = "COBYLA"):
        self.p = n_layers
        self.n_shots = n_shots
        self.optimizer = optimizer
        self.history: List[float] = []
        self.optimal_params: Optional[np.ndarray] = None
        self.n_qubits: int = 0

    # ── Circuit construction ────────────────────────────────────────────────

    def _mixer_hamiltonian(self, n: int) -> np.ndarray:
        """Transverse field mixer H_B = sum_i X_i"""
        H_B = np.zeros((2**n, 2**n), dtype=complex)
        for i in range(n):
            H_B += pauli_string(X, i, n)
        return H_B

    def _cost_unitary(self, H_C: np.ndarray, gamma: float) -> np.ndarray:
        """e^{-i gamma H_C} via eigendecomposition for efficiency."""
        from scipy.linalg import expm
        return expm(-1j * gamma * H_C)

    def _mixer_unitary(self, n: int, beta: float) -> np.ndarray:
        """e^{-i beta H_B} = tensor product of Rx(2β) gates."""
        # Single-qubit Rx
        Rx = np.cos(beta) * I2 - 1j * np.sin(beta) * X
        result = Rx
        for _ in range(n - 1):
            result = np.kron(result, Rx)
        return result

    def _run_circuit(self, params: np.ndarray, H_C: np.ndarray,
                     n: int) -> QuantumState:
        """Execute QAOA circuit with given parameters."""
        gammas = params[:self.p]
        betas  = params[self.p:]

        H_B = self._mixer_hamiltonian(n)
        qs = QuantumState(n)
        qs.hadamard_all()  # Initialize |+⟩^n

        for layer in range(self.p):
            # Cost layer
            U_C = self._cost_unitary(H_C, gammas[layer])
            qs.apply(U_C)
            # Mixer layer
            U_B = self._mixer_unitary(n, betas[layer])
            qs.apply(U_B)

        return qs

    # ── Objective function ──────────────────────────────────────────────────

    def _objective(self, params: np.ndarray, H_C: np.ndarray, n: int) -> float:
        """Negative expected modularity (minimize = maximize modularity)."""
        qs = self._run_circuit(params, H_C, n)
        exp_val = qs.expectation(H_C)
        self.history.append(-exp_val)
        return -exp_val  # Minimize negative = maximize

    # ── Community extraction ────────────────────────────────────────────────

    def _bitstring_to_communities(self, idx: int, n: int) -> np.ndarray:
        """Map integer index to community labels (0 or 1 per node)."""
        return np.array([(idx >> (n - 1 - i)) & 1 for i in range(n)])

    def _extract_communities(self, qs: QuantumState, n: int,
                              adj: np.ndarray) -> np.ndarray:
        """
        Sample measurement outcomes and pick best community assignment
        by majority vote weighted by probability.
        """
        probs = qs.probabilities()
        # Score each bitstring by modularity
        degree = np.sum(adj, axis=1)
        m = np.sum(adj) / 2.0

        best_Q = -np.inf
        best_labels = np.zeros(n, dtype=int)

        # Evaluate top-k most probable bitstrings
        top_k = np.argsort(probs)[-min(50, 2**n):]
        for idx in top_k:
            if probs[idx] < 1e-6:
                continue
            labels = self._bitstring_to_communities(idx, n)
            Q = self._compute_modularity(adj, labels, m)
            if Q > best_Q:
                best_Q = Q
                best_labels = labels

        return best_labels

    def _compute_modularity(self, adj: np.ndarray, labels: np.ndarray,
                             m: float) -> float:
        """Newman-Girvan modularity Q."""
        degree = np.sum(adj, axis=1)
        n = len(labels)
        Q = 0.0
        for i in range(n):
            for j in range(n):
                if labels[i] == labels[j]:
                    B_ij = adj[i, j] - degree[i] * degree[j] / (2 * m + 1e-10)
                    Q += B_ij
        return Q / (2 * m + 1e-10)

    # ── Main API ────────────────────────────────────────────────────────────

    def fit(self, adj: np.ndarray, verbose: bool = False) -> "QAOACommunityDetection":
        """
        Run QAOA on adjacency matrix adj.

        Returns self for chaining. Access .labels_ for results.
        """
        n = adj.shape[0]
        self.n_qubits = n
        self.history = []

        H_C = modularity_hamiltonian(adj)

        # Initialize parameters: gammas in [0, π], betas in [0, π/2]
        params0 = np.concatenate([
            np.random.uniform(0, np.pi, self.p),
            np.random.uniform(0, np.pi / 2, self.p)
        ])

        if verbose:
            print(f"QAOA: {n} qubits, {self.p} layers, dim={2**n}")
            print(f"Optimizing {2*self.p} parameters with {self.optimizer}...")

        result = minimize(
            self._objective,
            params0,
            args=(H_C, n),
            method=self.optimizer,
            options={"maxiter": 200, "rhobeg": 0.5}
        )

        self.optimal_params = result.x
        self.optimal_energy = -result.fun

        # Final circuit run with optimal params
        qs = self._run_circuit(result.x, H_C, n)
        self.labels_ = self._extract_communities(qs, n, adj)
        self.final_state = qs
        self.modularity_ = self._compute_modularity(adj, self.labels_,
                                                     np.sum(adj) / 2.0)

        if verbose:
            print(f"Optimal modularity Q = {self.modularity_:.4f}")
            print(f"Community labels: {self.labels_}")

        return self

    def get_circuit_stats(self) -> Dict[str, Any]:
        return {
            "algorithm": "QAOA",
            "n_layers": self.p,
            "n_params": 2 * self.p,
            "n_qubits": self.n_qubits,
            "circuit_depth": 2 * self.p + 1,  # +1 for Hadamard init
            "optimizer": self.optimizer,
        }
