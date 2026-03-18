"""
Quantum Community Detection - Quantum Simulator Backend
Pure numpy-based statevector simulator for small graphs (up to ~16 nodes/qubits).
"""

import numpy as np
from typing import Optional
from itertools import product


# ─── Pauli matrices ────────────────────────────────────────────────────────────
I2 = np.eye(2, dtype=complex)
X  = np.array([[0,1],[1,0]], dtype=complex)
Y  = np.array([[0,-1j],[1j,0]], dtype=complex)
Z  = np.array([[1,0],[0,-1]], dtype=complex)
H  = np.array([[1,1],[1,-1]], dtype=complex) / np.sqrt(2)


def kron_n(*ops):
    """Tensor (Kronecker) product of multiple operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def pauli_string(op: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Place single-qubit operator `op` on qubit `qubit` of n-qubit system."""
    ops = [I2] * n_qubits
    ops[qubit] = op
    return kron_n(*ops)


def zz_term(i: int, j: int, n: int) -> np.ndarray:
    """ZZ interaction on qubits i,j in n-qubit system."""
    ops = [I2] * n
    ops[i] = Z
    ops[j] = Z
    return kron_n(*ops)


def xx_term(i: int, j: int, n: int) -> np.ndarray:
    """XX interaction on qubits i,j."""
    ops = [I2] * n
    ops[i] = X
    ops[j] = X
    return kron_n(*ops)


# ─── Statevector simulator ─────────────────────────────────────────────────────
class QuantumState:
    """Statevector representation of n-qubit register."""

    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.dim = 2 ** n_qubits
        self.state = np.zeros(self.dim, dtype=complex)
        self.state[0] = 1.0  # |00...0⟩

    def apply(self, U: np.ndarray) -> "QuantumState":
        """Apply unitary U to full state."""
        self.state = U @ self.state
        return self

    def apply_gate(self, gate: np.ndarray, qubit: int) -> "QuantumState":
        """Apply single-qubit gate to specified qubit."""
        U = pauli_string(gate, qubit, self.n)
        return self.apply(U)

    def apply_unitary(self, H_op: np.ndarray, t: float) -> "QuantumState":
        """Apply e^{-iHt} via matrix exponentiation."""
        from scipy.linalg import expm
        U = expm(-1j * t * H_op)
        return self.apply(U)

    def hadamard_all(self) -> "QuantumState":
        """Apply H to every qubit."""
        H_full = kron_n(*[H] * self.n)
        return self.apply(H_full)

    def expectation(self, obs: np.ndarray) -> float:
        """⟨ψ|O|ψ⟩"""
        return float(np.real(self.state.conj() @ obs @ self.state))

    def probabilities(self) -> np.ndarray:
        """Measurement probability distribution."""
        return np.abs(self.state) ** 2

    def sample(self, n_shots: int = 1024) -> np.ndarray:
        """Sample bitstrings from measurement."""
        probs = self.probabilities()
        indices = np.random.choice(self.dim, size=n_shots, p=probs)
        return indices

    def reduced_density_matrix(self, subsystem_A: list) -> np.ndarray:
        """
        Compute reduced density matrix of subsystem A via partial trace.
        subsystem_A: list of qubit indices in subsystem A
        """
        n = self.n
        subsystem_B = [i for i in range(n) if i not in subsystem_A]
        dim_A = 2 ** len(subsystem_A)
        dim_B = 2 ** len(subsystem_B)

        # Reshape state into tensor with one index per qubit
        psi = self.state.reshape([2] * n)

        # Reorder axes: A qubits first, then B qubits
        order = subsystem_A + subsystem_B
        psi = np.transpose(psi, order)
        psi = psi.reshape(dim_A, dim_B)

        # rho_A = Tr_B[|ψ⟩⟨ψ|]
        rho_A = psi @ psi.conj().T
        return rho_A

    def entanglement_entropy(self, subsystem_A: list) -> float:
        """
        Von Neumann entanglement entropy S(A) = -Tr(rho_A log rho_A)
        via Schmidt decomposition.
        """
        rho_A = self.reduced_density_matrix(subsystem_A)
        eigenvalues = np.linalg.eigvalsh(rho_A)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))

    def schmidt_values(self, subsystem_A: list) -> np.ndarray:
        """Schmidt coefficients (singular values of bipartite split)."""
        n = self.n
        subsystem_B = [i for i in range(n) if i not in subsystem_A]
        dim_A = 2 ** len(subsystem_A)
        dim_B = 2 ** len(subsystem_B)
        order = subsystem_A + subsystem_B
        psi = np.transpose(self.state.reshape([2] * n), order).reshape(dim_A, dim_B)
        _, s, _ = np.linalg.svd(psi)
        return s

    def clone(self) -> "QuantumState":
        qs = QuantumState(self.n)
        qs.state = self.state.copy()
        return qs


# ─── Graph → Hamiltonian encoding ─────────────────────────────────────────────
def graph_to_ising_hamiltonian(adj: np.ndarray) -> np.ndarray:
    """
    Encode graph as Ising cost Hamiltonian for MaxCut / community detection.
    H_C = (1/2) * sum_{(i,j) in E} w_{ij} * (I - Z_i Z_j)
    Minimizing H_C ↔ finding balanced cuts with high modularity.
    """
    n = adj.shape[0]
    H_C = np.zeros((2**n, 2**n), dtype=complex)
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] != 0:
                w = adj[i, j]
                H_C += 0.5 * w * (kron_n(*[I2]*n) - zz_term(i, j, n))
    return H_C


def graph_to_laplacian_hamiltonian(adj: np.ndarray) -> np.ndarray:
    """
    Encode normalized graph Laplacian as quantum Hamiltonian.
    H_L = D - A  (degree matrix minus adjacency)
    Used in quantum walk and VQE spectral methods.
    """
    n = adj.shape[0]
    degree = np.sum(adj, axis=1)
    H_L = np.zeros((2**n, 2**n), dtype=complex)
    for i in range(n):
        H_L += degree[i] * pauli_string(Z, i, n)  # diagonal degree terms
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] != 0:
                w = adj[i, j]
                # Off-diagonal Laplacian terms via XX+YY hopping
                H_L -= w * (xx_term(i, j, n) + kron_n(*([I2 if k not in (i,j) else (Y if k==i else Y) for k in range(n)])))
    # Simpler: just embed classical Laplacian into top-left block
    L = np.diag(degree) - adj
    # Embed into 2^n space (acts on first n basis states)
    H_embed = np.zeros((2**n, 2**n), dtype=complex)
    H_embed[:n, :n] = L
    return H_embed


def modularity_hamiltonian(adj: np.ndarray) -> np.ndarray:
    """
    Newman-Girvan modularity matrix as Hamiltonian.
    B_{ij} = A_{ij} - k_i k_j / (2m)
    H_Q = -sum_{ij} B_{ij} s_i s_j  (spin formulation)
    """
    n = adj.shape[0]
    m = np.sum(adj) / 2.0
    degree = np.sum(adj, axis=1)
    B = adj - np.outer(degree, degree) / (2 * m + 1e-10)
    H_Q = np.zeros((2**n, 2**n), dtype=complex)
    for i in range(n):
        for j in range(i+1, n):
            H_Q -= B[i, j] * zz_term(i, j, n)
    return H_Q
